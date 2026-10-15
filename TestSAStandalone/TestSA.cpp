#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// =============================================================================
// AdaptiveCpp atomic-capability + host/shared USM InterCoOp validation suite
// =============================================================================
//
// This standalone program validates the atomic-capability patches at three
// independent layers:
//
//   1. Atomic semantic capability:
//      * atomic_memory_order_capabilities
//      * atomic_memory_scope_capabilities
//      * context intersection semantics
//
//   2. USM allocation capability (kept deliberately independent of scope):
//      * usm_host_allocations / usm_atomic_host_allocations
//      * usm_shared_allocations / usm_atomic_shared_allocations
//      * host-USM and shared-USM allocation/access smoke tests
//
//   3. Executable atomic semantics:
//      * device/work-group atomics
//      * system-scope RMW sequences independently on host USM and shared USM
//      * CPU -> device release/acquire publication
//      * device -> CPU release/acquire publication
//      * CPU/device fetch_add on the same uint64_t
//      * CPU/device fetch_or + fetch_and on the same uint64_t
//      * CPU/device compare-exchange races
//      * explicit system-scope atomic_fence smoke tests
//
// IMPORTANT:
//   memory_scope::system is NOT inferred from either USM allocation aspect.
//   A path qualifies only when BOTH the semantic atomic contract and that
//   allocation kind's atomic-USM aspect are advertised.
//
//   Unsupported capability => corresponding execution test SKIPs.
//   Advertised capability that fails execution => FAIL.
//
// Suggested builds:
//   acpp -std=c++20 --acpp-targets=generic TestSA.cpp -o TestSA
//   acpp -std=c++20 --acpp-targets=cuda:sm_XX TestSA.cpp -o TestSA
//   acpp -std=c++20 --acpp-targets=hip:gfxXXXX TestSA.cpp -o TestSA
//
// Useful invocations:
//   ./TestSA
//   ./TestSA --stress=500000
//   ./TestSA --require-intercoop
//   ./TestSA --require-gpu-intercoop
//   ./TestSA --device=0
//
// =============================================================================

namespace {

using U32 = std::uint32_t;
using U64 = std::uint64_t;

constexpr std::chrono::milliseconds kHostPollTimeout{5000};
constexpr std::size_t kDefaultStress = 100'000;

struct Options {
  std::size_t stress = kDefaultStress;
  bool require_intercoop = false;
  bool require_gpu_intercoop = false;
  std::optional<std::size_t> device_index;
};

struct TestState {
  std::size_t passed = 0;
  std::size_t failed = 0;
  std::size_t skipped = 0;

  void Pass(std::string_view message) {
    ++passed;
    std::cout << "  [PASS] " << message << '\n';
  }

  void Fail(std::string_view message) {
    ++failed;
    std::cerr << "  [FAIL] " << message << '\n';
  }

  void Skip(std::string_view message) {
    ++skipped;
    std::cout << "  [SKIP] " << message << '\n';
  }

  void Check(bool condition, std::string_view message) {
    if(condition)
      Pass(message);
    else
      Fail(message);
  }
};

const char* OrderName(sycl::memory_order order) {
  switch(order) {
  case sycl::memory_order::relaxed: return "relaxed";
  case sycl::memory_order::acquire: return "acquire";
  case sycl::memory_order::release: return "release";
  case sycl::memory_order::acq_rel: return "acq_rel";
  case sycl::memory_order::seq_cst: return "seq_cst";
  }
  return "unknown";
}

const char* ScopeName(sycl::memory_scope scope) {
  switch(scope) {
  case sycl::memory_scope::work_item: return "work_item";
  case sycl::memory_scope::sub_group: return "sub_group";
  case sycl::memory_scope::work_group: return "work_group";
  case sycl::memory_scope::device: return "device";
  case sycl::memory_scope::system: return "system";
  }
  return "unknown";
}

const char* BackendName(sycl::backend backend) {
  switch(backend) {
  case sycl::backend::cuda: return "cuda";
  case sycl::backend::hip: return "hip";
  case sycl::backend::level_zero: return "level_zero";
  case sycl::backend::ocl: return "opencl";
  case sycl::backend::omp: return "openmp";
  case sycl::backend::metal: return "metal";
  case sycl::backend::vk: return "vulkan";
  }
  return "unknown";
}

template<class T>
bool Contains(const std::vector<T>& values, T value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

template<class T>
bool HasDuplicates(const std::vector<T>& values) {
  for(std::size_t i = 0; i < values.size(); ++i)
    for(std::size_t j = i + 1; j < values.size(); ++j)
      if(values[i] == values[j])
        return true;
  return false;
}

void PrintOrders(const std::vector<sycl::memory_order>& orders) {
  std::cout << "{ ";
  for(const auto order : orders)
    std::cout << OrderName(order) << ' ';
  std::cout << '}';
}

void PrintScopes(const std::vector<sycl::memory_scope>& scopes) {
  std::cout << "{ ";
  for(const auto scope : scopes)
    std::cout << ScopeName(scope) << ' ';
  std::cout << '}';
}

std::vector<sycl::memory_order>
ManualOrderIntersection(const std::vector<sycl::device>& devices) {
  const std::array canonical{
      sycl::memory_order::relaxed,
      sycl::memory_order::acquire,
      sycl::memory_order::release,
      sycl::memory_order::acq_rel,
      sycl::memory_order::seq_cst};

  std::vector<sycl::memory_order> result;
  for(const auto order : canonical) {
    bool all = true;
    for(const auto& device : devices) {
      const auto caps = device.get_info<
          sycl::info::device::atomic_memory_order_capabilities>();
      if(!Contains(caps, order)) {
        all = false;
        break;
      }
    }
    if(all)
      result.push_back(order);
  }
  return result;
}

std::vector<sycl::memory_scope>
ManualScopeIntersection(const std::vector<sycl::device>& devices) {
  const std::array canonical{
      sycl::memory_scope::sub_group,
      sycl::memory_scope::work_group,
      sycl::memory_scope::device,
      sycl::memory_scope::system};

  std::vector<sycl::memory_scope> result;
  for(const auto scope : canonical) {
    bool all = true;
    for(const auto& device : devices) {
      const auto caps = device.get_info<
          sycl::info::device::atomic_memory_scope_capabilities>();
      if(!Contains(caps, scope)) {
        all = false;
        break;
      }
    }
    if(all)
      result.push_back(scope);
  }
  return result;
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for(int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if(arg == "--require-intercoop") {
      options.require_intercoop = true;
    } else if(arg == "--require-gpu-intercoop") {
      options.require_gpu_intercoop = true;
    } else if(arg.rfind("--stress=", 0) == 0) {
      options.stress = static_cast<std::size_t>(
          std::stoull(arg.substr(std::string{"--stress="}.size())));
      if(options.stress == 0)
        throw std::runtime_error("--stress must be greater than zero");
    } else if(arg.rfind("--device=", 0) == 0) {
      options.device_index = static_cast<std::size_t>(
          std::stoull(arg.substr(std::string{"--device="}.size())));
    } else if(arg == "--help") {
      std::cout
          << "Usage: TestSA [options]\n"
          << "  --stress=N                 Atomic increments per CPU/device side\n"
          << "  --device=N                 Test only discovered device N\n"
          << "  --require-intercoop        Fail if no device qualifies for uint64 InterCoOp\n"
          << "  --require-gpu-intercoop    Fail if no GPU qualifies for uint64 InterCoOp\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return options;
}

enum class InteropUsmKind { Host, Shared };

const char* InteropUsmName(InteropUsmKind kind) {
  return kind == InteropUsmKind::Host ? "host USM" : "shared USM";
}

const char* InteropUsmShortName(InteropUsmKind kind) {
  return kind == InteropUsmKind::Host ? "host" : "shared";
}

template<class T>
class InteropUsmArray {
public:
  InteropUsmArray(sycl::queue& queue, InteropUsmKind kind,
                  std::size_t count, std::size_t alignment)
      : queue_{&queue}, kind_{kind}, count_{count} {
    if(kind == InteropUsmKind::Host)
      ptr_ = sycl::aligned_alloc_host<T>(alignment, count, queue);
    else
      ptr_ = sycl::aligned_alloc_shared<T>(alignment, count, queue);

    if(!ptr_) {
      throw std::runtime_error(
          std::string{"sycl::aligned_alloc_"} + InteropUsmShortName(kind) +
          " returned nullptr");
    }
  }

  ~InteropUsmArray() {
    if(ptr_)
      sycl::free(ptr_, *queue_);
  }

  InteropUsmArray(const InteropUsmArray&) = delete;
  InteropUsmArray& operator=(const InteropUsmArray&) = delete;

  T* get() noexcept { return ptr_; }
  const T* get() const noexcept { return ptr_; }
  T& operator[](std::size_t i) noexcept { return ptr_[i]; }
  const T& operator[](std::size_t i) const noexcept { return ptr_[i]; }
  std::size_t size() const noexcept { return count_; }
  InteropUsmKind kind() const noexcept { return kind_; }

private:
  sycl::queue* queue_ = nullptr;
  InteropUsmKind kind_ = InteropUsmKind::Shared;
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

template<class T>
class DeviceUsmArray {
public:
  DeviceUsmArray(sycl::queue& queue, std::size_t count)
      : queue_{&queue}, count_{count} {
    ptr_ = sycl::malloc_device<T>(count, queue);
    if(!ptr_)
      throw std::runtime_error("sycl::malloc_device returned nullptr");
  }

  ~DeviceUsmArray() {
    if(ptr_)
      sycl::free(ptr_, *queue_);
  }

  DeviceUsmArray(const DeviceUsmArray&) = delete;
  DeviceUsmArray& operator=(const DeviceUsmArray&) = delete;

  T* get() noexcept { return ptr_; }
  const T* get() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return count_; }

private:
  sycl::queue* queue_ = nullptr;
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

template<class T, sycl::memory_scope Scope>
using GlobalAtomic = sycl::atomic_ref<
    T,
    sycl::memory_order::relaxed,
    Scope,
    sycl::access::address_space::global_space>;

template<class T>
constexpr std::size_t SharedAtomicAlignment() {
  using SysAtomic = GlobalAtomic<T, sycl::memory_scope::system>;
  return std::max({alignof(T),
                   SysAtomic::required_alignment,
                   std::atomic_ref<T>::required_alignment});
}


bool HasUsmAllocation(const sycl::device& device, InteropUsmKind kind) {
  return kind == InteropUsmKind::Host
      ? device.has(sycl::aspect::usm_host_allocations)
      : device.has(sycl::aspect::usm_shared_allocations);
}

bool HasAtomicUsmAllocation(const sycl::device& device, InteropUsmKind kind) {
  return kind == InteropUsmKind::Host
      ? device.has(sycl::aspect::usm_atomic_host_allocations)
      : device.has(sycl::aspect::usm_atomic_shared_allocations);
}

bool HasSystemAcqRelContract(const sycl::device& device) {
  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();

  return Contains(orders, sycl::memory_order::acquire) &&
         Contains(orders, sycl::memory_order::release) &&
         Contains(orders, sycl::memory_order::acq_rel) &&
         Contains(scopes, sycl::memory_scope::system);
}

bool IsInteropPathQualified(const sycl::device& device, InteropUsmKind kind) {
  return device.has(sycl::aspect::atomic64) &&
         HasUsmAllocation(device, kind) &&
         HasAtomicUsmAllocation(device, kind) &&
         HasSystemAcqRelContract(device);
}

bool IsInterCoOpQualified(const sycl::device& device) {
  return IsInteropPathQualified(device, InteropUsmKind::Host) ||
         IsInteropPathQualified(device, InteropUsmKind::Shared);
}

template<class T>
bool HostWaitEqual(T& object, T expected,
                   std::chrono::milliseconds timeout = kHostPollTimeout) {
  std::atomic_ref<T> ref{object};
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while(std::chrono::steady_clock::now() < deadline) {
    if(ref.load(std::memory_order_acquire) == expected)
      return true;
    std::this_thread::yield();
  }
  return ref.load(std::memory_order_acquire) == expected;
}

void RethrowThreadException(const std::exception_ptr& error) {
  if(error)
    std::rethrow_exception(error);
}

// -----------------------------------------------------------------------------
// Patch 1: public metadata/API tests.
// -----------------------------------------------------------------------------
void TestPatch1DeviceInfo(const sycl::device& device, TestState& state) {
  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();

  static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(orders)>,
      std::vector<sycl::memory_order>>);
  static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(scopes)>,
      std::vector<sycl::memory_scope>>);

  std::cout << "  atomic orders : ";
  PrintOrders(orders);
  std::cout << '\n';
  std::cout << "  atomic scopes : ";
  PrintScopes(scopes);
  std::cout << '\n';

  state.Check(!orders.empty(), "device order capability vector is non-empty");
  state.Check(!scopes.empty(), "device scope capability vector is non-empty");
  state.Check(!HasDuplicates(orders), "device order capability vector has no duplicates");
  state.Check(!HasDuplicates(scopes), "device scope capability vector has no duplicates");
  state.Check(Contains(orders, sycl::memory_order::relaxed),
              "mandatory relaxed order is reported");
  state.Check(Contains(scopes, sycl::memory_scope::sub_group),
              "mandatory sub_group scope is reported");
  state.Check(Contains(scopes, sycl::memory_scope::work_group),
              "mandatory work_group scope is reported");

  const bool acquire = Contains(orders, sycl::memory_order::acquire);
  const bool release = Contains(orders, sycl::memory_order::release);
  const bool acq_rel = Contains(orders, sycl::memory_order::acq_rel);
  const bool seq_cst = Contains(orders, sycl::memory_order::seq_cst);
  const bool device_scope = Contains(scopes, sycl::memory_scope::device);
  const bool system_scope = Contains(scopes, sycl::memory_scope::system);

  if(acq_rel)
    state.Check(acquire && release,
                "acq_rel report is accompanied by acquire and release");
  if(seq_cst)
    state.Check(acquire && release && acq_rel,
                "seq_cst report is accompanied by acquire/release/acq_rel");
  if(system_scope)
    state.Check(device_scope,
                "system scope report is accompanied by device scope");

  const bool usm_device = device.has(sycl::aspect::usm_device_allocations);
  const bool usm_host = device.has(sycl::aspect::usm_host_allocations);
  const bool usm_atomic_host =
      device.has(sycl::aspect::usm_atomic_host_allocations);
  const bool usm_shared = device.has(sycl::aspect::usm_shared_allocations);
  const bool usm_atomic_shared =
      device.has(sycl::aspect::usm_atomic_shared_allocations);
  const bool atomic64 = device.has(sycl::aspect::atomic64);

  const auto aspects = device.get_info<sycl::info::device::aspects>();
  const auto aspect_consistent = [&](sycl::aspect aspect) {
    return device.has(aspect) == Contains(aspects, aspect);
  };

  state.Check(aspect_consistent(sycl::aspect::usm_device_allocations),
              "usm_device_allocations has() agrees with aspects list");
  state.Check(aspect_consistent(sycl::aspect::usm_host_allocations),
              "usm_host_allocations has() agrees with aspects list");
  state.Check(aspect_consistent(sycl::aspect::usm_atomic_host_allocations),
              "usm_atomic_host_allocations has() agrees with aspects list");
  state.Check(aspect_consistent(sycl::aspect::usm_shared_allocations),
              "usm_shared_allocations has() agrees with aspects list");
  state.Check(aspect_consistent(sycl::aspect::usm_atomic_shared_allocations),
              "usm_atomic_shared_allocations has() agrees with aspects list");
  state.Check(aspect_consistent(sycl::aspect::atomic64),
              "atomic64 has() agrees with aspects list");

  if(usm_atomic_host)
    state.Check(usm_host, "atomic host USM implies host USM");
  if(usm_atomic_shared)
    state.Check(usm_shared, "atomic shared USM implies shared USM");

  std::cout << "  atomic64                    : " << atomic64 << '\n';
  std::cout << "  usm_device_allocations      : " << usm_device << '\n';
  std::cout << "  usm_host_allocations        : " << usm_host << '\n';
  std::cout << "  usm_atomic_host_allocs      : " << usm_atomic_host << '\n';
  std::cout << "  usm_shared_allocations      : " << usm_shared << '\n';
  std::cout << "  usm_atomic_shared_allocs    : " << usm_atomic_shared << '\n';
  std::cout << "  acquire/release/acq_rel     : "
            << (acquire && release && acq_rel) << '\n';
  std::cout << "  system scope                : " << system_scope << '\n';
  std::cout << "  host-USM InterCoOp path     : "
            << IsInteropPathQualified(device, InteropUsmKind::Host) << '\n';
  std::cout << "  shared-USM InterCoOp path   : "
            << IsInteropPathQualified(device, InteropUsmKind::Shared) << '\n';
}

// -----------------------------------------------------------------------------
// Patch 2: backend-specific reporting invariants for the exact patched backends.
// -----------------------------------------------------------------------------
void TestPatch2BackendReporting(const sycl::device& device, TestState& state) {
  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();

  const auto has_order = [&](sycl::memory_order o) { return Contains(orders, o); };
  const auto has_scope = [&](sycl::memory_scope s) { return Contains(scopes, s); };

  const bool atomic_host =
      device.has(sycl::aspect::usm_atomic_host_allocations);
  const bool atomic_shared =
      device.has(sycl::aspect::usm_atomic_shared_allocations);

  switch(device.get_backend()) {
  case sycl::backend::cuda:
  case sycl::backend::hip:
  case sycl::backend::level_zero:
  case sycl::backend::ocl:
    state.Check(has_order(sycl::memory_order::relaxed) &&
                    has_order(sycl::memory_order::acquire) &&
                    has_order(sycl::memory_order::release) &&
                    has_order(sycl::memory_order::acq_rel),
                "PATCH 2 GPU backend reports relaxed/acquire/release/acq_rel");
    state.Check(has_scope(sycl::memory_scope::sub_group) &&
                    has_scope(sycl::memory_scope::work_group) &&
                    has_scope(sycl::memory_scope::device),
                "PATCH 2 GPU backend reports sub_group/work_group/device scopes");
    // Deliberately do NOT equate system scope with either atomic-USM aspect.
    // Scope and allocation capability are orthogonal dimensions.
    break;

  case sycl::backend::omp:
    state.Check(has_order(sycl::memory_order::relaxed) &&
                    has_order(sycl::memory_order::acquire) &&
                    has_order(sycl::memory_order::release) &&
                    has_order(sycl::memory_order::acq_rel) &&
                    has_order(sycl::memory_order::seq_cst),
                "PATCH 2 OpenMP reports full order set including seq_cst");
    state.Check(has_scope(sycl::memory_scope::sub_group) &&
                    has_scope(sycl::memory_scope::work_group) &&
                    has_scope(sycl::memory_scope::device) &&
                    has_scope(sycl::memory_scope::system),
                "PATCH 2 OpenMP reports sub_group/work_group/device/system scopes");
    state.Check(atomic_host,
                "PATCH 2 OpenMP reports atomic host USM");
    state.Check(atomic_shared,
                "PATCH 2 OpenMP reports atomic shared USM");
    break;

  case sycl::backend::metal:
  case sycl::backend::vk:
    state.Check(has_order(sycl::memory_order::relaxed),
                "unpatched backend retains relaxed baseline order");
    state.Check(has_scope(sycl::memory_scope::sub_group) &&
                    has_scope(sycl::memory_scope::work_group),
                "unpatched backend retains sub_group/work_group baseline scopes");
    break;
  }
}

void TestContext(const sycl::context& context, TestState& state,
                 std::string_view label) {
  const auto devices = context.get_devices();
  const auto orders = context.get_info<
      sycl::info::context::atomic_memory_order_capabilities>();
  const auto scopes = context.get_info<
      sycl::info::context::atomic_memory_scope_capabilities>();

  static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(orders)>,
      std::vector<sycl::memory_order>>);
  static_assert(std::is_same_v<
      std::remove_cvref_t<decltype(scopes)>,
      std::vector<sycl::memory_scope>>);

  const auto expected_orders = ManualOrderIntersection(devices);
  const auto expected_scopes = ManualScopeIntersection(devices);

  std::cout << "\n[Context] " << label << "\n";
  std::cout << "  devices       : " << devices.size() << '\n';
  std::cout << "  orders        : ";
  PrintOrders(orders);
  std::cout << '\n';
  std::cout << "  scopes        : ";
  PrintScopes(scopes);
  std::cout << '\n';

  state.Check(orders == expected_orders,
              "PATCH 1 context order query equals independent intersection");
  state.Check(scopes == expected_scopes,
              "PATCH 1 context scope query equals independent intersection");
}

void TestAllContexts(TestState& state) {
  const auto platforms = sycl::platform::get_platforms();
  for(std::size_t i = 0; i < platforms.size(); ++i) {
    const auto devices = platforms[i].get_devices();
    if(devices.empty())
      continue;

    try {
      const sycl::context context{devices};
      std::ostringstream label;
      label << "platform " << i << " / "
            << platforms[i].get_info<sycl::info::platform::name>();
      TestContext(context, state, label.str());
    } catch(const sycl::exception& e) {
      state.Fail(std::string{"context construction/query threw: "} + e.what());
    }
  }
}

// -----------------------------------------------------------------------------
// Integral RMW sequence. Executes every integral operation touched by Patch 2.
// -----------------------------------------------------------------------------
template<class T, sycl::memory_scope Scope, sycl::memory_order RmwOrder>
void IntegralRmwKernel(T* m) {
  using A = GlobalAtomic<T, Scope>;
  A a{m[0]};

  constexpr auto store_order =
      RmwOrder == sycl::memory_order::seq_cst
          ? sycl::memory_order::seq_cst
          : (RmwOrder == sycl::memory_order::acq_rel
                 ? sycl::memory_order::release
                 : sycl::memory_order::relaxed);
  constexpr auto load_order =
      RmwOrder == sycl::memory_order::seq_cst
          ? sycl::memory_order::seq_cst
          : (RmwOrder == sycl::memory_order::acq_rel
                 ? sycl::memory_order::acquire
                 : sycl::memory_order::relaxed);
  constexpr auto failure_order =
      RmwOrder == sycl::memory_order::seq_cst
          ? sycl::memory_order::seq_cst
          : (RmwOrder == sycl::memory_order::acq_rel
                 ? sycl::memory_order::acquire
                 : sycl::memory_order::relaxed);

  a.store(static_cast<T>(0xF0), store_order, Scope);
  m[1] = a.load(load_order, Scope);
  m[2] = a.exchange(static_cast<T>(0x0F), RmwOrder, Scope);
  m[3] = a.load(load_order, Scope);

  T expected = static_cast<T>(0x0F);
  const bool cas_success = a.compare_exchange_strong(
      expected, static_cast<T>(0x33), RmwOrder, failure_order, Scope);
  m[4] = static_cast<T>(cas_success ? 1 : 0);
  m[5] = expected;
  m[6] = a.load(load_order, Scope);

  expected = static_cast<T>(0x44);
  const bool cas_failure = a.compare_exchange_strong(
      expected, static_cast<T>(0x55), RmwOrder, failure_order, Scope);
  m[7] = static_cast<T>(cas_failure ? 1 : 0);
  m[8] = expected;

  m[9]  = a.fetch_add(static_cast<T>(5), RmwOrder, Scope);     // 0x33 -> 0x38
  m[10] = a.fetch_sub(static_cast<T>(6), RmwOrder, Scope);     // 0x38 -> 0x32
  m[11] = a.fetch_or(static_cast<T>(0x0F), RmwOrder, Scope);   // 0x32 -> 0x3F
  m[12] = a.fetch_and(static_cast<T>(0x3A), RmwOrder, Scope);  // 0x3F -> 0x3A
  m[13] = a.fetch_xor(static_cast<T>(0x0F), RmwOrder, Scope);  // 0x3A -> 0x35
  m[14] = a.fetch_min(static_cast<T>(0x20), RmwOrder, Scope);  // 0x35 -> 0x20
  m[15] = a.fetch_max(static_cast<T>(0x80), RmwOrder, Scope);  // 0x20 -> 0x80
  m[16] = a.load(load_order, Scope);
}

template<class T>
bool ValidateIntegralSequence(const std::array<T, 17>& m,
                              std::string& reason) {
  const std::array<T, 17> expected{
      static_cast<T>(0x80), // final atomic word
      static_cast<T>(0xF0),
      static_cast<T>(0xF0),
      static_cast<T>(0x0F),
      static_cast<T>(1),
      static_cast<T>(0x0F),
      static_cast<T>(0x33),
      static_cast<T>(0),
      static_cast<T>(0x33),
      static_cast<T>(0x33),
      static_cast<T>(0x38),
      static_cast<T>(0x32),
      static_cast<T>(0x3F),
      static_cast<T>(0x3A),
      static_cast<T>(0x35),
      static_cast<T>(0x20),
      static_cast<T>(0x80)};

  for(std::size_t i = 0; i < expected.size(); ++i) {
    if(m[i] != expected[i]) {
      std::ostringstream out;
      out << "integral RMW result[" << i << "] expected 0x" << std::hex
          << static_cast<unsigned long long>(expected[i]) << " got 0x"
          << static_cast<unsigned long long>(m[i]);
      reason = out.str();
      return false;
    }
  }
  return true;
}

template<class T, sycl::memory_scope Scope, sycl::memory_order Order>
void RunIntegralDeviceUsmSequence(sycl::queue& q, TestState& state,
                                  std::string_view label) {
  DeviceUsmArray<T> memory{q, 17};
  std::array<T, 17> zero{};
  q.memcpy(memory.get(), zero.data(), sizeof(zero)).wait_and_throw();

  T* ptr = memory.get();
  q.single_task([=]() { IntegralRmwKernel<T, Scope, Order>(ptr); })
      .wait_and_throw();

  std::array<T, 17> result{};
  q.memcpy(result.data(), memory.get(), sizeof(result)).wait_and_throw();

  std::string reason;
  if(ValidateIntegralSequence(result, reason))
    state.Pass(label);
  else
    state.Fail(std::string{label} + ": " + reason);
}

template<class T, sycl::memory_scope Scope, sycl::memory_order Order>
void RunIntegralInteropSequence(sycl::queue& q, TestState& state,
                                InteropUsmKind kind, std::string_view label) {
  InteropUsmArray<T> memory{q, kind, 17, SharedAtomicAlignment<T>()};
  for(std::size_t i = 0; i < memory.size(); ++i)
    memory[i] = T{};

  const auto addr = reinterpret_cast<std::uintptr_t>(memory.get());
  state.Check(addr % GlobalAtomic<T, Scope>::required_alignment == 0,
              std::string{label} + " SYCL atomic_ref alignment satisfied");
  if constexpr(Scope == sycl::memory_scope::system) {
    state.Check(addr % std::atomic_ref<T>::required_alignment == 0,
                std::string{label} + " std::atomic_ref alignment satisfied");
  }

  T* ptr = memory.get();
  q.single_task([=]() { IntegralRmwKernel<T, Scope, Order>(ptr); })
      .wait_and_throw();

  std::array<T, 17> result{};
  std::copy(memory.get(), memory.get() + result.size(), result.begin());

  std::string reason;
  if(ValidateIntegralSequence(result, reason))
    state.Pass(label);
  else
    state.Fail(std::string{label} + ": " + reason);
}

// -----------------------------------------------------------------------------
// Floating-point atomic sequence: exact powers/small integers avoid tolerance
// issues and target add/sub/min/max + CAS paths modified by Patch 2.
// -----------------------------------------------------------------------------
template<class T, sycl::memory_scope Scope, sycl::memory_order RmwOrder>
void FloatingRmwKernel(T* m) {
  using A = GlobalAtomic<T, Scope>;
  A a{m[0]};

  constexpr auto store_order =
      RmwOrder == sycl::memory_order::seq_cst
          ? sycl::memory_order::seq_cst
          : (RmwOrder == sycl::memory_order::acq_rel
                 ? sycl::memory_order::release
                 : sycl::memory_order::relaxed);
  constexpr auto load_order =
      RmwOrder == sycl::memory_order::seq_cst
          ? sycl::memory_order::seq_cst
          : (RmwOrder == sycl::memory_order::acq_rel
                 ? sycl::memory_order::acquire
                 : sycl::memory_order::relaxed);
  constexpr auto failure_order =
      RmwOrder == sycl::memory_order::seq_cst
          ? sycl::memory_order::seq_cst
          : (RmwOrder == sycl::memory_order::acq_rel
                 ? sycl::memory_order::acquire
                 : sycl::memory_order::relaxed);

  a.store(T{4}, store_order, Scope);
  m[1] = a.load(load_order, Scope);
  m[2] = a.exchange(T{8}, RmwOrder, Scope);

  T expected = T{8};
  const bool cas_success = a.compare_exchange_strong(
      expected, T{2}, RmwOrder, failure_order, Scope);
  m[3] = cas_success ? T{1} : T{0};
  m[4] = expected;
  m[5] = a.fetch_add(T{3}, RmwOrder, Scope); // 2 -> 5
  m[6] = a.fetch_sub(T{1}, RmwOrder, Scope); // 5 -> 4
  m[7] = a.fetch_min(T{3}, RmwOrder, Scope); // 4 -> 3
  m[8] = a.fetch_max(T{7}, RmwOrder, Scope); // 3 -> 7
  m[9] = a.load(load_order, Scope);
}

template<class T>
bool ValidateFloatingSequence(const std::array<T, 10>& m, std::string& reason) {
  const std::array<T, 10> expected{
      T{7}, T{4}, T{4}, T{1}, T{8}, T{2}, T{5}, T{4}, T{3}, T{7}};
  for(std::size_t i = 0; i < expected.size(); ++i) {
    if(m[i] != expected[i]) {
      std::ostringstream out;
      out << "floating RMW result[" << i << "] expected " << expected[i]
          << " got " << m[i];
      reason = out.str();
      return false;
    }
  }
  return true;
}

template<class T, sycl::memory_scope Scope, sycl::memory_order Order>
void RunFloatingDeviceUsmSequence(sycl::queue& q, TestState& state,
                                  std::string_view label) {
  DeviceUsmArray<T> memory{q, 10};
  std::array<T, 10> zero{};
  q.memcpy(memory.get(), zero.data(), sizeof(zero)).wait_and_throw();
  T* ptr = memory.get();
  q.single_task([=]() { FloatingRmwKernel<T, Scope, Order>(ptr); })
      .wait_and_throw();
  std::array<T, 10> result{};
  q.memcpy(result.data(), memory.get(), sizeof(result)).wait_and_throw();
  std::string reason;
  if(ValidateFloatingSequence(result, reason))
    state.Pass(label);
  else
    state.Fail(std::string{label} + ": " + reason);
}

template<class T, sycl::memory_scope Scope, sycl::memory_order Order>
void RunFloatingInteropSequence(sycl::queue& q, TestState& state,
                                InteropUsmKind kind, std::string_view label) {
  InteropUsmArray<T> memory{q, kind, 10, SharedAtomicAlignment<T>()};
  for(std::size_t i = 0; i < memory.size(); ++i)
    memory[i] = T{};
  T* ptr = memory.get();
  q.single_task([=]() { FloatingRmwKernel<T, Scope, Order>(ptr); })
      .wait_and_throw();
  std::array<T, 10> result{};
  std::copy(memory.get(), memory.get() + result.size(), result.begin());
  std::string reason;
  if(ValidateFloatingSequence(result, reason))
    state.Pass(label);
  else
    state.Fail(std::string{label} + ": " + reason);
}

// -----------------------------------------------------------------------------
// Work-group local-memory atomic test.
// -----------------------------------------------------------------------------
void TestWorkGroupAtomic(sycl::queue& q, const sycl::device& device,
                         TestState& state) {
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();
  if(!Contains(scopes, sycl::memory_scope::work_group) ||
     !device.has(sycl::aspect::usm_device_allocations)) {
    state.Skip("work_group local atomic execution prerequisites unavailable");
    return;
  }

  const std::size_t max_wg =
      device.get_info<sycl::info::device::max_work_group_size>();
  const std::size_t wg = std::max<std::size_t>(1, std::min<std::size_t>(64, max_wg));

  DeviceUsmArray<U32> output{q, 1};
  U32 zero = 0;
  q.memcpy(output.get(), &zero, sizeof(zero)).wait_and_throw();
  U32* out = output.get();

  q.submit([&](sycl::handler& cgh) {
     sycl::local_accessor<U32, 1> local{sycl::range<1>{1}, cgh};
     cgh.parallel_for(
         sycl::nd_range<1>{sycl::range<1>{wg}, sycl::range<1>{wg}},
         [=](sycl::nd_item<1> item) {
           if(item.get_local_linear_id() == 0)
             local[0] = 0;
           item.barrier(sycl::access::fence_space::local_space);

           using A = sycl::atomic_ref<
               U32,
               sycl::memory_order::relaxed,
               sycl::memory_scope::work_group,
               sycl::access::address_space::local_space>;
           A a{local[0]};
           a.fetch_add(1, sycl::memory_order::relaxed,
                       sycl::memory_scope::work_group);

           item.barrier(sycl::access::fence_space::local_space);
           if(item.get_local_linear_id() == 0)
             out[0] = local[0];
         });
   }).wait_and_throw();

  U32 result = 0;
  q.memcpy(&result, output.get(), sizeof(result)).wait_and_throw();
  state.Check(result == static_cast<U32>(wg),
              "work_group-scope local atomic fetch_add reaches exact work-group count");
}

// -----------------------------------------------------------------------------
// Device-scope contention: many device work-items race on one uint64 counter.
// -----------------------------------------------------------------------------
void TestDeviceContention64(sycl::queue& q, const sycl::device& device,
                            TestState& state, std::size_t stress) {
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();
  if(!device.has(sycl::aspect::atomic64) ||
     !device.has(sycl::aspect::usm_device_allocations) ||
     !Contains(scopes, sycl::memory_scope::device)) {
    state.Skip("uint64 device-scope contention prerequisites unavailable");
    return;
  }

  DeviceUsmArray<U64> counter{q, 1};
  U64 zero = 0;
  q.memcpy(counter.get(), &zero, sizeof(zero)).wait_and_throw();

  constexpr std::size_t work_items = 256;
  const std::size_t loops = std::max<std::size_t>(1, stress / work_items);
  U64* ptr = counter.get();

  q.parallel_for(sycl::range<1>{work_items}, [=](sycl::id<1>) {
     GlobalAtomic<U64, sycl::memory_scope::device> a{ptr[0]};
     for(std::size_t i = 0; i < loops; ++i)
       a.fetch_add(1, sycl::memory_order::relaxed,
                   sycl::memory_scope::device);
   }).wait_and_throw();

  U64 result = 0;
  q.memcpy(&result, counter.get(), sizeof(result)).wait_and_throw();
  const U64 expected = static_cast<U64>(work_items * loops);
  state.Check(result == expected,
              "uint64 device-scope fetch_add contention has no lost updates");
}

// -----------------------------------------------------------------------------
// Explicit atomic_fence smoke. This primarily detects compilation/lowering or
// runtime failures in the Patch-2 fence path; publication semantics are tested
// separately below using real host/device synchronization.
// -----------------------------------------------------------------------------
void TestUsmAllocationSmoke(sycl::queue& q, const sycl::device& device,
                            TestState& state, InteropUsmKind kind) {
  const std::string prefix = std::string{InteropUsmName(kind)} + " allocation";
  if(!HasUsmAllocation(device, kind)) {
    state.Skip(prefix + ": allocation aspect unavailable");
    return;
  }

  InteropUsmArray<U64> memory{q, kind, 2, alignof(U64)};
  constexpr U64 magic = 0x1234'5678'9ABC'DEF0ULL;
  memory[0] = magic;
  memory[1] = 0;
  U64* p = memory.get();

  q.single_task([=]() { p[1] = p[0] ^ U64{0x55}; }).wait_and_throw();
  state.Check(memory[1] == (magic ^ U64{0x55}),
              prefix + ": host write -> device read/write -> host read works");
}

void TestAtomicFenceSmoke(sycl::queue& q, const sycl::device& device,
                          TestState& state, InteropUsmKind kind) {
  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();
  const std::string prefix = std::string{InteropUsmName(kind)} + " system fence";

  if(!Contains(orders, sycl::memory_order::acquire) ||
     !Contains(orders, sycl::memory_order::release) ||
     !Contains(scopes, sycl::memory_scope::system) ||
     !HasAtomicUsmAllocation(device, kind)) {
    state.Skip(prefix + ": prerequisites unavailable");
    return;
  }

  InteropUsmArray<U32> marker{q, kind, 1, SharedAtomicAlignment<U32>()};
  marker[0] = 0;
  U32* ptr = marker.get();
  q.single_task([=]() {
     sycl::atomic_fence(sycl::memory_order::release,
                        sycl::memory_scope::system);
     ptr[0] = 1;
     sycl::atomic_fence(sycl::memory_order::acquire,
                        sycl::memory_scope::system);
   }).wait_and_throw();

  state.Check(marker[0] == 1, prefix + ": acquire/release path executes");
}

// Host release -> device system-scope acquire. The host publishes before the
// kernel is submitted, avoiding a scheduler-dependent device spin loop while
// still validating cross-agent release/acquire visibility on the allocation.
void TestCpuToDevicePublication64(sycl::queue& q, TestState& state,
                                  InteropUsmKind kind) {
  InteropUsmArray<U64> m{q, kind, 4, SharedAtomicAlignment<U64>()};
  for(std::size_t i = 0; i < m.size(); ++i)
    m[i] = 0;

  constexpr U64 magic = 0xD1CE'B00C'CAFEBABEULL;
  m[1] = magic;
  std::atomic_ref<U64>{m[0]}.store(1, std::memory_order_release);

  U64* p = m.get();
  q.single_task([=]() {
     GlobalAtomic<U64, sycl::memory_scope::system> flag{p[0]};
     const U64 seen = flag.load(sycl::memory_order::acquire,
                                sycl::memory_scope::system);
     p[2] = seen;
     if(seen == 1)
       p[3] = p[1];
   }).wait_and_throw();

  const std::string prefix = std::string{InteropUsmName(kind)} + " CPU->device";
  state.Check(m[2] == 1, prefix + ": device acquire observes CPU release store");
  state.Check(m[3] == magic,
              prefix + ": CPU release/device acquire publishes plain payload");
}

// Device release -> host acquire. The device never waits for the host; the host
// polls with a real wall-clock timeout, so this cannot fail merely because a CPU
// backend consumed an arbitrary fixed spin count too quickly.
void TestDeviceToCpuPublication64(sycl::queue& q, TestState& state,
                                  InteropUsmKind kind) {
  InteropUsmArray<U64> m{q, kind, 2, SharedAtomicAlignment<U64>()};
  m[0] = 0;
  m[1] = 0;
  U64* p = m.get();
  constexpr U64 magic = 0xA70B'1C5E'DEADBEEFULL;

  std::exception_ptr worker_error;
  std::thread worker([&] {
    try {
      q.single_task([=]() {
         p[1] = magic;
         GlobalAtomic<U64, sycl::memory_scope::system> flag{p[0]};
         flag.store(1, sycl::memory_order::release,
                    sycl::memory_scope::system);
       }).wait_and_throw();
    } catch(...) {
      worker_error = std::current_exception();
    }
  });

  const bool observed = HostWaitEqual(m[0], U64{1});
  worker.join();
  RethrowThreadException(worker_error);

  const std::string prefix = std::string{InteropUsmName(kind)} + " device->CPU";
  state.Check(observed, prefix + ": CPU acquire observes device release store");
  if(observed)
    state.Check(m[1] == magic,
                prefix + ": device release/CPU acquire publishes plain payload");
}

// CPU and device both fetch_add the same word. A host-side start gate aligns the
// CPU loop with kernel submission without making the device spin waiting for a
// host flag. The exact 2*N result verifies one coherent atomic modification order.
void TestConcurrentHostDeviceAdd64(sycl::queue& q, TestState& state,
                                   InteropUsmKind kind, std::size_t stress) {
  InteropUsmArray<U64> m{q, kind, 1, SharedAtomicAlignment<U64>()};
  m[0] = 0;
  U64* p = m.get();
  const U64 iterations = static_cast<U64>(stress);

  std::atomic<bool> worker_ready{false};
  std::atomic<bool> start{false};
  std::exception_ptr worker_error;

  std::thread worker([&] {
    try {
      worker_ready.store(true, std::memory_order_release);
      while(!start.load(std::memory_order_acquire))
        std::this_thread::yield();

      q.single_task([=]() {
         GlobalAtomic<U64, sycl::memory_scope::system> counter{p[0]};
         for(U64 i = 0; i < iterations; ++i)
           counter.fetch_add(1, sycl::memory_order::acq_rel,
                             sycl::memory_scope::system);
       }).wait_and_throw();
    } catch(...) {
      worker_error = std::current_exception();
    }
  });

  while(!worker_ready.load(std::memory_order_acquire))
    std::this_thread::yield();

  std::atomic_ref<U64> counter{m[0]};
  start.store(true, std::memory_order_release);
  for(U64 i = 0; i < iterations; ++i)
    counter.fetch_add(1, std::memory_order_acq_rel);

  worker.join();
  RethrowThreadException(worker_error);

  const U64 expected = iterations * 2;
  const U64 actual = counter.load(std::memory_order_acquire);
  std::ostringstream label;
  label << InteropUsmName(kind)
        << " concurrent CPU/device uint64 fetch_add exact 2*N ("
        << expected << ')';
  state.Check(actual == expected, label.str());
}

// Mixed host/device bitwise RMW on the same word. OR and AND are commutative,
// so the final state is deterministic regardless of which participant wins the
// race first while still exercising the common atomic domain.
void TestHostDeviceBitwise64(sycl::queue& q, TestState& state,
                             InteropUsmKind kind) {
  InteropUsmArray<U64> m{q, kind, 1, SharedAtomicAlignment<U64>()};
  U64* p = m.get();
  const std::string prefix = std::string{InteropUsmName(kind)} + " CPU/device";

  auto run = [&](bool do_or) {
    m[0] = do_or ? U64{0} : U64{0xF};
    std::atomic<bool> worker_ready{false};
    std::atomic<bool> start{false};
    std::exception_ptr worker_error;

    std::thread worker([&] {
      try {
        worker_ready.store(true, std::memory_order_release);
        while(!start.load(std::memory_order_acquire))
          std::this_thread::yield();
        q.single_task([=]() {
           GlobalAtomic<U64, sycl::memory_scope::system> a{p[0]};
           if(do_or)
             a.fetch_or(U64{0x2}, sycl::memory_order::acq_rel,
                        sycl::memory_scope::system);
           else
             a.fetch_and(~U64{0x2}, sycl::memory_order::acq_rel,
                         sycl::memory_scope::system);
         }).wait_and_throw();
      } catch(...) {
        worker_error = std::current_exception();
      }
    });

    while(!worker_ready.load(std::memory_order_acquire))
      std::this_thread::yield();
    std::atomic_ref<U64> host{m[0]};
    start.store(true, std::memory_order_release);
    if(do_or)
      host.fetch_or(U64{0x1}, std::memory_order_acq_rel);
    else
      host.fetch_and(~U64{0x1}, std::memory_order_acq_rel);

    worker.join();
    RethrowThreadException(worker_error);
    return host.load(std::memory_order_acquire);
  };

  state.Check(run(true) == U64{0x3},
              prefix + " fetch_or merges both participants exactly");
  state.Check(run(false) == U64{0xC},
              prefix + " fetch_and merges both participants exactly");
}

void TestHostDeviceCasRace64(sycl::queue& q, TestState& state,
                             InteropUsmKind kind) {
  constexpr std::size_t N = 1024;
  InteropUsmArray<U64> cells{q, kind, N, SharedAtomicAlignment<U64>()};
  InteropUsmArray<U64> device_success{q, kind, N, SharedAtomicAlignment<U64>()};
  for(std::size_t i = 0; i < N; ++i) {
    cells[i] = 0;
    device_success[i] = 0;
  }

  U64* c = cells.get();
  U64* d = device_success.get();
  std::atomic<bool> worker_ready{false};
  std::atomic<bool> start{false};
  std::exception_ptr worker_error;

  std::thread worker([&] {
    try {
      worker_ready.store(true, std::memory_order_release);
      while(!start.load(std::memory_order_acquire))
        std::this_thread::yield();

      q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> id) {
        const std::size_t i = id[0];
        GlobalAtomic<U64, sycl::memory_scope::system> a{c[i]};
        U64 expected = 0;
        const bool won = a.compare_exchange_strong(
            expected, U64{2},
            sycl::memory_order::acq_rel,
            sycl::memory_order::acquire,
            sycl::memory_scope::system);
        d[i] = won ? 1 : 0;
      }).wait_and_throw();
    } catch(...) {
      worker_error = std::current_exception();
    }
  });

  while(!worker_ready.load(std::memory_order_acquire))
    std::this_thread::yield();
  start.store(true, std::memory_order_release);

  std::vector<bool> host_success(N, false);
  for(std::size_t i = 0; i < N; ++i) {
    std::atomic_ref<U64> a{cells[i]};
    U64 expected = 0;
    host_success[i] = a.compare_exchange_strong(
        expected, U64{1},
        std::memory_order_acq_rel,
        std::memory_order_acquire);
  }

  worker.join();
  RethrowThreadException(worker_error);

  std::size_t bad = 0;
  for(std::size_t i = 0; i < N; ++i) {
    const U64 value = std::atomic_ref<U64>{cells[i]}.load(std::memory_order_acquire);
    const bool device_won = device_success[i] != 0;
    const bool host_won = host_success[i];
    const bool exactly_one = device_won != host_won;
    const bool value_matches =
        (device_won && value == 2) || (host_won && value == 1);
    if(!exactly_one || !value_matches)
      ++bad;
  }

  std::ostringstream label;
  label << InteropUsmName(kind)
        << " 1024 CPU/device CAS races each have exactly one coherent winner";
  state.Check(bad == 0, label.str());
}

void TestInteropSemanticSuite(sycl::queue& q, const sycl::device& device,
                              TestState& state, InteropUsmKind kind,
                              std::size_t stress) {
  const std::string path = InteropUsmName(kind);
  if(!IsInteropPathQualified(device, kind)) {
    state.Skip(path + ": uint64 host/device InterCoOp contract not advertised");
    return;
  }

  std::cout << "\n  -- " << path << " CPU/device InterCoOp semantics --\n";
  TestCpuToDevicePublication64(q, state, kind);
  TestDeviceToCpuPublication64(q, state, kind);
  TestConcurrentHostDeviceAdd64(q, state, kind, stress);
  TestHostDeviceBitwise64(q, state, kind);
  TestHostDeviceCasRace64(q, state, kind);
}

void RunSystemScopePathTests(sycl::queue& q, const sycl::device& device,
                             TestState& state, const Options& options,
                             InteropUsmKind kind) {
  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();

  const bool acq_rel = Contains(orders, sycl::memory_order::acq_rel);
  const bool seq_cst = Contains(orders, sycl::memory_order::seq_cst);
  const bool system_scope = Contains(scopes, sycl::memory_scope::system);
  const bool atomic64 = device.has(sycl::aspect::atomic64);
  const bool fp64 = device.has(sycl::aspect::fp64);
  const bool allocation = HasUsmAllocation(device, kind);
  const bool atomic_allocation = HasAtomicUsmAllocation(device, kind);

  std::cout << "\n  -- " << InteropUsmName(kind) << " path --\n";
  std::cout << "  allocation aspect           : " << allocation << '\n';
  std::cout << "  atomic allocation aspect    : " << atomic_allocation << '\n';
  std::cout << "  system scope                : " << system_scope << '\n';
  std::cout << "  acq_rel                     : " << acq_rel << '\n';
  std::cout << "  uint64 InterCoOp qualified  : "
            << IsInteropPathQualified(device, kind) << '\n';

  TestUsmAllocationSmoke(q, device, state, kind);

  if(!(system_scope && allocation && atomic_allocation)) {
    state.Skip(std::string{InteropUsmName(kind)} +
               ": system-scope atomic RMW suite prerequisites unavailable");
    return;
  }

  const auto label = [&](std::string_view what) {
    return std::string{InteropUsmName(kind)} + " " + std::string{what};
  };

  RunIntegralInteropSequence<U32, sycl::memory_scope::system,
                             sycl::memory_order::relaxed>(
      q, state, kind, label("uint32 system-scope complete RMW / relaxed"));
  RunFloatingInteropSequence<float, sycl::memory_scope::system,
                             sycl::memory_order::relaxed>(
      q, state, kind, label("float system-scope RMW / relaxed"));

  if(acq_rel) {
    RunIntegralInteropSequence<U32, sycl::memory_scope::system,
                               sycl::memory_order::acq_rel>(
        q, state, kind, label("uint32 system-scope complete RMW / acq_rel"));
    RunFloatingInteropSequence<float, sycl::memory_scope::system,
                               sycl::memory_order::acq_rel>(
        q, state, kind, label("float system-scope RMW / acq_rel"));
  }

  if(seq_cst) {
    RunIntegralInteropSequence<U32, sycl::memory_scope::system,
                               sycl::memory_order::seq_cst>(
        q, state, kind, label("uint32 system-scope complete RMW / seq_cst"));
    RunFloatingInteropSequence<float, sycl::memory_scope::system,
                               sycl::memory_order::seq_cst>(
        q, state, kind, label("float system-scope RMW / seq_cst"));
  }

  if(atomic64) {
    RunIntegralInteropSequence<U64, sycl::memory_scope::system,
                               sycl::memory_order::relaxed>(
        q, state, kind, label("uint64 system-scope complete RMW / relaxed"));
    if(acq_rel)
      RunIntegralInteropSequence<U64, sycl::memory_scope::system,
                                 sycl::memory_order::acq_rel>(
          q, state, kind, label("uint64 system-scope complete RMW / acq_rel"));
    if(seq_cst)
      RunIntegralInteropSequence<U64, sycl::memory_scope::system,
                                 sycl::memory_order::seq_cst>(
          q, state, kind, label("uint64 system-scope complete RMW / seq_cst"));
  } else {
    state.Skip(label("uint64 system-scope sequence: atomic64 unavailable"));
  }

  if(fp64 && atomic64) {
    RunFloatingInteropSequence<double, sycl::memory_scope::system,
                               sycl::memory_order::relaxed>(
        q, state, kind, label("double system-scope RMW / relaxed"));
    if(acq_rel)
      RunFloatingInteropSequence<double, sycl::memory_scope::system,
                                 sycl::memory_order::acq_rel>(
          q, state, kind, label("double system-scope RMW / acq_rel"));
    if(seq_cst)
      RunFloatingInteropSequence<double, sycl::memory_scope::system,
                                 sycl::memory_order::seq_cst>(
          q, state, kind, label("double system-scope RMW / seq_cst"));
  } else {
    state.Skip(label("double system-scope sequence: fp64+atomic64 unavailable"));
  }

  TestAtomicFenceSmoke(q, device, state, kind);

  // The synchronization suite is valid for any device exposing this contract.
  // On a GPU it is literally CPU<->GPU; on the OpenMP device it validates the
  // same host/system atomic domain without claiming heterogeneous hardware.
  TestInteropSemanticSuite(q, device, state, kind, options.stress);
}

void TestPatch2Execution(const sycl::device& device, TestState& state,
                         const Options& options) {
  sycl::queue q{device};

  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();

  const bool acq_rel = Contains(orders, sycl::memory_order::acq_rel);
  const bool seq_cst = Contains(orders, sycl::memory_order::seq_cst);
  const bool device_scope = Contains(scopes, sycl::memory_scope::device);
  const bool usm_device = device.has(sycl::aspect::usm_device_allocations);
  const bool atomic64 = device.has(sycl::aspect::atomic64);
  const bool fp64 = device.has(sycl::aspect::fp64);

  std::cout << "\n  -- Device/work-group atomic execution --\n";
  TestWorkGroupAtomic(q, device, state);

  if(device_scope && usm_device) {
    RunIntegralDeviceUsmSequence<U32, sycl::memory_scope::device,
                                 sycl::memory_order::relaxed>(
        q, state, "uint32 device-scope complete RMW / relaxed");
    RunFloatingDeviceUsmSequence<float, sycl::memory_scope::device,
                                 sycl::memory_order::relaxed>(
        q, state, "float device-scope RMW / relaxed");

    if(acq_rel) {
      RunIntegralDeviceUsmSequence<U32, sycl::memory_scope::device,
                                   sycl::memory_order::acq_rel>(
          q, state, "uint32 device-scope complete RMW / acq_rel");
      RunFloatingDeviceUsmSequence<float, sycl::memory_scope::device,
                                   sycl::memory_order::acq_rel>(
          q, state, "float device-scope RMW / acq_rel");
    }

    if(seq_cst) {
      RunIntegralDeviceUsmSequence<U32, sycl::memory_scope::device,
                                   sycl::memory_order::seq_cst>(
          q, state, "uint32 device-scope complete RMW / seq_cst");
      RunFloatingDeviceUsmSequence<float, sycl::memory_scope::device,
                                   sycl::memory_order::seq_cst>(
          q, state, "float device-scope RMW / seq_cst");
    }

    if(atomic64) {
      RunIntegralDeviceUsmSequence<U64, sycl::memory_scope::device,
                                   sycl::memory_order::relaxed>(
          q, state, "uint64 device-scope complete RMW / relaxed");
      if(acq_rel)
        RunIntegralDeviceUsmSequence<U64, sycl::memory_scope::device,
                                     sycl::memory_order::acq_rel>(
            q, state, "uint64 device-scope complete RMW / acq_rel");
      if(seq_cst)
        RunIntegralDeviceUsmSequence<U64, sycl::memory_scope::device,
                                     sycl::memory_order::seq_cst>(
            q, state, "uint64 device-scope complete RMW / seq_cst");
      TestDeviceContention64(q, device, state, options.stress);
    } else {
      state.Skip("uint64 device-scope sequence: atomic64 not advertised");
      state.Skip("uint64 device contention: atomic64 not advertised");
    }

    if(fp64 && atomic64) {
      RunFloatingDeviceUsmSequence<double, sycl::memory_scope::device,
                                   sycl::memory_order::relaxed>(
          q, state, "double device-scope RMW / relaxed");
      if(acq_rel)
        RunFloatingDeviceUsmSequence<double, sycl::memory_scope::device,
                                     sycl::memory_order::acq_rel>(
            q, state, "double device-scope RMW / acq_rel");
      if(seq_cst)
        RunFloatingDeviceUsmSequence<double, sycl::memory_scope::device,
                                     sycl::memory_order::seq_cst>(
            q, state, "double device-scope RMW / seq_cst");
    } else {
      state.Skip("double device-scope atomic sequence: fp64+atomic64 unavailable");
    }
  } else {
    state.Skip("device-scope RMW suite: device scope or device USM unavailable");
  }

  // Crucially, host and shared USM are evaluated independently. A backend can
  // expose system scope while only one (or neither) allocation kind is atomic.
  const auto run_path = [&](InteropUsmKind kind) {
    try {
      RunSystemScopePathTests(q, device, state, options, kind);
    } catch(const sycl::exception& e) {
      state.Fail(std::string{InteropUsmName(kind)} +
                 ": SYCL exception while testing path: " + e.what());
    } catch(const std::exception& e) {
      state.Fail(std::string{InteropUsmName(kind)} +
                 ": std::exception while testing path: " + e.what());
    }
  };

  run_path(InteropUsmKind::Host);
  run_path(InteropUsmKind::Shared);

  q.wait_and_throw();
}

void PrintDeviceHeader(const sycl::device& device, std::size_t index) {
  std::cout << "\n============================================================\n";
  std::cout << "Device " << index << '\n';
  std::cout << "  name          : "
            << device.get_info<sycl::info::device::name>() << '\n';
  std::cout << "  vendor        : "
            << device.get_info<sycl::info::device::vendor>() << '\n';
  std::cout << "  driver        : "
            << device.get_info<sycl::info::device::driver_version>() << '\n';
  std::cout << "  backend       : " << BackendName(device.get_backend()) << '\n';
  std::cout << "  is_cpu/is_gpu : " << device.is_cpu() << '/' << device.is_gpu() << '\n';
}

} // namespace

int main(int argc, char** argv) {
  TestState state;

  try {
    const Options options = ParseOptions(argc, argv);

    std::cout
        << "AdaptiveCpp host/shared USM + atomic semantic validation\n"
        << "C++ standard expectation: C++20 (std::atomic_ref)\n"
        << "Stress iterations per CPU/device side: " << options.stress << "\n";

    const auto devices = sycl::device::get_devices();
    if(devices.empty()) {
      std::cerr << "FATAL: AdaptiveCpp exposed no SYCL devices.\n";
      return 100;
    }

    std::cout << "Discovered devices: " << devices.size() << '\n';

    if(options.device_index && *options.device_index >= devices.size()) {
      std::cerr << "FATAL: --device index " << *options.device_index
                << " is outside [0," << (devices.size() - 1) << "]\n";
      return 101;
    }

    std::size_t intercoop_devices = 0;
    std::size_t gpu_intercoop_devices = 0;
    std::size_t host_intercoop_devices = 0;
    std::size_t shared_intercoop_devices = 0;
    std::size_t gpu_host_intercoop_devices = 0;
    std::size_t gpu_shared_intercoop_devices = 0;

    for(std::size_t i = 0; i < devices.size(); ++i) {
      if(options.device_index && i != *options.device_index)
        continue;

      const auto& device = devices[i];
      PrintDeviceHeader(device, i);

      try {
        std::cout << "\n  -- Patch 1 metadata/API tests --\n";
        TestPatch1DeviceInfo(device, state);

        std::cout << "\n  -- Patch 2 backend reporting tests --\n";
        TestPatch2BackendReporting(device, state);

        const bool host_intercoop =
            IsInteropPathQualified(device, InteropUsmKind::Host);
        const bool shared_intercoop =
            IsInteropPathQualified(device, InteropUsmKind::Shared);
        const bool intercoop = host_intercoop || shared_intercoop;

        std::cout << "\n  SuperNova uint64 InterCoOp qualification\n"
                  << "    host-USM path   : "
                  << (host_intercoop ? "TRUE" : "FALSE") << '\n'
                  << "    shared-USM path : "
                  << (shared_intercoop ? "TRUE" : "FALSE") << '\n'
                  << "    any path        : "
                  << (intercoop ? "TRUE" : "FALSE") << '\n';

        if(host_intercoop) {
          ++host_intercoop_devices;
          if(device.is_gpu())
            ++gpu_host_intercoop_devices;
        }
        if(shared_intercoop) {
          ++shared_intercoop_devices;
          if(device.is_gpu())
            ++gpu_shared_intercoop_devices;
        }
        if(intercoop) {
          ++intercoop_devices;
          if(device.is_gpu())
            ++gpu_intercoop_devices;
        }

        TestPatch2Execution(device, state, options);
      } catch(const sycl::exception& e) {
        state.Fail(std::string{"SYCL exception while testing device: "} + e.what());
      } catch(const std::exception& e) {
        state.Fail(std::string{"std::exception while testing device: "} + e.what());
      } catch(...) {
        state.Fail("unknown exception while testing device");
      }
    }

    std::cout << "\n============================================================\n";
    std::cout << "Patch 1 context/intersection tests\n";
    try {
      TestAllContexts(state);
    } catch(const sycl::exception& e) {
      state.Fail(std::string{"context suite SYCL exception: "} + e.what());
    } catch(const std::exception& e) {
      state.Fail(std::string{"context suite std::exception: "} + e.what());
    }

    if(options.require_intercoop) {
      state.Check(intercoop_devices > 0,
                  "--require-intercoop: at least one tested device qualifies");
    }

    if(options.require_gpu_intercoop) {
      state.Check(gpu_intercoop_devices > 0,
                  "--require-gpu-intercoop: at least one tested GPU qualifies");
    }

    std::cout << "\n============================================================\n";
    std::cout << "FINAL SUMMARY\n";
    std::cout << "  PASS : " << state.passed << '\n';
    std::cout << "  FAIL : " << state.failed << '\n';
    std::cout << "  SKIP : " << state.skipped << '\n';
    std::cout << "  InterCoOp-qualified devices          : " << intercoop_devices << '\n';
    std::cout << "  InterCoOp-qualified GPU devices      : " << gpu_intercoop_devices << '\n';
    std::cout << "  Host-USM qualified devices           : " << host_intercoop_devices << '\n';
    std::cout << "  Shared-USM qualified devices         : " << shared_intercoop_devices << '\n';
    std::cout << "  Host-USM qualified GPU devices       : " << gpu_host_intercoop_devices << '\n';
    std::cout << "  Shared-USM qualified GPU devices     : " << gpu_shared_intercoop_devices << '\n';

    if(state.failed == 0) {
      std::cout << "\nRESULT: PASS\n";
      return 0;
    }

    std::cout << "\nRESULT: FAIL\n";
    return 1;
  } catch(const std::exception& e) {
    std::cerr << "FATAL: " << e.what() << '\n';
    return 102;
  }
}
