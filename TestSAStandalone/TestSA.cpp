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
// AdaptiveCpp Patch 1 + Patch 2 validation suite
// =============================================================================
//
// This program targets the two atomic-capability patches directly.
//
// Patch 1 coverage:
//   * info::device::atomic_memory_order_capabilities
//   * info::device::atomic_memory_scope_capabilities
//   * info::context::{atomic_memory_order_capabilities,
//                    atomic_memory_scope_capabilities}
//   * device::has(aspect::usm_atomic_shared_allocations) public dispatch
//   * conservative/fallback capability invariants
//   * context capability intersection semantics
//
// Patch 2 coverage:
//   * backend-specific capability reporting for CUDA/HIP/Level Zero/OpenCL/OpenMP
//   * device-scope integral/floating atomic execution
//   * work-group-scope local atomic execution
//   * system-scope shared-USM integral/floating atomic execution
//   * 64-bit store/load/exchange/CAS/add/sub/and/or/xor/min/max
//   * release/acquire publication CPU -> device
//   * release/acquire publication device -> CPU
//   * concurrent CPU + device acq_rel fetch_add on one shared uint64_t
//   * CPU/device compare-exchange races on shared uint64_t cells
//   * explicit atomic_fence smoke execution
//
// IMPORTANT:
//   A capability that is NOT advertised causes the corresponding execution test
//   to SKIP. A capability that IS advertised and then fails execution is a FAIL.
//   This is intentional: unsupported hardware is not a bug, false advertising is.
//
// Suggested builds:
//   acpp -std=c++20 --acpp-targets=generic TestSA-Patch1-Patch2.cpp -o TestSA
//   acpp -std=c++20 --acpp-targets=cuda:sm_XX TestSA-Patch1-Patch2.cpp -o TestSA
//   acpp -std=c++20 --acpp-targets=hip:gfxXXXX TestSA-Patch1-Patch2.cpp -o TestSA
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
constexpr std::uint64_t kDeviceSpinLimit = 5'000'000ULL;
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

template<class T>
class SharedUsmArray {
public:
  SharedUsmArray(sycl::queue& queue, std::size_t count, std::size_t alignment)
      : queue_{&queue}, count_{count} {
    ptr_ = sycl::aligned_alloc_shared<T>(alignment, count, queue);
    if(!ptr_)
      throw std::runtime_error("sycl::aligned_alloc_shared returned nullptr");
  }

  ~SharedUsmArray() {
    if(ptr_)
      sycl::free(ptr_, *queue_);
  }

  SharedUsmArray(const SharedUsmArray&) = delete;
  SharedUsmArray& operator=(const SharedUsmArray&) = delete;

  T* get() noexcept { return ptr_; }
  const T* get() const noexcept { return ptr_; }
  T& operator[](std::size_t i) noexcept { return ptr_[i]; }
  const T& operator[](std::size_t i) const noexcept { return ptr_[i]; }
  std::size_t size() const noexcept { return count_; }

private:
  sycl::queue* queue_ = nullptr;
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
  state.Check(aspect_consistent(sycl::aspect::usm_shared_allocations),
              "usm_shared_allocations has() agrees with aspects list");
  state.Check(aspect_consistent(sycl::aspect::usm_atomic_shared_allocations),
              "PATCH 1 usm_atomic_shared_allocations dispatch agrees with aspects list");
  state.Check(aspect_consistent(sycl::aspect::atomic64),
              "atomic64 has() agrees with aspects list");

  if(usm_atomic_shared)
    state.Check(usm_shared,
                "atomic shared USM implies shared USM");

  std::cout << "  atomic64                    : " << atomic64 << '\n';
  std::cout << "  usm_device_allocations      : " << usm_device << '\n';
  std::cout << "  usm_shared_allocations      : " << usm_shared << '\n';
  std::cout << "  usm_atomic_shared_allocs    : " << usm_atomic_shared << '\n';
  std::cout << "  acq_rel                     : " << acq_rel << '\n';
  std::cout << "  system scope                : " << system_scope << '\n';
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
    state.Check(has_scope(sycl::memory_scope::system) == atomic_shared,
                "PATCH 2 GPU system scope matches atomic shared-USM qualification");
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
    state.Check(atomic_shared,
                "PATCH 2 OpenMP reports atomic shared USM");
    break;

  case sycl::backend::metal:
  case sycl::backend::vk:
    // These backends were not given Patch-2-specific overrides. Verify that the
    // Patch-1 conservative fallback remains usable rather than inventing claims.
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
void RunIntegralSharedSequence(sycl::queue& q, TestState& state,
                               std::string_view label) {
  SharedUsmArray<T> memory{q, 17, SharedAtomicAlignment<T>()};
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
void RunFloatingSharedSequence(sycl::queue& q, TestState& state,
                               std::string_view label) {
  SharedUsmArray<T> memory{q, 10, SharedAtomicAlignment<T>()};
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
void TestAtomicFenceSmoke(sycl::queue& q, const sycl::device& device,
                          TestState& state) {
  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();

  if(!Contains(orders, sycl::memory_order::acquire) ||
     !Contains(orders, sycl::memory_order::release) ||
     !Contains(scopes, sycl::memory_scope::system) ||
     !device.has(sycl::aspect::usm_shared_allocations)) {
    state.Skip("system atomic_fence smoke prerequisites unavailable");
    return;
  }

  SharedUsmArray<U32> marker{q, 1, alignof(U32)};
  marker[0] = 0;
  U32* ptr = marker.get();
  q.single_task([=]() {
     sycl::atomic_fence(sycl::memory_order::release,
                        sycl::memory_scope::system);
     ptr[0] = 1;
     sycl::atomic_fence(sycl::memory_order::acquire,
                        sycl::memory_scope::system);
   }).wait_and_throw();

  state.Check(marker[0] == 1,
              "system-scope acquire/release atomic_fence path executes");
}

// -----------------------------------------------------------------------------
// CPU -> GPU release/acquire publication on shared uint64 data.
// The kernel runs on a separate host thread so even a synchronous backend can
// wait for a host publication without deadlocking the test driver thread.
// -----------------------------------------------------------------------------
void TestCpuToDevicePublication64(sycl::queue& q, TestState& state) {
  // [0]=ready, [1]=go/flag, [2]=payload, [3]=observed, [4]=timeout marker
  SharedUsmArray<U64> m{q, 5, SharedAtomicAlignment<U64>()};
  for(std::size_t i = 0; i < m.size(); ++i)
    m[i] = 0;

  U64* p = m.get();
  std::exception_ptr worker_error;

  std::thread worker([&] {
    try {
      q.single_task([=]() {
         GlobalAtomic<U64, sycl::memory_scope::system> ready{p[0]};
         GlobalAtomic<U64, sycl::memory_scope::system> flag{p[1]};
         ready.store(1, sycl::memory_order::release,
                     sycl::memory_scope::system);

         U64 seen = 0;
         for(U64 spin = 0; spin < kDeviceSpinLimit && seen == 0; ++spin)
           seen = flag.load(sycl::memory_order::acquire,
                            sycl::memory_scope::system);

         if(seen == 1)
           p[3] = p[2];
         else
           p[4] = 1;
       }).wait_and_throw();
    } catch(...) {
      worker_error = std::current_exception();
    }
  });

  const bool ready_seen = HostWaitEqual(m[0], U64{1});
  if(!ready_seen)
    state.Fail("CPU->device publication: host did not observe device ready flag");
  else
    state.Pass("CPU->device publication: host observed device ready flag");

  constexpr U64 magic = 0xD1CE'B00C'CAFEBABEULL;
  m[2] = magic;
  std::atomic_ref<U64>{m[1]}.store(1, std::memory_order_release);

  worker.join();
  RethrowThreadException(worker_error);

  state.Check(m[4] == 0,
              "CPU->device publication: device observed release flag before spin timeout");
  state.Check(m[3] == magic,
              "CPU release + device system-scope acquire publishes plain payload");
}

// -----------------------------------------------------------------------------
// GPU -> CPU release/acquire publication.
// -----------------------------------------------------------------------------
void TestDeviceToCpuPublication64(sycl::queue& q, TestState& state) {
  // [0]=flag, [1]=payload
  SharedUsmArray<U64> m{q, 2, SharedAtomicAlignment<U64>()};
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
  if(observed) {
    state.Pass("device->CPU publication: host acquired device release flag");
    state.Check(m[1] == magic,
                "device system-scope release + CPU acquire publishes plain payload");
  } else {
    state.Fail("device->CPU publication: host timed out waiting for release flag");
  }

  worker.join();
  RethrowThreadException(worker_error);
}

// -----------------------------------------------------------------------------
// CPU + device simultaneous fetch_add on the same uint64_t.
// This is the strongest practical InterCoOp stress in this file.
// -----------------------------------------------------------------------------
void TestConcurrentHostDeviceAdd64(sycl::queue& q, TestState& state,
                                   std::size_t stress) {
  // [0]=counter, [1]=ready, [2]=go, [3]=device timed out, [4]=done
  SharedUsmArray<U64> m{q, 5, SharedAtomicAlignment<U64>()};
  for(std::size_t i = 0; i < m.size(); ++i)
    m[i] = 0;

  U64* p = m.get();
  const U64 iterations = static_cast<U64>(stress);
  std::exception_ptr worker_error;

  std::thread worker([&] {
    try {
      q.single_task([=]() {
         GlobalAtomic<U64, sycl::memory_scope::system> counter{p[0]};
         GlobalAtomic<U64, sycl::memory_scope::system> ready{p[1]};
         GlobalAtomic<U64, sycl::memory_scope::system> go{p[2]};
         GlobalAtomic<U64, sycl::memory_scope::system> done{p[4]};

         ready.store(1, sycl::memory_order::release,
                     sycl::memory_scope::system);

         U64 started = 0;
         for(U64 spin = 0; spin < kDeviceSpinLimit && started == 0; ++spin)
           started = go.load(sycl::memory_order::acquire,
                             sycl::memory_scope::system);

         if(started != 1) {
           p[3] = 1;
           done.store(1, sycl::memory_order::release,
                      sycl::memory_scope::system);
           return;
         }

         for(U64 i = 0; i < iterations; ++i)
           counter.fetch_add(1, sycl::memory_order::acq_rel,
                             sycl::memory_scope::system);

         done.store(1, sycl::memory_order::release,
                    sycl::memory_scope::system);
       }).wait_and_throw();
    } catch(...) {
      worker_error = std::current_exception();
    }
  });

  const bool ready_seen = HostWaitEqual(m[1], U64{1});
  state.Check(ready_seen,
              "concurrent add: CPU observes device ready release");

  std::atomic_ref<U64> counter{m[0]};
  std::atomic_ref<U64> go{m[2]};
  go.store(1, std::memory_order_release);

  for(U64 i = 0; i < iterations; ++i)
    counter.fetch_add(1, std::memory_order_acq_rel);

  worker.join();
  RethrowThreadException(worker_error);

  state.Check(m[3] == 0,
              "concurrent add: device observed host go release before timeout");
  state.Check(m[4] == 1,
              "concurrent add: device completed its atomic loop");

  const U64 expected = iterations * 2;
  const U64 actual = counter.load(std::memory_order_acquire);
  if(actual == expected) {
    state.Pass("concurrent CPU + device uint64 fetch_add has exact 2*N result");
  } else {
    std::ostringstream out;
    out << "concurrent CPU + device uint64 fetch_add expected " << expected
        << " got " << actual;
    state.Fail(out.str());
  }
}

// -----------------------------------------------------------------------------
// Batch of independent CPU/device CAS races. Exactly one side may change each
// cell from zero. This catches incoherent compare-exchange behavior.
// -----------------------------------------------------------------------------
void TestHostDeviceCasRace64(sycl::queue& q, TestState& state) {
  constexpr std::size_t N = 1024;
  SharedUsmArray<U64> cells{q, N, SharedAtomicAlignment<U64>()};
  SharedUsmArray<U64> gpu_success{q, N, SharedAtomicAlignment<U64>()};
  for(std::size_t i = 0; i < N; ++i) {
    cells[i] = 0;
    gpu_success[i] = 0;
  }

  U64* c = cells.get();
  U64* g = gpu_success.get();

  auto event = q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> id) {
    const std::size_t i = id[0];
    GlobalAtomic<U64, sycl::memory_scope::system> a{c[i]};
    U64 expected = 0;
    const bool won = a.compare_exchange_strong(
        expected, U64{2},
        sycl::memory_order::acq_rel,
        sycl::memory_order::acquire,
        sycl::memory_scope::system);
    g[i] = won ? 1 : 0;
  });

  std::vector<bool> host_success(N, false);
  for(std::size_t i = 0; i < N; ++i) {
    std::atomic_ref<U64> a{cells[i]};
    U64 expected = 0;
    host_success[i] = a.compare_exchange_strong(
        expected, U64{1},
        std::memory_order_acq_rel,
        std::memory_order_acquire);
  }

  event.wait_and_throw();

  std::size_t bad = 0;
  for(std::size_t i = 0; i < N; ++i) {
    const U64 value = std::atomic_ref<U64>{cells[i]}.load(std::memory_order_acquire);
    const bool gpu_won = gpu_success[i] != 0;
    const bool host_won = host_success[i];
    const bool exactly_one = gpu_won != host_won;
    const bool value_matches = (gpu_won && value == 2) || (host_won && value == 1);
    if(!exactly_one || !value_matches)
      ++bad;
  }

  if(bad == 0)
    state.Pass("1024 CPU/device CAS races each have exactly one coherent winner");
  else {
    std::ostringstream out;
    out << "CPU/device CAS race produced " << bad << " invalid cells out of " << N;
    state.Fail(out.str());
  }
}

bool IsInterCoOpQualified(const sycl::device& device) {
  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();

  return device.has(sycl::aspect::atomic64) &&
         device.has(sycl::aspect::usm_shared_allocations) &&
         device.has(sycl::aspect::usm_atomic_shared_allocations) &&
         Contains(orders, sycl::memory_order::acq_rel) &&
         Contains(orders, sycl::memory_order::acquire) &&
         Contains(orders, sycl::memory_order::release) &&
         Contains(scopes, sycl::memory_scope::system);
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
  const bool system_scope = Contains(scopes, sycl::memory_scope::system);
  const bool usm_device = device.has(sycl::aspect::usm_device_allocations);
  const bool usm_shared = device.has(sycl::aspect::usm_shared_allocations);
  const bool atomic_shared =
      device.has(sycl::aspect::usm_atomic_shared_allocations);
  const bool atomic64 = device.has(sycl::aspect::atomic64);
  const bool fp64 = device.has(sycl::aspect::fp64);

  std::cout << "\n  -- Patch 2 execution tests --\n";

  TestWorkGroupAtomic(q, device, state);

  if(device_scope && usm_device) {
    RunIntegralDeviceUsmSequence<U32, sycl::memory_scope::device, sycl::memory_order::relaxed>(
        q, state,
        "uint32 device-scope complete RMW sequence / relaxed");
    RunFloatingDeviceUsmSequence<float, sycl::memory_scope::device, sycl::memory_order::relaxed>(
        q, state,
        "float device-scope RMW sequence / relaxed");

    if(acq_rel) {
      RunIntegralDeviceUsmSequence<U32, sycl::memory_scope::device, sycl::memory_order::acq_rel>(
          q, state,
          "uint32 device-scope complete RMW sequence / acq_rel");
      RunFloatingDeviceUsmSequence<float, sycl::memory_scope::device, sycl::memory_order::acq_rel>(
          q, state,
          "float device-scope RMW sequence / acq_rel");
    }

    if(seq_cst) {
      RunIntegralDeviceUsmSequence<U32, sycl::memory_scope::device, sycl::memory_order::seq_cst>(
          q, state,
          "uint32 device-scope complete RMW sequence / seq_cst");
      RunFloatingDeviceUsmSequence<float, sycl::memory_scope::device, sycl::memory_order::seq_cst>(
          q, state,
          "float device-scope RMW sequence / seq_cst");
    }

    if(atomic64) {
      RunIntegralDeviceUsmSequence<U64, sycl::memory_scope::device, sycl::memory_order::relaxed>(
          q, state,
          "uint64 device-scope complete RMW sequence / relaxed");
      if(acq_rel)
        RunIntegralDeviceUsmSequence<U64, sycl::memory_scope::device, sycl::memory_order::acq_rel>(
            q, state,
            "uint64 device-scope complete RMW sequence / acq_rel");
      if(seq_cst)
        RunIntegralDeviceUsmSequence<U64, sycl::memory_scope::device, sycl::memory_order::seq_cst>(
            q, state,
            "uint64 device-scope complete RMW sequence / seq_cst");
      TestDeviceContention64(q, device, state, options.stress);
    } else {
      state.Skip("uint64 device-scope sequence: atomic64 not advertised");
      state.Skip("uint64 device contention: atomic64 not advertised");
    }

    if(fp64 && atomic64) {
      RunFloatingDeviceUsmSequence<double, sycl::memory_scope::device, sycl::memory_order::relaxed>(
          q, state,
          "double device-scope RMW sequence / relaxed");
      if(acq_rel)
        RunFloatingDeviceUsmSequence<double, sycl::memory_scope::device, sycl::memory_order::acq_rel>(
            q, state,
            "double device-scope RMW sequence / acq_rel");
      if(seq_cst)
        RunFloatingDeviceUsmSequence<double, sycl::memory_scope::device, sycl::memory_order::seq_cst>(
            q, state,
            "double device-scope RMW sequence / seq_cst");
    } else {
      state.Skip("double device-scope atomic sequence: fp64+atomic64 unavailable");
    }
  } else {
    state.Skip("device-scope RMW suite: device scope or device USM unavailable");
  }

  if(system_scope && usm_shared && atomic_shared) {
    RunIntegralSharedSequence<U32, sycl::memory_scope::system, sycl::memory_order::relaxed>(
        q, state,
        "uint32 system-scope complete RMW sequence / relaxed");
    RunFloatingSharedSequence<float, sycl::memory_scope::system, sycl::memory_order::relaxed>(
        q, state,
        "float system-scope RMW sequence / relaxed");

    if(acq_rel) {
      RunIntegralSharedSequence<U32, sycl::memory_scope::system, sycl::memory_order::acq_rel>(
          q, state,
          "uint32 system-scope complete RMW sequence / acq_rel");
      RunFloatingSharedSequence<float, sycl::memory_scope::system, sycl::memory_order::acq_rel>(
          q, state,
          "float system-scope RMW sequence / acq_rel");
    }

    if(seq_cst) {
      RunIntegralSharedSequence<U32, sycl::memory_scope::system, sycl::memory_order::seq_cst>(
          q, state,
          "uint32 system-scope complete RMW sequence / seq_cst");
      RunFloatingSharedSequence<float, sycl::memory_scope::system, sycl::memory_order::seq_cst>(
          q, state,
          "float system-scope RMW sequence / seq_cst");
    }

    if(atomic64) {
      RunIntegralSharedSequence<U64, sycl::memory_scope::system, sycl::memory_order::relaxed>(
          q, state,
          "uint64 system-scope complete RMW sequence / relaxed");
      if(acq_rel) {
        RunIntegralSharedSequence<U64, sycl::memory_scope::system, sycl::memory_order::acq_rel>(
            q, state,
            "uint64 system-scope complete RMW sequence / acq_rel");
      }
      if(seq_cst) {
        RunIntegralSharedSequence<U64, sycl::memory_scope::system, sycl::memory_order::seq_cst>(
            q, state,
            "uint64 system-scope complete RMW sequence / seq_cst");
      }
    } else {
      state.Skip("uint64 system-scope sequence: atomic64 not advertised");
    }

    if(fp64 && atomic64) {
      RunFloatingSharedSequence<double, sycl::memory_scope::system, sycl::memory_order::relaxed>(
          q, state,
          "double system-scope RMW sequence / relaxed");
      if(acq_rel)
        RunFloatingSharedSequence<double, sycl::memory_scope::system, sycl::memory_order::acq_rel>(
            q, state,
            "double system-scope RMW sequence / acq_rel");
      if(seq_cst)
        RunFloatingSharedSequence<double, sycl::memory_scope::system, sycl::memory_order::seq_cst>(
            q, state,
            "double system-scope RMW sequence / seq_cst");
    } else {
      state.Skip("double system-scope atomic sequence: fp64+atomic64 unavailable");
    }

    TestAtomicFenceSmoke(q, device, state);
  } else {
    state.Skip("system-scope RMW suite: system scope or atomic shared USM unavailable");
  }

  if(IsInterCoOpQualified(device)) {
    std::cout << "\n  -- uint64 host/device InterCoOp semantic tests --\n";
    TestCpuToDevicePublication64(q, state);
    TestDeviceToCpuPublication64(q, state);
    TestConcurrentHostDeviceAdd64(q, state, options.stress);
    TestHostDeviceCasRace64(q, state);
  } else {
    state.Skip("uint64 host/device InterCoOp suite: device does not advertise full contract");
  }

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
        << "AdaptiveCpp Patch 1 + Patch 2 comprehensive atomic validation\n"
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

        const bool intercoop = IsInterCoOpQualified(device);
        std::cout << "\n  SuperNova uint64 InterCoOp qualification: "
                  << (intercoop ? "TRUE" : "FALSE") << '\n';
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
    std::cout << "  InterCoOp-qualified devices     : " << intercoop_devices << '\n';
    std::cout << "  InterCoOp-qualified GPU devices : " << gpu_intercoop_devices << '\n';

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
