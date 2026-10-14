#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{

template<class T>
bool Contains(const std::vector<T>& values, T value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

const char* OrderName(sycl::memory_order order)
{
    switch (order)
    {
        case sycl::memory_order::relaxed: return "relaxed";
        case sycl::memory_order::acquire: return "acquire";
        case sycl::memory_order::release: return "release";
        case sycl::memory_order::acq_rel: return "acq_rel";
        case sycl::memory_order::seq_cst: return "seq_cst";
    }

    return "unknown";
}

const char* ScopeName(sycl::memory_scope scope)
{
    switch (scope)
    {
        case sycl::memory_scope::work_item:  return "work_item";
        case sycl::memory_scope::sub_group:  return "sub_group";
        case sycl::memory_scope::work_group: return "work_group";
        case sycl::memory_scope::device:     return "device";
        case sycl::memory_scope::system:     return "system";
    }

    return "unknown";
}

void PrintOrders(const std::vector<sycl::memory_order>& orders)
{
    std::cout << "{ ";
    for (const auto order : orders)
        std::cout << OrderName(order) << ' ';
    std::cout << '}';
}

void PrintScopes(const std::vector<sycl::memory_scope>& scopes)
{
    std::cout << "{ ";
    for (const auto scope : scopes)
        std::cout << ScopeName(scope) << ' ';
    std::cout << '}';
}

template<class T>
bool SameVector(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    return lhs == rhs;
}

bool TestDevicePatch1(const sycl::device& device, std::size_t index)
{
    std::cout << "\n============================================================\n";
    std::cout << "Device " << index << ": "
              << device.get_info<sycl::info::device::name>() << '\n';
    std::cout << "Backend enum value: "
              << static_cast<int>(device.get_backend()) << '\n';

    // ---------------------------------------------------------------------
    // PATCH 1 TEST A:
    // The two new device info descriptors must exist, compile and dispatch
    // through hardware_context.
    // ---------------------------------------------------------------------
    const auto orders =
        device.get_info<
            sycl::info::device::atomic_memory_order_capabilities>();

    const auto scopes =
        device.get_info<
            sycl::info::device::atomic_memory_scope_capabilities>();

    static_assert(
        std::is_same_v<
            std::remove_cvref_t<decltype(orders)>,
            std::vector<sycl::memory_order>>,
        "Patch 1 failure: device order query has the wrong return type.");

    static_assert(
        std::is_same_v<
            std::remove_cvref_t<decltype(scopes)>,
            std::vector<sycl::memory_scope>>,
        "Patch 1 failure: device scope query has the wrong return type.");

    std::cout << "Device atomic orders : ";
    PrintOrders(orders);
    std::cout << '\n';

    std::cout << "Device atomic scopes : ";
    PrintScopes(scopes);
    std::cout << '\n';

    // ---------------------------------------------------------------------
    // PATCH 1 TEST B:
    // Accp(6).h is Patch-1-only: there are no backend overrides yet.
    // Therefore hardware_context's conservative defaults must be returned
    // exactly:
    //
    //     order  = { relaxed }
    //     scopes = { sub_group, work_group }
    //
    // If this fails BEFORE Patch 2, Patch 1 runtime plumbing is wrong.
    // If this fails AFTER Patch 2, that is expected because Patch 2 replaces
    // these conservative defaults with backend-specific answers.
    // ---------------------------------------------------------------------
    const std::vector<sycl::memory_order> expected_orders{
        sycl::memory_order::relaxed
    };

    const std::vector<sycl::memory_scope> expected_scopes{
        sycl::memory_scope::sub_group,
        sycl::memory_scope::work_group
    };

    if (!SameVector(orders, expected_orders))
    {
        std::cerr
            << "FAIL: Patch 1 device-order plumbing did not return the "
               "conservative Patch-1-only default.\n";
        return false;
    }

    if (!SameVector(scopes, expected_scopes))
    {
        std::cerr
            << "FAIL: Patch 1 device-scope plumbing did not return the "
               "conservative Patch-1-only defaults.\n";
        return false;
    }

    // ---------------------------------------------------------------------
    // PATCH 1 TEST C:
    // Exercise the public device::has() branch that Patch 1 added for
    // usm_atomic_shared_allocations.
    //
    // We do NOT require it to be true on every backend. Patch 1's job is
    // dispatching the query correctly, not changing each backend's answer.
    // ---------------------------------------------------------------------
    const bool atomic_shared =
        device.has(sycl::aspect::usm_atomic_shared_allocations);

    const auto aspects =
        device.get_info<sycl::info::device::aspects>();

    const bool listed_in_aspects =
        Contains(aspects, sycl::aspect::usm_atomic_shared_allocations);

    std::cout << "usm_atomic_shared_allocations via has()     : "
              << atomic_shared << '\n';
    std::cout << "usm_atomic_shared_allocations in aspects(): "
              << listed_in_aspects << '\n';

    if (atomic_shared != listed_in_aspects)
    {
        std::cerr
            << "FAIL: public aspect dispatch and device aspect list disagree.\n";
        return false;
    }

    // ---------------------------------------------------------------------
    // PATCH 1 TEST D:
    // Test the new context-level descriptors and their intersection logic.
    //
    // AdaptiveCpp may automatically include its host device in a context, so
    // compute the expected intersection manually from context.get_devices()
    // rather than assuming the context contains only `device`.
    // ---------------------------------------------------------------------
    const sycl::context context{device};

    const auto context_orders =
        context.get_info<
            sycl::info::context::atomic_memory_order_capabilities>();

    const auto context_scopes =
        context.get_info<
            sycl::info::context::atomic_memory_scope_capabilities>();

    static_assert(
        std::is_same_v<
            std::remove_cvref_t<decltype(context_orders)>,
            std::vector<sycl::memory_order>>,
        "Patch 1 failure: context order query has the wrong return type.");

    static_assert(
        std::is_same_v<
            std::remove_cvref_t<decltype(context_scopes)>,
            std::vector<sycl::memory_scope>>,
        "Patch 1 failure: context scope query has the wrong return type.");

    const auto context_devices = context.get_devices();

    std::cout << "Context device count : "
              << context_devices.size() << '\n';

    std::cout << "Context atomic orders: ";
    PrintOrders(context_orders);
    std::cout << '\n';

    std::cout << "Context atomic scopes: ";
    PrintScopes(context_scopes);
    std::cout << '\n';

    // Compute the same intersection independently from the device queries.
    const std::vector<sycl::memory_order> canonical_orders{
        sycl::memory_order::relaxed,
        sycl::memory_order::acquire,
        sycl::memory_order::release,
        sycl::memory_order::acq_rel,
        sycl::memory_order::seq_cst
    };

    const std::vector<sycl::memory_scope> canonical_scopes{
        sycl::memory_scope::sub_group,
        sycl::memory_scope::work_group,
        sycl::memory_scope::device,
        sycl::memory_scope::system
    };

    std::vector<sycl::memory_order> manually_intersected_orders;
    for (const auto order : canonical_orders)
    {
        bool supported_by_all = true;

        for (const auto& context_device : context_devices)
        {
            const auto device_orders =
                context_device.get_info<
                    sycl::info::device::atomic_memory_order_capabilities>();

            if (!Contains(device_orders, order))
            {
                supported_by_all = false;
                break;
            }
        }

        if (supported_by_all)
            manually_intersected_orders.push_back(order);
    }

    std::vector<sycl::memory_scope> manually_intersected_scopes;
    for (const auto scope : canonical_scopes)
    {
        bool supported_by_all = true;

        for (const auto& context_device : context_devices)
        {
            const auto device_scopes =
                context_device.get_info<
                    sycl::info::device::atomic_memory_scope_capabilities>();

            if (!Contains(device_scopes, scope))
            {
                supported_by_all = false;
                break;
            }
        }

        if (supported_by_all)
            manually_intersected_scopes.push_back(scope);
    }

    if (!SameVector(context_orders, manually_intersected_orders))
    {
        std::cerr
            << "FAIL: context atomic-order intersection is incorrect.\n";
        return false;
    }

    if (!SameVector(context_scopes, manually_intersected_scopes))
    {
        std::cerr
            << "FAIL: context atomic-scope intersection is incorrect.\n";
        return false;
    }

    // Because this test is intentionally Patch-1-only, every context device
    // should still inherit the same conservative defaults.
    if (!SameVector(context_orders, expected_orders))
    {
        std::cerr
            << "FAIL: Patch-1-only context orders are not {relaxed}.\n";
        return false;
    }

    if (!SameVector(context_scopes, expected_scopes))
    {
        std::cerr
            << "FAIL: Patch-1-only context scopes are not "
               "{sub_group, work_group}.\n";
        return false;
    }

    std::cout << "Device Patch 1 checks: PASS\n";
    return true;
}

} // namespace

int main()
{
    try
    {
        const auto devices = sycl::device::get_devices();

        if (devices.empty())
        {
            std::cerr << "FAIL: AdaptiveCpp exposed no SYCL devices.\n";
            return 1;
        }

        std::cout
            << "AdaptiveCpp Patch 1 standalone validation\n"
            << "Devices discovered: " << devices.size() << "\n";

        for (std::size_t i = 0; i < devices.size(); ++i)
        {
            if (!TestDevicePatch1(devices[i], i))
                return 2;
        }

        std::cout
            << "\n============================================================\n"
            << "PATCH 1: PASS\n"
            << "Verified:\n"
            << "  [1] device atomic-order descriptor compiles and dispatches\n"
            << "  [2] device atomic-scope descriptor compiles and dispatches\n"
            << "  [3] Patch-1 conservative runtime defaults are exact\n"
            << "  [4] usm_atomic_shared_allocations public aspect dispatch works\n"
            << "  [5] context atomic-order descriptor works\n"
            << "  [6] context atomic-scope descriptor works\n"
            << "  [7] context capability intersection is correct\n"
            << "\n"
            << "This test intentionally does NOT test Patch 2, InterCoOp,\n"
            << "system scope, acq_rel support, or atomic execution.\n";

        return 0;
    }
    catch (const sycl::exception& error)
    {
        std::cerr
            << "FAIL: SYCL exception: "
            << error.what() << '\n';
        return 3;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "FAIL: std::exception: "
            << error.what() << '\n';
        return 4;
    }
}
