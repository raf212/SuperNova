

#pragma once

// SuperNova APC/Fabric paper-quality systems test kit (C++20)
//
// The benchmark sections deliberately distinguish:
//   1) a minimal one-parent vector forest lower bound,
//   2) a feature-matched fixed-capacity vector DAG with one global mutation mutex,
//   3) APC/Fabric with public generation/transaction/read contracts.
//
// Baselines are not claimed to provide APC/Fabric relocation, schema protocols,
// generation-safe handles, retirement/ABA protection, or concurrent reader semantics.
// Those properties are tested separately as SuperNova correctness properties.
//
// Put this file in core/TestFiles and compile a tiny runner:
//
//   #include "TestKit.hpp"
//   int main() { return APCDAGTests::RunAll(); }
//
// Add core/headers to the compiler include path and link the production .cpp files.
// Tests 1-5 and 7 use only public APC/Fabric operations. Correctness paths resolve
// returned APC facades by slab slot identity; they never compare host-object addresses.
// Timed traversal in Test 1 deliberately excludes this TestKit-only facade->index
// conversion so adapter bookkeeping is not charged to APC/Fabric traversal.
// Test 6 uses a read-only derived Fabric probe to verify the compact schema-table
// geometry and protocol storage. Test 8 exercises whole-slab Save/Attach/Detach
// relocation without involving GHGF.

#ifndef APC_DAG_TEST_EXTERNAL_TYPES
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"

#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace APCDAGTests
{
using namespace BidirectionalInMemGraph;

using Clock = std::chrono::steady_clock;
using ReadOperation = FabricToAPCLinker::SeqLockedOperation;

static_assert(ADS::META_CELL_COUNT == 8u);
static_assert(sizeof(SchemaDefinition::RegionSchemaRecord) == 5u * sizeof(std::uint64_t));
static_assert(alignof(SchemaDefinition::RegionSchemaRecord) == alignof(std::uint64_t));
static_assert(std::is_trivially_copyable_v<SchemaDefinition::RegionSchemaRecord>);
static_assert(std::is_trivially_destructible_v<SchemaDefinition::RegionSchemaRecord>);

enum class Axis : std::uint8_t
{
    HORIZONTAL,
    VERTICAL
};

constexpr FabricSegments EdgeTableForAxis(Axis axis) noexcept
{
    return axis == Axis::HORIZONTAL
        ? FabricSegments::VALUE_PARENT_EDGE_TABLE_H
        : FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V;
}

enum class Result : std::uint8_t
{
    PASS,
    FAIL
};

constexpr const char* ResultName(Result result) noexcept
{
    return result == Result::PASS ? "PASS" : "FAIL";
}

inline void Banner(const char* title)
{
    std::cout
        << "\n================================================================================\n"
        << title
        << "\n================================================================================\n";
}

inline void PrintBenchmarkEnvironment()
{
    std::cout
        << "\nBENCHMARK CONTEXT\n";

#if defined(_MSC_VER) && defined(_MSVC_LANG)
    std::cout
        << "  C++ language level      : " << _MSVC_LANG << " (_MSVC_LANG)\n"
        << "  legacy __cplusplus      : " << __cplusplus << '\n';
#else
    std::cout
        << "  C++ language level      : " << __cplusplus << '\n';
#endif

    std::cout
        << "  pointer width           : " << (sizeof(void*) * 8u) << " bits\n"
        << "  hardware_concurrency    : " << std::thread::hardware_concurrency() << '\n'
        << "  uint64 atomic lock-free : "
        << (std::atomic<std::uint64_t>::is_always_lock_free ? "YES" : "NO") << '\n'
#ifdef NDEBUG
        << "  assertions              : disabled (NDEBUG)\n";
#else
        << "  assertions              : enabled\n";
#endif

#if defined(_MSC_VER)
    std::cout << "  compiler                : MSVC " << _MSC_VER << '\n';
#elif defined(__clang__)
    std::cout << "  compiler                : Clang "
        << __clang_major__ << '.' << __clang_minor__ << '.' << __clang_patchlevel__ << '\n';
#elif defined(__GNUC__)
    std::cout << "  compiler                : GCC "
        << __GNUC__ << '.' << __GNUC_MINOR__ << '.' << __GNUC_PATCHLEVEL__ << '\n';
#else
    std::cout << "  compiler                : unknown\n";
#endif

    std::cout
        << "  note                    : CPU model, clock policy, compiler flags, and OS\n"
        << "                            should be recorded externally for publication.\n";
}

inline double Ratio(double numerator, double denominator) noexcept
{
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

template <std::size_t N>
double Median(std::array<double, N> samples)
{
    static_assert(N > 0u);
    std::sort(samples.begin(), samples.end());
    return samples[N / 2u];
}

inline void PerturbSchedule(std::uint64_t value) noexcept
{
    if ((value & 63u) == 0u)
    {
        std::this_thread::yield();
    }
}

struct ReadResult
{
    static constexpr std::size_t NO_NODE = std::numeric_limits<std::size_t>::max();
    static constexpr std::uint32_t NO_LOCATOR = UINT32_MAX;

    std::size_t Node = NO_NODE;
    std::uint32_t Locator = NO_LOCATOR;
    ReadOperation Outcome = ReadOperation::NONE;
    bool NodePresent = false;

    bool IsFound() const noexcept
    {
        return Outcome == ReadOperation::FOUND &&
            NodePresent &&
            Node != NO_NODE &&
            Locator != NO_LOCATOR;
    }

    bool IsNone() const noexcept
    {
        return Outcome == ReadOperation::NONE &&
            !NodePresent &&
            Node == NO_NODE &&
            Locator == NO_LOCATOR;
    }

    bool IsRetry() const noexcept
    {
        return Outcome == ReadOperation::RETRY &&
            !NodePresent &&
            Node == NO_NODE &&
            Locator == NO_LOCATOR;
    }

    bool ContractValid() const noexcept
    {
        return IsFound() || IsNone() || IsRetry();
    }
};

// Lightweight timed-read result. Unlike ReadResult, this does not require the
// benchmark adapter to revalidate a returned APC facade just to recover a logical
// test-node index. The real public Find* call still executes in full.
struct BenchmarkReadResult
{
    static constexpr std::size_t NO_NODE = ReadResult::NO_NODE;
    static constexpr std::uint32_t NO_LOCATOR = ReadResult::NO_LOCATOR;

    std::size_t NodeHint = NO_NODE;
    std::uint32_t Locator = NO_LOCATOR;
    ReadOperation Outcome = ReadOperation::NONE;
    bool ObjectPresent = false;

    bool IsFound() const noexcept
    {
        return Outcome == ReadOperation::FOUND &&
            ObjectPresent &&
            Locator != NO_LOCATOR;
    }

    bool IsNone() const noexcept
    {
        return Outcome == ReadOperation::NONE &&
            !ObjectPresent &&
            Locator == NO_LOCATOR;
    }

    bool IsRetry() const noexcept
    {
        return Outcome == ReadOperation::RETRY &&
            !ObjectPresent &&
            Locator == NO_LOCATOR;
    }

    bool ContractValid() const noexcept
    {
        return IsFound() || IsNone() || IsRetry();
    }

    bool HasNodeHint() const noexcept
    {
        return IsFound() && NodeHint != NO_NODE;
    }
};

struct ReadCounts
{
    std::uint64_t Found = 0u;
    std::uint64_t None = 0u;
    std::uint64_t Retry = 0u;
    std::uint64_t BadContract = 0u;

    void Observe(const ReadResult& read) noexcept
    {
        if (!read.ContractValid())
        {
            ++BadContract;
            return;
        }

        if (read.IsFound()) ++Found;
        else if (read.IsRetry()) ++Retry;
        else ++None;
    }

    void Observe(const BenchmarkReadResult& read) noexcept
    {
        if (!read.ContractValid())
        {
            ++BadContract;
            return;
        }

        if (read.IsFound()) ++Found;
        else if (read.IsRetry()) ++Retry;
        else ++None;
    }

    void Add(const ReadCounts& other) noexcept
    {
        Found += other.Found;
        None += other.None;
        Retry += other.Retry;
        BadContract += other.BadContract;
    }

    std::uint64_t Calls() const noexcept
    {
        return Found + None + Retry + BadContract;
    }
};

inline void PrintReadCounts(const char* label, const ReadCounts& counts)
{
    std::cout
        << "  " << std::left << std::setw(31) << label
        << " calls=" << std::right << std::setw(10) << counts.Calls()
        << " FOUND=" << std::setw(10) << counts.Found
        << " RETRY=" << std::setw(10) << counts.Retry
        << " NONE=" << std::setw(10) << counts.None
        << " bad=" << counts.BadContract << '\n';
}

// -----------------------------------------------------------------------------
// Global-mutex vector forest baseline.
// It deliberately has one parent per axis; that is the old forest lower bound.
// Quiescent reads are raw. Every mutation holds one mutex across the whole move.
// -----------------------------------------------------------------------------

template <std::size_t NodeCount, std::size_t PayloadWords>
class VectorLockedForest
{
    static constexpr std::uint32_t NIL = UINT32_MAX;

    struct AxisState
    {
        std::uint32_t Parent = NIL;
        std::uint32_t Previous = NIL;
        std::uint32_t Next = NIL;
        std::uint32_t First = NIL;
        std::uint32_t Last = NIL;
    };

    struct Node
    {
        AxisState H{};
        AxisState V{};
        std::array<std::uint64_t, PayloadWords> Payload{};
    };

public:
    bool Initialize() noexcept
    {
        Nodes_ = {};
        return true;
    }

    bool AddParent(
        std::size_t parent,
        std::size_t child,
        Axis axis,
        std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        std::lock_guard<std::mutex> lock(GraphMutex_);
        return AddUnlocked_(parent, child, axis);
    }

    bool RemoveParent(
        std::size_t parent,
        std::size_t child,
        Axis axis,
        std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        std::lock_guard<std::mutex> lock(GraphMutex_);
        return RemoveUnlocked_(parent, child, axis);
    }

    bool ReplaceParent(
        std::size_t old_parent,
        std::size_t new_parent,
        std::size_t child,
        Axis axis,
        std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        std::lock_guard<std::mutex> lock(GraphMutex_);

        if (
            old_parent >= NodeCount ||
            new_parent >= NodeCount ||
            child >= NodeCount ||
            old_parent == new_parent ||
            new_parent >= child ||
            Axis_(child, axis).Parent != old_parent
        )
        {
            return false;
        }

        if (!RemoveUnlocked_(old_parent, child, axis))
        {
            return false;
        }
        if (AddUnlocked_(new_parent, child, axis))
        {
            return true;
        }

        (void)AddUnlocked_(old_parent, child, axis);
        return false;
    }

    ReadResult FindParent(
        std::size_t child,
        Axis axis,
        std::uint8_t ordinal,
        std::uint32_t = 1u) noexcept
    {
        if (child >= NodeCount || ordinal != 0u)
        {
            return {};
        }

        const std::uint32_t parent = Axis_(child, axis).Parent;
        return parent == NIL
            ? ReadResult{}
            : ReadResult{parent, static_cast<std::uint32_t>(child), ReadOperation::FOUND, true};
    }

    ReadResult FindFirstChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t = 1u) noexcept
    {
        return ChildResult_(parent < NodeCount ? Axis_(parent, axis).First : NIL);
    }

    ReadResult FindLastChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t = 1u) noexcept
    {
        return ChildResult_(parent < NodeCount ? Axis_(parent, axis).Last : NIL);
    }

    ReadResult FindNextChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator,
        std::uint32_t = 1u) noexcept
    {
        if (parent >= NodeCount || locator >= NodeCount)
        {
            return {};
        }
        const AxisState& child = Axis_(locator, axis);
        if (child.Parent != parent)
        {
            return {};
        }
        return ChildResult_(child.Next);
    }

    ReadResult FindPreviousChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator,
        std::uint32_t = 1u) noexcept
    {
        if (parent >= NodeCount || locator >= NodeCount)
        {
            return {};
        }
        const AxisState& child = Axis_(locator, axis);
        if (child.Parent != parent)
        {
            return {};
        }
        return ChildResult_(child.Previous);
    }

    bool StorePayload(
        std::size_t node,
        std::uint32_t word,
        std::uint64_t value,
        bool atomic) noexcept
    {
        if (node >= NodeCount || word >= PayloadWords)
        {
            return false;
        }
        if (atomic)
        {
            std::atomic_ref<std::uint64_t>(Nodes_[node].Payload[word]).store(
                value,
                std::memory_order_release
            );
        }
        else
        {
            Nodes_[node].Payload[word] = value;
        }
        return true;
    }

    bool LoadPayload(
        std::size_t node,
        std::uint32_t word,
        std::uint64_t& value,
        bool atomic) noexcept
    {
        if (node >= NodeCount || word >= PayloadWords)
        {
            return false;
        }
        value = atomic
            ? std::atomic_ref<std::uint64_t>(Nodes_[node].Payload[word]).load(
                std::memory_order_acquire
            )
            : Nodes_[node].Payload[word];
        return true;
    }

    std::size_t ApproxStorageBytes() const noexcept
    {
        return sizeof(Nodes_);
    }

private:
    std::array<Node, NodeCount> Nodes_{};
    std::mutex GraphMutex_{};

    AxisState& Axis_(std::size_t node, Axis axis) noexcept
    {
        return axis == Axis::HORIZONTAL ? Nodes_[node].H : Nodes_[node].V;
    }

    const AxisState& Axis_(std::size_t node, Axis axis) const noexcept
    {
        return axis == Axis::HORIZONTAL ? Nodes_[node].H : Nodes_[node].V;
    }

    static ReadResult ChildResult_(std::uint32_t child) noexcept
    {
        return child == NIL
            ? ReadResult{}
            : ReadResult{child, child, ReadOperation::FOUND, true};
    }

    bool AddUnlocked_(std::size_t parent, std::size_t child, Axis axis) noexcept
    {
        if (
            parent >= NodeCount ||
            child >= NodeCount ||
            parent >= child
        )
        {
            return false;
        }

        AxisState& child_axis = Axis_(child, axis);
        AxisState& parent_axis = Axis_(parent, axis);
        if (child_axis.Parent != NIL)
        {
            return false;
        }

        child_axis.Parent = static_cast<std::uint32_t>(parent);
        child_axis.Previous = parent_axis.Last;
        child_axis.Next = NIL;

        if (parent_axis.Last == NIL)
        {
            parent_axis.First = static_cast<std::uint32_t>(child);
        }
        else
        {
            Axis_(parent_axis.Last, axis).Next = static_cast<std::uint32_t>(child);
        }
        parent_axis.Last = static_cast<std::uint32_t>(child);
        return true;
    }

    bool RemoveUnlocked_(std::size_t parent, std::size_t child, Axis axis) noexcept
    {
        if (parent >= NodeCount || child >= NodeCount)
        {
            return false;
        }

        AxisState& child_axis = Axis_(child, axis);
        AxisState& parent_axis = Axis_(parent, axis);
        if (child_axis.Parent != parent)
        {
            return false;
        }

        if (child_axis.Previous == NIL)
        {
            parent_axis.First = child_axis.Next;
        }
        else
        {
            Axis_(child_axis.Previous, axis).Next = child_axis.Next;
        }

        if (child_axis.Next == NIL)
        {
            parent_axis.Last = child_axis.Previous;
        }
        else
        {
            Axis_(child_axis.Next, axis).Previous = child_axis.Previous;
        }

        child_axis = {};
        return true;

    }
};

// -----------------------------------------------------------------------------
// Feature-matched fixed-capacity vector DAG baseline.
//
// This baseline supports the same two relation axes, the same parent<child DAG
// rule, the same configurable direct-parent capacity, bidirectional traversal,
// and atomic ReplaceParent under one global mutation mutex.
//
// It intentionally does NOT emulate APC/Fabric generations, schema protocols,
// relocation, use scopes, sequence-validated public reads, or retirement.
// Quiescent reads are raw and must not race with mutation.
// -----------------------------------------------------------------------------

template <
    std::size_t NodeCount,
    std::size_t PayloadWords,
    std::uint8_t ParentCapacity
>
class VectorLockedDAG
{
    static_assert(NodeCount > 0u);
    static_assert(ParentCapacity > 0u);

    static constexpr std::uint32_t NIL = UINT32_MAX;
    static constexpr std::size_t RELATION_COUNT =
        NodeCount * static_cast<std::size_t>(ParentCapacity);

    struct AxisState
    {
        std::array<std::uint32_t, ParentCapacity> ParentRelations{};
        std::uint32_t FirstChildRelation = NIL;
        std::uint32_t LastChildRelation = NIL;
    };

    struct Node
    {
        AxisState H{};
        AxisState V{};
        std::array<std::uint64_t, PayloadWords> Payload{};
    };

    struct Relation
    {
        std::uint32_t Parent = NIL;
        std::uint32_t Child = NIL;
        std::uint32_t Previous = NIL;
        std::uint32_t Next = NIL;
        bool Used = false;
    };

public:
    bool Initialize() noexcept
    {
        Nodes_ = {};
        HRelations_ = {};
        VRelations_ = {};

        for (Node& node : Nodes_)
        {
            node.H.ParentRelations.fill(NIL);
            node.V.ParentRelations.fill(NIL);
            node.H.FirstChildRelation = NIL;
            node.H.LastChildRelation = NIL;
            node.V.FirstChildRelation = NIL;
            node.V.LastChildRelation = NIL;
        }

        for (Relation& relation : HRelations_) relation = Relation{};
        for (Relation& relation : VRelations_) relation = Relation{};
        return true;
    }

    bool AddParent(
        std::size_t parent,
        std::size_t child,
        Axis axis,
        std::uint32_t = DEFAULT_MAX_TRIES
    ) noexcept
    {
        std::lock_guard<std::mutex> lock(GraphMutex_);
        return AddUnlocked_(parent, child, axis);
    }

    bool RemoveParent(
        std::size_t parent,
        std::size_t child,
        Axis axis,
        std::uint32_t = DEFAULT_MAX_TRIES
    ) noexcept
    {
        std::lock_guard<std::mutex> lock(GraphMutex_);
        return RemoveUnlocked_(parent, child, axis);
    }

    bool ReplaceParent(
        std::size_t old_parent,
        std::size_t new_parent,
        std::size_t child,
        Axis axis,
        std::uint32_t = DEFAULT_MAX_TRIES
    ) noexcept
    {
        std::lock_guard<std::mutex> lock(GraphMutex_);

        if (
            old_parent >= NodeCount ||
            new_parent >= NodeCount ||
            child >= NodeCount ||
            old_parent == new_parent ||
            new_parent >= child
        )
        {
            return false;
        }

        const std::uint32_t locator =
            FindParentRelationLocator_(child, old_parent, axis);

        if (
            locator == NIL ||
            FindParentRelationLocator_(child, new_parent, axis) != NIL
        )
        {
            return false;
        }

        Relation& relation = Relations_(axis)[locator];

        UnlinkFromParent_(locator, axis);

        relation.Parent = static_cast<std::uint32_t>(new_parent);
        relation.Previous = Axis_(new_parent, axis).LastChildRelation;
        relation.Next = NIL;

        AxisState& new_parent_axis = Axis_(new_parent, axis);
        if (new_parent_axis.LastChildRelation == NIL)
        {
            new_parent_axis.FirstChildRelation = locator;
        }
        else
        {
            Relations_(axis)[new_parent_axis.LastChildRelation].Next = locator;
        }
        new_parent_axis.LastChildRelation = locator;
        return true;
    }

    ReadResult FindParent(
        std::size_t child,
        Axis axis,
        std::uint8_t ordinal,
        std::uint32_t = 1u
    ) noexcept
    {
        if (child >= NodeCount || ordinal >= ParentCapacity)
        {
            return {};
        }

        const std::uint32_t locator =
            static_cast<std::uint32_t>(
                child * static_cast<std::size_t>(ParentCapacity) + ordinal
            );

        const Relation& relation = Relations_(axis)[locator];

        return relation.Used
            ? ReadResult{
                relation.Parent,
                locator,
                ReadOperation::FOUND,
                true
            }
            : ReadResult{};
    }

    ReadResult FindFirstChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t = 1u
    ) noexcept
    {
        return parent < NodeCount
            ? ChildResult_(Axis_(parent, axis).FirstChildRelation, axis)
            : ReadResult{};
    }

    ReadResult FindLastChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t = 1u
    ) noexcept
    {
        return parent < NodeCount
            ? ChildResult_(Axis_(parent, axis).LastChildRelation, axis)
            : ReadResult{};
    }

    ReadResult FindNextChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator,
        std::uint32_t = 1u
    ) noexcept
    {
        if (parent >= NodeCount || locator >= RELATION_COUNT)
        {
            return {};
        }

        const Relation& relation = Relations_(axis)[locator];
        if (!relation.Used || relation.Parent != parent)
        {
            return {};
        }

        return ChildResult_(relation.Next, axis);
    }

    ReadResult FindPreviousChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator,
        std::uint32_t = 1u
    ) noexcept
    {
        if (parent >= NodeCount || locator >= RELATION_COUNT)
        {
            return {};
        }

        const Relation& relation = Relations_(axis)[locator];
        if (!relation.Used || relation.Parent != parent)
        {
            return {};
        }

        return ChildResult_(relation.Previous, axis);
    }

    bool StorePayload(
        std::size_t node,
        std::uint32_t word,
        std::uint64_t value,
        bool atomic
    ) noexcept
    {
        if (node >= NodeCount || word >= PayloadWords)
        {
            return false;
        }

        if (atomic)
        {
            std::atomic_ref<std::uint64_t>(Nodes_[node].Payload[word]).store(
                value,
                std::memory_order_release
            );
        }
        else
        {
            Nodes_[node].Payload[word] = value;
        }
        return true;
    }

    bool LoadPayload(
        std::size_t node,
        std::uint32_t word,
        std::uint64_t& value,
        bool atomic
    ) noexcept
    {
        if (node >= NodeCount || word >= PayloadWords)
        {
            return false;
        }

        value = atomic
            ? std::atomic_ref<std::uint64_t>(Nodes_[node].Payload[word]).load(
                std::memory_order_acquire
            )
            : Nodes_[node].Payload[word];

        return true;
    }

    std::size_t ApproxStorageBytes() const noexcept
    {
        return
            sizeof(Nodes_) +
            sizeof(HRelations_) +
            sizeof(VRelations_);
    }

private:
    std::array<Node, NodeCount> Nodes_{};
    std::array<Relation, RELATION_COUNT> HRelations_{};
    std::array<Relation, RELATION_COUNT> VRelations_{};
    std::mutex GraphMutex_{};

    AxisState& Axis_(std::size_t node, Axis axis) noexcept
    {
        return axis == Axis::HORIZONTAL
            ? Nodes_[node].H
            : Nodes_[node].V;
    }

    const AxisState& Axis_(std::size_t node, Axis axis) const noexcept
    {
        return axis == Axis::HORIZONTAL
            ? Nodes_[node].H
            : Nodes_[node].V;
    }

    std::array<Relation, RELATION_COUNT>& Relations_(Axis axis) noexcept
    {
        return axis == Axis::HORIZONTAL
            ? HRelations_
            : VRelations_;
    }

    const std::array<Relation, RELATION_COUNT>& Relations_(Axis axis) const noexcept
    {
        return axis == Axis::HORIZONTAL
            ? HRelations_
            : VRelations_;
    }

    ReadResult ChildResult_(
        std::uint32_t locator,
        Axis axis
    ) const noexcept
    {
        if (locator == NIL || locator >= RELATION_COUNT)
        {
            return {};
        }

        const Relation& relation = Relations_(axis)[locator];
        return relation.Used
            ? ReadResult{
                relation.Child,
                locator,
                ReadOperation::FOUND,
                true
            }
            : ReadResult{};
    }

    std::uint32_t FindParentRelationLocator_(
        std::size_t child,
        std::size_t parent,
        Axis axis
    ) const noexcept
    {
        if (child >= NodeCount || parent >= NodeCount)
        {
            return NIL;
        }

        for (std::uint8_t ordinal = 0u; ordinal < ParentCapacity; ++ordinal)
        {
            const std::uint32_t locator =
                static_cast<std::uint32_t>(
                    child * static_cast<std::size_t>(ParentCapacity) +
                    ordinal
                );

            const Relation& relation = Relations_(axis)[locator];
            if (
                relation.Used &&
                relation.Parent == parent &&
                relation.Child == child
            )
            {
                return locator;
            }
        }

        return NIL;
    }

    void UnlinkFromParent_(
        std::uint32_t locator,
        Axis axis
    ) noexcept
    {
        Relation& relation = Relations_(axis)[locator];
        AxisState& parent_axis = Axis_(relation.Parent, axis);

        if (relation.Previous == NIL)
        {
            parent_axis.FirstChildRelation = relation.Next;
        }
        else
        {
            Relations_(axis)[relation.Previous].Next = relation.Next;
        }

        if (relation.Next == NIL)
        {
            parent_axis.LastChildRelation = relation.Previous;
        }
        else
        {
            Relations_(axis)[relation.Next].Previous = relation.Previous;
        }

        relation.Previous = NIL;
        relation.Next = NIL;
    }

    bool AddUnlocked_(
        std::size_t parent,
        std::size_t child,
        Axis axis
    ) noexcept
    {
        if (
            parent >= NodeCount ||
            child >= NodeCount ||
            parent >= child ||
            FindParentRelationLocator_(child, parent, axis) != NIL
        )
        {
            return false;
        }

        std::uint8_t ordinal = ParentCapacity;
        for (std::uint8_t i = 0u; i < ParentCapacity; ++i)
        {
            const std::uint32_t locator =
                static_cast<std::uint32_t>(
                    child * static_cast<std::size_t>(ParentCapacity) + i
                );
            if (!Relations_(axis)[locator].Used)
            {
                ordinal = i;
                break;
            }
        }

        if (ordinal == ParentCapacity)
        {
            return false;
        }

        const std::uint32_t locator =
            static_cast<std::uint32_t>(
                child * static_cast<std::size_t>(ParentCapacity) + ordinal
            );

        Relation& relation = Relations_(axis)[locator];
        AxisState& parent_axis = Axis_(parent, axis);

        relation.Used = true;
        relation.Parent = static_cast<std::uint32_t>(parent);
        relation.Child = static_cast<std::uint32_t>(child);
        relation.Previous = parent_axis.LastChildRelation;
        relation.Next = NIL;

        if (parent_axis.LastChildRelation == NIL)
        {
            parent_axis.FirstChildRelation = locator;
        }
        else
        {
            Relations_(axis)[parent_axis.LastChildRelation].Next = locator;
        }
        parent_axis.LastChildRelation = locator;

        Axis_(child, axis).ParentRelations[ordinal] = locator;
        return true;
    }

    bool RemoveUnlocked_(
        std::size_t parent,
        std::size_t child,
        Axis axis
    ) noexcept
    {
        const std::uint32_t locator =
            FindParentRelationLocator_(child, parent, axis);

        if (locator == NIL)
        {
            return false;
        }

        UnlinkFromParent_(locator, axis);

        const std::uint8_t ordinal =
            static_cast<std::uint8_t>(
                locator % static_cast<std::uint32_t>(ParentCapacity)
            );

        Axis_(child, axis).ParentRelations[ordinal] = NIL;
        Relations_(axis)[locator] = Relation{};
        return true;
    }
};

// -----------------------------------------------------------------------------
// Public APC/Fabric adapter for the completed DAG API.
// No runtime pointer registry is used: read results are resolved from the
// returned ephemeral APC facade by slab slot identity.
// -----------------------------------------------------------------------------

class TestAPC final : public AdaptivePackedCellContainer
{
public:
    using RelationOperationForTest = FabricToAPCLinker::RelationOparation;

    std::uint32_t GenerationForTest() noexcept
    {
        APCUseScope use = AcquireAPCUse_();
        return use ? APCCache_.CurrentGeneration_ : 0u;
    }
};

class ResolverTestFabric final : public APCFinilizer
{
public:
    std::size_t SlabBytesForTest() const noexcept
    {
        return FabCache_
            ? static_cast<std::size_t>(FabCache_->SlabCellCount_) *
                sizeof(std::uint64_t)
            : 0u;
    }

    bool ResolveExistingForTest(
        std::uint32_t slot,
        AdaptivePackedCellContainer& apc,
        APCUseScope& use,
        std::optional<std::uint32_t> expected_generation = std::nullopt
    ) noexcept
    {
        return GetExistingAPC_(slot, apc, use, expected_generation);
    }
};

template <std::size_t NodeCount, std::size_t PayloadWords, std::uint8_t ParentCapacity>
class APCFabricBackend
{
public:
    using SD = SchemaDefinition;
    static constexpr std::uint32_t SLOT_WORDS = MINIMUM_APC_CELL_COUNT;
    static constexpr std::uint32_t FABRIC_SLOT_COUNT =
        static_cast<std::uint32_t>(NodeCount);
    static constexpr std::uint8_t PARENT_CAPACITY = ParentCapacity;

    bool Initialize() noexcept
    {
        Slots_.fill(ADS::APC_INDEX_BOUND_SENTINAL);

        constexpr std::uint32_t matrix_width =
            PayloadWords == 0u ? 1u : static_cast<std::uint32_t>(PayloadWords);

        const SD::FabricRegionConfig region_config{
            static_cast<std::uint16_t>(
                ADS::RegionBit(MacroColumnOfAPC::BOTTOM_UP_SLOT) |
                ADS::RegionBit(MacroColumnOfAPC::TOP_DOWN_SLOT)
            ),
            0u,
            matrix_width
        };

        if (!Fabric_.InitializeFabric(
            FABRIC_SLOT_COUNT,
            SLOT_WORDS,
            region_config,
            ParentCapacity
        ))
        {
            return false;
        }

        SD::RegionSchemaTable schemas{};
        SD::MakeDisabledSchemaTable(schemas);



        SD::RegionSchemaRecord& ff_schema_prop = schemas[static_cast<std::size_t>(MacroColumnOfAPC::BOTTOM_UP_SLOT)];
        SD::RegionSchemaRecord& fb_schema = schemas[static_cast<std::size_t>(MacroColumnOfAPC::TOP_DOWN_SLOT)];
        ff_schema_prop.Region = MacroColumnOfAPC::BOTTOM_UP_SLOT;
        ff_schema_prop.Dtype = SD::DataTypeOfMacroColumn::UINT64_T;
        ff_schema_prop.Protocol = SD::SchemaProtocols::PRIVATE_REGION;
        ff_schema_prop.MatrixHeight = 1u;
        ff_schema_prop.MatrixWidth = matrix_width;
        ff_schema_prop.Flags = SD::SchemaFlags::BATCHED_LAST_DIM;

        fb_schema = ff_schema_prop;
        fb_schema.Region = MacroColumnOfAPC::TOP_DOWN_SLOT;
        fb_schema.Protocol = SD::SchemaProtocols::ATOMIC_WORD_ARRAY;

        if (
            !SD::SealDesiredSchema(
                ff_schema_prop,
                0u
            ) ||
            !SD::SealDesiredSchema(
                fb_schema,
                0u
            )
        )
        {
            return false;
        }

        for (std::size_t i = 0u; i < NodeCount; ++i)
        {
            if (!Fabric_.CreateAPC(
                Nodes_[i],
                schemas
            ))
            {
                return false;
            }

            const std::uint32_t slot = Nodes_[i].GetThisSlotIdx();
            if (slot != i)
            {
                return false;
            }
            Slots_[i] = slot;

            if constexpr (PayloadWords > 0u)
            {
                auto direct = Nodes_[i].template BuildAViewOverRegion<std::uint64_t>(
                    MacroColumnOfAPC::BOTTOM_UP_SLOT
                );
                auto atomic = Nodes_[i].template BuildAViewOverRegion<std::uint64_t>(
                    MacroColumnOfAPC::TOP_DOWN_SLOT
                );

                if (
                    !direct.has_value() ||
                    !atomic.has_value() ||
                    direct->Size() < PayloadWords ||
                    atomic->Size() < PayloadWords ||
                    !direct->RawMutableSpan().has_value()
                )
                {
                    return false;
                }

                DirectViews_[i] = std::move(direct.value());
                AtomicViews_[i] = std::move(atomic.value());
            }
        }
        return true;
    }

    bool AddParent(
        std::size_t parent,
        std::size_t child,
        Axis axis,
        std::uint32_t max_tries = DEFAULT_MAX_TRIES) noexcept
    {
        return parent < NodeCount && child < NodeCount &&
            Nodes_[child].AddParent(
                Nodes_[parent],
                EdgeTableForAxis(axis),
                max_tries
            );
    }

    bool RemoveParent(
        std::size_t parent,
        std::size_t child,
        Axis axis,
        std::uint32_t max_tries = DEFAULT_MAX_TRIES) noexcept
    {
        return parent < NodeCount && child < NodeCount &&
            Nodes_[child].RemoveParent(
                Nodes_[parent],
                EdgeTableForAxis(axis),
                max_tries
            );
    }

    bool ReplaceParent(
        std::size_t old_parent,
        std::size_t new_parent,
        std::size_t child,
        Axis axis,
        std::uint32_t max_tries = DEFAULT_MAX_TRIES) noexcept
    {
        return old_parent < NodeCount && new_parent < NodeCount && child < NodeCount &&
            Nodes_[child].ReplaceParent(
                Nodes_[old_parent],
                Nodes_[new_parent],
                EdgeTableForAxis(axis),
                max_tries
            );
    }

    ReadResult FindParent(
        std::size_t child,
        Axis axis,
        std::uint8_t ordinal,
        std::uint32_t max_tries = 1u) noexcept
    {
        if (child >= NodeCount)
        {
            return {};
        }

        TestAPC::RelationOperationForTest operation{};
        AdaptivePackedCellContainer found = Nodes_[child].FindParent(
            EdgeTableForAxis(axis),
            ordinal,
            &operation,
            max_tries
        );
        return Convert_(found, operation);
    }

    ReadResult FindFirstChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t max_tries = 1u) noexcept
    {
        if (parent >= NodeCount)
        {
            return {};
        }

        TestAPC::RelationOperationForTest operation{};
        AdaptivePackedCellContainer found = Nodes_[parent].FindFirstChild(
            EdgeTableForAxis(axis),
            &operation,
            max_tries
        );
        return Convert_(found, operation);
    }

    ReadResult FindLastChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t max_tries = 1u) noexcept
    {
        if (parent >= NodeCount)
        {
            return {};
        }

        TestAPC::RelationOperationForTest operation{};
        AdaptivePackedCellContainer found = Nodes_[parent].FindLastChild(
            EdgeTableForAxis(axis),
            &operation,
            max_tries
        );
        return Convert_(found, operation);
    }

    ReadResult FindNextChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator,
        std::uint32_t max_tries = 1u) noexcept
    {
        if (parent >= NodeCount)
        {
            return {};
        }

        TestAPC::RelationOperationForTest operation{};
        AdaptivePackedCellContainer found = Nodes_[parent].FindNextChild(
            EdgeTableForAxis(axis),
            locator,
            &operation,
            max_tries
        );
        return Convert_(found, operation);
    }

    ReadResult FindPreviousChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator,
        std::uint32_t max_tries = 1u) noexcept
    {
        if (parent >= NodeCount)
        {
            return {};
        }

        TestAPC::RelationOperationForTest operation{};
        AdaptivePackedCellContainer found = Nodes_[parent].FindPreviousChild(
            EdgeTableForAxis(axis),
            locator,
            &operation,
            max_tries
        );
        return Convert_(found, operation);
    }

    BenchmarkReadResult BenchmarkFindParent(
        std::size_t child,
        Axis axis,
        std::uint8_t ordinal,
        std::uint32_t max_tries = 1u) noexcept
    {
        if (child >= NodeCount)
        {
            return {};
        }

        TestAPC::RelationOperationForTest operation{};
        AdaptivePackedCellContainer found = Nodes_[child].FindParent(
            EdgeTableForAxis(axis),
            ordinal,
            &operation,
            max_tries
        );
        (void)found;
        return BenchmarkConvert_(operation, BenchmarkReadResult::NO_NODE);
    }

    BenchmarkReadResult BenchmarkFindFirstChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t max_tries = 1u) noexcept
    {
        if (parent >= NodeCount)
        {
            return {};
        }

        TestAPC::RelationOperationForTest operation{};
        AdaptivePackedCellContainer found = Nodes_[parent].FindFirstChild(
            EdgeTableForAxis(axis),
            &operation,
            max_tries
        );
        (void)found;
        return BenchmarkChildConvert_(operation);
    }

    BenchmarkReadResult BenchmarkFindLastChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t max_tries = 1u) noexcept
    {
        if (parent >= NodeCount)
        {
            return {};
        }

        TestAPC::RelationOperationForTest operation{};
        AdaptivePackedCellContainer found = Nodes_[parent].FindLastChild(
            EdgeTableForAxis(axis),
            &operation,
            max_tries
        );
        (void)found;
        return BenchmarkChildConvert_(operation);
    }

    BenchmarkReadResult BenchmarkFindNextChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator,
        std::uint32_t max_tries = 1u) noexcept
    {
        if (parent >= NodeCount)
        {
            return {};
        }

        TestAPC::RelationOperationForTest operation{};
        AdaptivePackedCellContainer found = Nodes_[parent].FindNextChild(
            EdgeTableForAxis(axis),
            locator,
            &operation,
            max_tries
        );
        (void)found;
        return BenchmarkChildConvert_(operation);
    }

    BenchmarkReadResult BenchmarkFindPreviousChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator,
        std::uint32_t max_tries = 1u) noexcept
    {
        if (parent >= NodeCount)
        {
            return {};
        }

        TestAPC::RelationOperationForTest operation{};
        AdaptivePackedCellContainer found = Nodes_[parent].FindPreviousChild(
            EdgeTableForAxis(axis),
            locator,
            &operation,
            max_tries
        );
        (void)found;
        return BenchmarkChildConvert_(operation);
    }

    bool StorePayload(
        std::size_t node,
        std::uint32_t word,
        std::uint64_t value,
        bool atomic) noexcept
    {
        if constexpr (PayloadWords == 0u)
        {
            (void)node; (void)word; (void)value; (void)atomic;
            return false;
        }
        else
        {
            if (node >= NodeCount || word >= PayloadWords)
            {
                return false;
            }
            if (atomic)
            {
                return AtomicViews_[node].AtomicStore(
                    word,
                    value,
                    std::memory_order_release
                );
            }
            auto span = DirectViews_[node].RawMutableSpan();
            if (!span.has_value())
            {
                return false;
            }
            span.value()[word] = value;
            return true;
        }
    }

    bool LoadPayload(
        std::size_t node,
        std::uint32_t word,
        std::uint64_t& value,
        bool atomic) noexcept
    {
        if constexpr (PayloadWords == 0u)
        {
            (void)node; (void)word; (void)value; (void)atomic;
            return false;
        }
        else
        {
            if (node >= NodeCount || word >= PayloadWords)
            {
                return false;
            }
            if (atomic)
            {
                value = AtomicViews_[node].AtomicLoad(
                    word,
                    std::memory_order_acquire
                );
                return true;
            }
            auto span = DirectViews_[node].RawMutableSpan();
            if (!span.has_value())
            {
                return false;
            }
            value = span.value()[word];
            return true;
        }
    }

    AdaptivePackedCellContainer& Node(std::size_t index) noexcept
    {
        return Nodes_[index];
    }

    std::size_t ApproxStorageBytes() const noexcept
    {
        return Fabric_.SlabBytesForTest();
    }

    ResolverTestFabric Fabric_{};

private:
    std::array<TestAPC, NodeCount> Nodes_{};
    std::array<std::uint32_t, NodeCount> Slots_{};
    std::array<RegionView<std::uint64_t>, NodeCount> DirectViews_{};
    std::array<RegionView<std::uint64_t>, NodeCount> AtomicViews_{};

    std::size_t IndexOfSlot_(std::uint32_t slot) const noexcept
    {
        // Initialize() requires the benchmark's logical node i to occupy Fabric slot i.
        // Therefore correctness conversion can be O(1); no linear scan is necessary.
        return
            slot < NodeCount && Slots_[slot] == slot
                ? static_cast<std::size_t>(slot)
                : ReadResult::NO_NODE;
    }

    static BenchmarkReadResult BenchmarkConvert_(
        const TestAPC::RelationOperationForTest& operation,
        std::size_t node_hint
    ) noexcept
    {
        const bool present =
            operation.MutationOP_ == ReadOperation::FOUND;

        return BenchmarkReadResult{
            present ? node_hint : BenchmarkReadResult::NO_NODE,
            operation.RelationLocator_,
            operation.MutationOP_,
            present
        };
    }

    static BenchmarkReadResult BenchmarkChildConvert_(
        const TestAPC::RelationOperationForTest& operation
    ) noexcept
    {
        const std::size_t node_hint =
            operation.MutationOP_ == ReadOperation::FOUND &&
            operation.RelationLocator_ != UINT32_MAX
                ? static_cast<std::size_t>(
                    EdgeBuilder::RelationSlot(operation.RelationLocator_)
                )
                : BenchmarkReadResult::NO_NODE;

        return BenchmarkConvert_(operation, node_hint);
    }

    ReadResult Convert_(
        AdaptivePackedCellContainer& found,
        const TestAPC::RelationOperationForTest& operation
    ) const noexcept
    {
        const std::uint32_t slot = found.GetThisSlotIdx();
        const std::size_t node = IndexOfSlot_(slot);
        const bool node_present = node != ReadResult::NO_NODE;

        return ReadResult{
            node,
            operation.RelationLocator_,
            operation.MutationOP_,
            node_present
        };
    }
};

// -----------------------------------------------------------------------------
// Benchmark call adapters.
//
// Vector baselines simply project their normal read result. APCFabricBackend has
// dedicated BenchmarkFind* methods that execute the same public APC Find* call but
// do not perform TestKit-only facade->logical-index conversion afterward.
// -----------------------------------------------------------------------------

template <typename Backend>
BenchmarkReadResult BenchmarkFindParentCall(
    Backend& backend,
    std::size_t child,
    Axis axis,
    std::uint8_t ordinal,
    std::uint32_t max_tries = 1u) noexcept
{
    if constexpr (requires {
        backend.BenchmarkFindParent(child, axis, ordinal, max_tries);
    })
    {
        return backend.BenchmarkFindParent(child, axis, ordinal, max_tries);
    }
    else
    {
        const ReadResult read = backend.FindParent(child, axis, ordinal, max_tries);
        return BenchmarkReadResult{
            read.Node,
            read.Locator,
            read.Outcome,
            read.NodePresent
        };
    }
}

template <typename Backend>
BenchmarkReadResult BenchmarkFindFirstChildCall(
    Backend& backend,
    std::size_t parent,
    Axis axis,
    std::uint32_t max_tries = 1u) noexcept
{
    if constexpr (requires {
        backend.BenchmarkFindFirstChild(parent, axis, max_tries);
    })
    {
        return backend.BenchmarkFindFirstChild(parent, axis, max_tries);
    }
    else
    {
        const ReadResult read = backend.FindFirstChild(parent, axis, max_tries);
        return BenchmarkReadResult{read.Node, read.Locator, read.Outcome, read.NodePresent};
    }
}

template <typename Backend>
BenchmarkReadResult BenchmarkFindLastChildCall(
    Backend& backend,
    std::size_t parent,
    Axis axis,
    std::uint32_t max_tries = 1u) noexcept
{
    if constexpr (requires {
        backend.BenchmarkFindLastChild(parent, axis, max_tries);
    })
    {
        return backend.BenchmarkFindLastChild(parent, axis, max_tries);
    }
    else
    {
        const ReadResult read = backend.FindLastChild(parent, axis, max_tries);
        return BenchmarkReadResult{read.Node, read.Locator, read.Outcome, read.NodePresent};
    }
}

template <typename Backend>
BenchmarkReadResult BenchmarkFindNextChildCall(
    Backend& backend,
    std::size_t parent,
    Axis axis,
    std::uint32_t locator,
    std::uint32_t max_tries = 1u) noexcept
{
    if constexpr (requires {
        backend.BenchmarkFindNextChild(parent, axis, locator, max_tries);
    })
    {
        return backend.BenchmarkFindNextChild(parent, axis, locator, max_tries);
    }
    else
    {
        const ReadResult read = backend.FindNextChild(parent, axis, locator, max_tries);
        return BenchmarkReadResult{read.Node, read.Locator, read.Outcome, read.NodePresent};
    }
}

template <typename Backend>
BenchmarkReadResult BenchmarkFindPreviousChildCall(
    Backend& backend,
    std::size_t parent,
    Axis axis,
    std::uint32_t locator,
    std::uint32_t max_tries = 1u) noexcept
{
    if constexpr (requires {
        backend.BenchmarkFindPreviousChild(parent, axis, locator, max_tries);
    })
    {
        return backend.BenchmarkFindPreviousChild(parent, axis, locator, max_tries);
    }
    else
    {
        const ReadResult read = backend.FindPreviousChild(parent, axis, locator, max_tries);
        return BenchmarkReadResult{read.Node, read.Locator, read.Outcome, read.NodePresent};
    }
}

// -----------------------------------------------------------------------------
// Exhaustive quiescent validator. It rebuilds H, V and H-union-V only through
// public reads, verifies both directions, then performs a full topological sort.
// -----------------------------------------------------------------------------

struct GraphProof
{
    bool ReadContracts = true;
    bool ParentOrder = true;
    bool NoDuplicates = true;
    bool ReverseLists = true;
    bool CombinedAcyclic = true;

    bool Passed() const noexcept
    {
        return ReadContracts && ParentOrder && NoDuplicates &&
            ReverseLists && CombinedAcyclic;
    }
};

template <std::size_t NodeCount, std::uint8_t ParentCapacity, typename Backend>
GraphProof ProveQuiescentCombinedDAG(Backend& backend)
{
    GraphProof proof{};
    std::array<std::array<std::array<bool, NodeCount>, NodeCount>, 2u> edges{};

    for (std::size_t axis_index = 0u; axis_index < 2u; ++axis_index)
    {
        const Axis axis = axis_index == 0u ? Axis::HORIZONTAL : Axis::VERTICAL;

        for (std::size_t child = 0u; child < NodeCount; ++child)
        {
            for (std::uint8_t ordinal = 0u; ordinal < ParentCapacity; ++ordinal)
            {
                const ReadResult read = backend.FindParent(child, axis, ordinal, DEFAULT_MAX_TRIES);
                proof.ReadContracts = proof.ReadContracts && read.ContractValid();
                if (read.IsRetry())
                {
                    proof.ReadContracts = false;
                    continue;
                }
                if (!read.IsFound())
                {
                    continue;
                }

                if (read.Node >= child)
                {
                    proof.ParentOrder = false;
                }
                if (read.Node >= NodeCount || edges[axis_index][read.Node][child])
                {
                    proof.NoDuplicates = false;
                    continue;
                }
                edges[axis_index][read.Node][child] = true;
            }
        }

        for (std::size_t parent = 0u; parent < NodeCount; ++parent)
        {
            std::array<bool, NodeCount> enumerated{};
            ReadResult read = backend.FindFirstChild(parent, axis, DEFAULT_MAX_TRIES);
            proof.ReadContracts = proof.ReadContracts && read.ContractValid();

            std::size_t steps = 0u;
            while (read.IsFound())
            {
                if (
                    read.Node >= NodeCount ||
                    enumerated[read.Node] ||
                    !edges[axis_index][parent][read.Node]
                )
                {
                    proof.ReverseLists = false;
                    break;
                }

                enumerated[read.Node] = true;
                if (++steps > NodeCount)
                {
                    proof.ReverseLists = false;
                    break;
                }

                read = backend.FindNextChild(
                    parent,
                    axis,
                    read.Locator,
                    DEFAULT_MAX_TRIES
                );
                proof.ReadContracts = proof.ReadContracts && read.ContractValid();
            }

            if (read.IsRetry())
            {
                proof.ReadContracts = false;
            }

            for (std::size_t child = 0u; child < NodeCount; ++child)
            {
                if (edges[axis_index][parent][child] != enumerated[child])
                {
                    proof.ReverseLists = false;
                }
            }
        }
    }

    std::array<std::uint32_t, NodeCount> indegree{};
    for (std::size_t parent = 0u; parent < NodeCount; ++parent)
    {
        for (std::size_t child = 0u; child < NodeCount; ++child)
        {
            if (edges[0u][parent][child] || edges[1u][parent][child])
            {
                ++indegree[child];
            }
        }
    }

    std::array<std::size_t, NodeCount> queue{};
    std::size_t head = 0u;
    std::size_t tail = 0u;
    for (std::size_t node = 0u; node < NodeCount; ++node)
    {
        if (indegree[node] == 0u)
        {
            queue[tail++] = node;
        }
    }

    std::size_t visited = 0u;
    while (head < tail)
    {
        const std::size_t parent = queue[head++];
        ++visited;
        for (std::size_t child = 0u; child < NodeCount; ++child)
        {
            if (
                (edges[0u][parent][child] || edges[1u][parent][child]) &&
                --indegree[child] == 0u
            )
            {
                queue[tail++] = child;
            }
        }
    }
    proof.CombinedAcyclic = visited == NodeCount;
    return proof;
}

template <typename Backend>
bool RetryReplace(
    Backend& backend,
    std::size_t old_parent,
    std::size_t new_parent,
    std::size_t child,
    Axis axis,
    std::uint64_t& retry_count,
    std::uint32_t attempt_limit = 100'000u) noexcept
{
    for (std::uint32_t attempt = 0u; attempt < attempt_limit; ++attempt)
    {
        if (backend.ReplaceParent(old_parent, new_parent, child, axis, 1u))
        {
            return true;
        }
        ++retry_count;
        PerturbSchedule(attempt);
    }
    return false;
}

// -----------------------------------------------------------------------------
// Test 1: original nine-row baseline, translated to explicit DAG operations.
// -----------------------------------------------------------------------------

namespace Test01_Baseline
{
constexpr std::size_t MAIN_V_PARENT = 0u;
constexpr std::size_t AUX_V_PARENT = 1u;
constexpr std::size_t CHAIN_BEGIN = 2u;
constexpr std::size_t CHAIN_LENGTH = 64u;
constexpr std::size_t CHAIN_END = CHAIN_BEGIN + CHAIN_LENGTH - 1u;
constexpr std::size_t AUX_ANCHOR = CHAIN_END + 1u;
constexpr std::size_t NODE_COUNT = AUX_ANCHOR + 1u;
constexpr std::size_t PAYLOAD_WORDS = 32u;
constexpr std::uint8_t PARENT_CAPACITY = 4u;
constexpr std::uint32_t TRAVERSAL_ROUNDS = 4'000u;
constexpr std::uint32_t PAYLOAD_ROUNDS = 100u;
constexpr std::uint32_t GRAPH_PAYLOAD_ROUNDS = 1'000u;
constexpr std::uint32_t MUTATION_ROUNDS = 512u;
constexpr std::uint32_t MEASURED_RUNS = 9u;
constexpr std::uint32_t CONSTRUCTION_RUNS = 7u;

using LowerBoundBackend =
    VectorLockedForest<NODE_COUNT, PAYLOAD_WORDS>;

using MatchedBackend =
    VectorLockedDAG<
        NODE_COUNT,
        PAYLOAD_WORDS,
        PARENT_CAPACITY
    >;

using APCBackend =
    APCFabricBackend<
        NODE_COUNT,
        PAYLOAD_WORDS,
        PARENT_CAPACITY
    >;

constexpr std::size_t Reverse6(std::size_t value) noexcept
{
    std::size_t result = 0u;
    for (std::size_t bit = 0u; bit < 6u; ++bit)
    {
        result =
            (result << 1u) |
            ((value >> bit) & 1u);
    }
    return result;
}

constexpr std::array<
    std::size_t,
    CHAIN_LENGTH
> MakeVerticalOrder() noexcept
{
    std::array<
        std::size_t,
        CHAIN_LENGTH
    > order{};

    for (
        std::size_t i = 0u;
        i < CHAIN_LENGTH;
        ++i
    )
    {
        order[i] =
            CHAIN_BEGIN +
            Reverse6(i);
    }

    return order;
}

constexpr auto VERTICAL_ORDER =
    MakeVerticalOrder();

template <typename Backend>
bool Build(Backend& backend)
{
    if (!backend.Initialize())
    {
        return false;
    }

    for (
        std::size_t child =
            CHAIN_BEGIN + 1u;
        child <= CHAIN_END;
        ++child
    )
    {
        if (!backend.AddParent(
            child - 1u,
            child,
            Axis::HORIZONTAL
        ))
        {
            return false;
        }
    }

    for (
        std::size_t child :
        VERTICAL_ORDER
    )
    {
        if (!backend.AddParent(
            MAIN_V_PARENT,
            child,
            Axis::VERTICAL
        ))
        {
            return false;
        }
    }

    if (!backend.AddParent(
        AUX_V_PARENT,
        AUX_ANCHOR,
        Axis::VERTICAL
    ))
    {
        return false;
    }

    for (
        std::size_t node = 0u;
        node < NODE_COUNT;
        ++node
    )
    {
        for (
            std::uint32_t word = 0u;
            word < PAYLOAD_WORDS;
            ++word
        )
        {
            const std::uint64_t value =
                (
                    static_cast<std::uint64_t>(
                        node + 1u
                    ) << 32u
                ) |
                word;

            if (
                !backend.StorePayload(
                    node,
                    word,
                    value,
                    false
                ) ||
                !backend.StorePayload(
                    node,
                    word,
                    value,
                    true
                )
            )
            {
                return false;
            }
        }
    }

    return true;
}

struct Timing
{
    bool Ok = false;
    std::uint64_t Checksum = 0u;
    std::uint64_t Operations = 0u;
    std::int64_t ElapsedNs = 0;
    ReadCounts Reads{};

    double NsPerOperation() const noexcept
    {
        return Operations == 0u
            ? 0.0
            : static_cast<double>(
                ElapsedNs
            ) /
                static_cast<double>(
                    Operations
                );
    }
};

template <typename Backend>
Timing HorizontalForward(Backend& backend)
{
    Timing timing{};
    const auto begin = Clock::now();

    for (
        std::uint32_t round = 0u;
        round < TRAVERSAL_ROUNDS;
        ++round
    )
    {
        for (
            std::size_t parent =
                CHAIN_BEGIN;
            parent < CHAIN_END;
            ++parent
        )
        {
            const BenchmarkReadResult read =
                BenchmarkFindFirstChildCall(
                    backend,
                    parent,
                    Axis::HORIZONTAL
                );

            timing.Reads.Observe(read);

            if (
                !read.IsFound() ||
                (read.HasNodeHint() && read.NodeHint != parent + 1u)
            )
            {
                return {};
            }

            timing.Checksum +=
                parent + 2u;
        }
    }

    timing.ElapsedNs =
        std::chrono::duration_cast<
            std::chrono::nanoseconds
        >(
            Clock::now() - begin
        ).count();

    timing.Operations =
        static_cast<std::uint64_t>(
            TRAVERSAL_ROUNDS
        ) *
        (CHAIN_LENGTH - 1u);

    timing.Ok = true;
    return timing;
}

template <typename Backend>
Timing HorizontalBackward(Backend& backend)
{
    Timing timing{};
    const auto begin = Clock::now();

    for (
        std::uint32_t round = 0u;
        round < TRAVERSAL_ROUNDS;
        ++round
    )
    {
        for (
            std::size_t child = CHAIN_END;
            child > CHAIN_BEGIN;
            --child
        )
        {
            const BenchmarkReadResult read =
                BenchmarkFindParentCall(
                    backend,
                    child,
                    Axis::HORIZONTAL,
                    0u
                );

            timing.Reads.Observe(read);

            if (
                !read.IsFound() ||
                (read.HasNodeHint() && read.NodeHint != child - 1u)
            )
            {
                return {};
            }

            timing.Checksum +=
                child;
        }
    }

    timing.ElapsedNs =
        std::chrono::duration_cast<
            std::chrono::nanoseconds
        >(
            Clock::now() - begin
        ).count();

    timing.Operations =
        static_cast<std::uint64_t>(
            TRAVERSAL_ROUNDS
        ) *
        (CHAIN_LENGTH - 1u);

    timing.Ok = true;
    return timing;
}

template <typename Backend>
Timing VerticalForward(
    Backend& backend,
    bool with_payload
)
{
    Timing timing{};

    const std::uint32_t rounds =
        with_payload
            ? GRAPH_PAYLOAD_ROUNDS
            : TRAVERSAL_ROUNDS;

    const auto begin = Clock::now();

    for (
        std::uint32_t round = 0u;
        round < rounds;
        ++round
    )
    {
        BenchmarkReadResult read =
            BenchmarkFindFirstChildCall(
                backend,
                MAIN_V_PARENT,
                Axis::VERTICAL
            );

        timing.Reads.Observe(read);

        for (
            std::size_t i = 0u;
            i < CHAIN_LENGTH;
            ++i
        )
        {
            if (
                !read.IsFound() ||
                !read.HasNodeHint() ||
                read.NodeHint !=
                    VERTICAL_ORDER[i]
            )
            {
                return {};
            }

            if (with_payload)
            {
                std::uint64_t value = 0u;

                if (!backend.LoadPayload(
                    read.NodeHint,
                    static_cast<std::uint32_t>(
                        i % PAYLOAD_WORDS
                    ),
                    value,
                    false
                ))
                {
                    return {};
                }

                timing.Checksum += value;
            }
            else
            {
                timing.Checksum +=
                    VERTICAL_ORDER[i] + 1u;
            }

            const std::uint32_t cursor =
                read.Locator;

            read =
                BenchmarkFindNextChildCall(
                    backend,
                    MAIN_V_PARENT,
                    Axis::VERTICAL,
                    cursor
                );

            timing.Reads.Observe(read);
        }

        if (!read.IsNone())
        {
            return {};
        }
    }

    timing.ElapsedNs =
        std::chrono::duration_cast<
            std::chrono::nanoseconds
        >(
            Clock::now() - begin
        ).count();

    timing.Operations =
        static_cast<std::uint64_t>(
            rounds
        ) *
        CHAIN_LENGTH;

    timing.Ok = true;
    return timing;
}

template <typename Backend>
Timing VerticalBackward(
    Backend& backend
)
{
    Timing timing{};
    const auto begin = Clock::now();

    for (
        std::uint32_t round = 0u;
        round < TRAVERSAL_ROUNDS;
        ++round
    )
    {
        BenchmarkReadResult read =
            BenchmarkFindLastChildCall(
                backend,
                MAIN_V_PARENT,
                Axis::VERTICAL
            );

        timing.Reads.Observe(read);

        for (
            std::size_t i = CHAIN_LENGTH;
            i-- > 0u;
        )
        {
            if (
                !read.IsFound() ||
                !read.HasNodeHint() ||
                read.NodeHint !=
                    VERTICAL_ORDER[i]
            )
            {
                return {};
            }

            timing.Checksum +=
                VERTICAL_ORDER[i] + 1u;

            const std::uint32_t cursor =
                read.Locator;

            read =
                BenchmarkFindPreviousChildCall(
                    backend,
                    MAIN_V_PARENT,
                    Axis::VERTICAL,
                    cursor
                );

            timing.Reads.Observe(read);
        }

        if (!read.IsNone())
        {
            return {};
        }
    }

    timing.ElapsedNs =
        std::chrono::duration_cast<
            std::chrono::nanoseconds
        >(
            Clock::now() - begin
        ).count();

    timing.Operations =
        static_cast<std::uint64_t>(
            TRAVERSAL_ROUNDS
        ) *
        CHAIN_LENGTH;

    timing.Ok = true;
    return timing;
}

template <typename Backend>
Timing PayloadRead(
    Backend& backend,
    bool atomic
)
{
    Timing timing{};
    const auto begin = Clock::now();

    for (
        std::uint32_t round = 0u;
        round < PAYLOAD_ROUNDS;
        ++round
    )
    {
        for (
            std::size_t node = CHAIN_BEGIN;
            node <= CHAIN_END;
            ++node
        )
        {
            for (
                std::uint32_t word = 0u;
                word < PAYLOAD_WORDS;
                ++word
            )
            {
                std::uint64_t value = 0u;

                if (!backend.LoadPayload(
                    node,
                    word,
                    value,
                    atomic
                ))
                {
                    return {};
                }

                timing.Checksum += value;
            }
        }
    }

    timing.ElapsedNs =
        std::chrono::duration_cast<
            std::chrono::nanoseconds
        >(
            Clock::now() - begin
        ).count();

    timing.Operations =
        static_cast<std::uint64_t>(
            PAYLOAD_ROUNDS
        ) *
        CHAIN_LENGTH *
        PAYLOAD_WORDS;

    timing.Ok = true;
    return timing;
}

template <typename Backend>
Timing ParentReplacement(
    Backend& backend,
    Axis axis
)
{
    const std::size_t child =
        CHAIN_END;

    const std::size_t parent_a =
        axis == Axis::HORIZONTAL
            ? CHAIN_END - 1u
            : MAIN_V_PARENT;

    const std::size_t parent_b =
        axis == Axis::HORIZONTAL
            ? CHAIN_END - 2u
            : AUX_V_PARENT;

    Timing timing{};
    std::size_t current = parent_a;

    const auto begin = Clock::now();

    for (
        std::uint32_t i = 0u;
        i < MUTATION_ROUNDS;
        ++i
    )
    {
        const std::size_t next =
            current == parent_a
                ? parent_b
                : parent_a;

        if (!backend.ReplaceParent(
            current,
            next,
            child,
            axis
        ))
        {
            return {};
        }

        current = next;
        ++timing.Operations;
    }

    timing.ElapsedNs =
        std::chrono::duration_cast<
            std::chrono::nanoseconds
        >(
            Clock::now() - begin
        ).count();

    timing.Checksum =
        timing.Operations;

    timing.Ok =
        current == parent_a;

    return timing;
}

enum class Metric : std::uint8_t
{
    H_FORWARD,
    H_BACKWARD,
    V_FORWARD,
    V_BACKWARD,
    PAYLOAD_DIRECT,
    PAYLOAD_ATOMIC,
    GRAPH_PAYLOAD,
    REPLACE_H_PARENT,
    REPLACE_V_PARENT,
    COUNT
};

constexpr std::size_t METRIC_COUNT =
    static_cast<std::size_t>(
        Metric::COUNT
    );

constexpr const char* MetricName(
    Metric metric
) noexcept
{
    switch (metric)
    {
    case Metric::H_FORWARD:
        return "H forward sequential";
    case Metric::H_BACKWARD:
        return "H parent read";
    case Metric::V_FORWARD:
        return "V child cursor scrambled";
    case Metric::V_BACKWARD:
        return "V reverse cursor scrambled";
    case Metric::PAYLOAD_DIRECT:
        return "payload direct read";
    case Metric::PAYLOAD_ATOMIC:
        return "payload atomic read";
    case Metric::GRAPH_PAYLOAD:
        return "graph+payload read";
    case Metric::REPLACE_H_PARENT:
        return "H parent replace";
    case Metric::REPLACE_V_PARENT:
        return "V parent replace";
    default:
        return "unknown";
    }
}

template <typename Backend>
Timing RunMetric(
    Backend& backend,
    Metric metric
)
{
    switch (metric)
    {
    case Metric::H_FORWARD:
        return HorizontalForward(backend);
    case Metric::H_BACKWARD:
        return HorizontalBackward(backend);
    case Metric::V_FORWARD:
        return VerticalForward(
            backend,
            false
        );
    case Metric::V_BACKWARD:
        return VerticalBackward(backend);
    case Metric::PAYLOAD_DIRECT:
        return PayloadRead(
            backend,
            false
        );
    case Metric::PAYLOAD_ATOMIC:
        return PayloadRead(
            backend,
            true
        );
    case Metric::GRAPH_PAYLOAD:
        return VerticalForward(
            backend,
            true
        );
    case Metric::REPLACE_H_PARENT:
        return ParentReplacement(
            backend,
            Axis::HORIZONTAL
        );
    case Metric::REPLACE_V_PARENT:
        return ParentReplacement(
            backend,
            Axis::VERTICAL
        );
    default:
        return {};
    }
}

template <typename Backend>
std::optional<double>
ConstructionMedianUs()
{
    std::array<
        double,
        CONSTRUCTION_RUNS
    > samples{};

    for (
        std::uint32_t run = 0u;
        run < CONSTRUCTION_RUNS;
        ++run
    )
    {
        const auto begin =
            Clock::now();

        Backend backend{};

        if (!Build(backend))
        {
            return std::nullopt;
        }

        const auto elapsed =
            std::chrono::duration_cast<
                std::chrono::nanoseconds
            >(
                Clock::now() - begin
            ).count();

        samples[run] =
            static_cast<double>(
                elapsed
            ) /
            1000.0;
    }

    return Median(samples);
}

inline Result Run()
{
    Banner(
        "TEST 1 - QUIESCENT CORE COST / STORAGE FOOTPRINT / BASELINE FAIRNESS"
    );

    std::cout
        << "Workload: identical 67-node H/V graph, 32 payload words/node, "
        << "K=4 structural capacity.\n"
        << "Baseline A is a one-parent-per-axis lower bound.\n"
        << "Baseline B supports the same K-parent H/V DAG rule under one global mutation mutex.\n"
        << "APC/Fabric additionally provides generations, schema protocols, relocation metadata,\n"
        << "and sequence-validated public read/mutation contracts.\n"
        << "Timed APC traversal executes the public Find* operation but excludes the TestKit-only\n"
        << "returned-facade -> logical-index conversion. Full facade identity is still checked\n"
        << "by the exhaustive correctness proof before and after timing.\n\n";

    const std::optional<double>
        lower_build_us =
            ConstructionMedianUs<
                LowerBoundBackend
            >();

    const std::optional<double>
        matched_build_us =
            ConstructionMedianUs<
                MatchedBackend
            >();

    const std::optional<double>
        apc_build_us =
            ConstructionMedianUs<
                APCBackend
            >();

    if (
        !lower_build_us.has_value() ||
        !matched_build_us.has_value() ||
        !apc_build_us.has_value()
    )
    {
        return Result::FAIL;
    }

    LowerBoundBackend lower{};
    MatchedBackend matched{};
    APCBackend apc{};

    if (
        !Build(lower) ||
        !Build(matched) ||
        !Build(apc)
    )
    {
        return Result::FAIL;
    }

    const GraphProof lower_proof =
        ProveQuiescentCombinedDAG<
            NODE_COUNT,
            PARENT_CAPACITY
        >(lower);

    const GraphProof matched_proof =
        ProveQuiescentCombinedDAG<
            NODE_COUNT,
            PARENT_CAPACITY
        >(matched);

    const GraphProof apc_initial_proof =
        ProveQuiescentCombinedDAG<
            NODE_COUNT,
            PARENT_CAPACITY
        >(apc);

    if (
        !lower_proof.Passed() ||
        !matched_proof.Passed() ||
        !apc_initial_proof.Passed()
    )
    {
        return Result::FAIL;
    }

    // Untimed warm-up.
    for (
        std::size_t index = 0u;
        index < METRIC_COUNT;
        ++index
    )
    {
        const Metric metric =
            static_cast<Metric>(index);

        const Timing a =
            RunMetric(lower, metric);

        const Timing b =
            RunMetric(matched, metric);

        const Timing c =
            RunMetric(apc, metric);

        if (
            !a.Ok ||
            !b.Ok ||
            !c.Ok ||
            a.Checksum != b.Checksum ||
            b.Checksum != c.Checksum
        )
        {
            return Result::FAIL;
        }
    }

    std::array<
        std::array<
            double,
            MEASURED_RUNS
        >,
        METRIC_COUNT
    > lower_samples{};

    std::array<
        std::array<
            double,
            MEASURED_RUNS
        >,
        METRIC_COUNT
    > matched_samples{};

    std::array<
        std::array<
            double,
            MEASURED_RUNS
        >,
        METRIC_COUNT
    > apc_samples{};

    ReadCounts lower_reads{};
    ReadCounts matched_reads{};
    ReadCounts apc_reads{};

    for (
        std::uint32_t run = 0u;
        run < MEASURED_RUNS;
        ++run
    )
    {
        for (
            std::size_t index = 0u;
            index < METRIC_COUNT;
            ++index
        )
        {
            const Metric metric =
                static_cast<Metric>(index);

            Timing lower_timing{};
            Timing matched_timing{};
            Timing apc_timing{};

            // Rotate measurement order to reduce systematic first/last bias.
            switch (run % 3u)
            {
            case 0u:
                lower_timing =
                    RunMetric(lower, metric);
                matched_timing =
                    RunMetric(matched, metric);
                apc_timing =
                    RunMetric(apc, metric);
                break;

            case 1u:
                matched_timing =
                    RunMetric(matched, metric);
                apc_timing =
                    RunMetric(apc, metric);
                lower_timing =
                    RunMetric(lower, metric);
                break;

            default:
                apc_timing =
                    RunMetric(apc, metric);
                lower_timing =
                    RunMetric(lower, metric);
                matched_timing =
                    RunMetric(matched, metric);
                break;
            }

            if (
                !lower_timing.Ok ||
                !matched_timing.Ok ||
                !apc_timing.Ok ||
                lower_timing.Checksum !=
                    matched_timing.Checksum ||
                matched_timing.Checksum !=
                    apc_timing.Checksum
            )
            {
                std::cout
                    << "  failed metric: "
                    << MetricName(metric)
                    << '\n';

                return Result::FAIL;
            }

            lower_samples[index][run] =
                lower_timing.NsPerOperation();

            matched_samples[index][run] =
                matched_timing.NsPerOperation();

            apc_samples[index][run] =
                apc_timing.NsPerOperation();

            lower_reads.Add(
                lower_timing.Reads
            );

            matched_reads.Add(
                matched_timing.Reads
            );

            apc_reads.Add(
                apc_timing.Reads
            );
        }
    }

    const GraphProof apc_final_proof =
        ProveQuiescentCombinedDAG<
            NODE_COUNT,
            PARENT_CAPACITY
        >(apc);

    const GraphProof matched_final_proof =
        ProveQuiescentCombinedDAG<
            NODE_COUNT,
            PARENT_CAPACITY
        >(matched);

    std::cout
        << "Correctness before/after timing: "
        << (
            apc_final_proof.Passed() &&
            matched_final_proof.Passed()
                ? "PASS"
                : "FAIL"
        )
        << "\n\n"
        << "MEDIAN CONSTRUCTION ("
        << CONSTRUCTION_RUNS
        << " fresh builds)\n"
        << "  vector forest lower bound : "
        << std::fixed
        << std::setprecision(2)
        << lower_build_us.value()
        << " us\n"
        << "  vector K-parent DAG/mutex  : "
        << matched_build_us.value()
        << " us\n"
        << "  APC/Fabric                 : "
        << apc_build_us.value()
        << " us\n\n"
        << "ALLOCATED STORAGE FOR TEST BACKING\n"
        << "  vector forest lower bound : "
        << lower.ApproxStorageBytes()
        << " bytes\n"
        << "  vector K-parent DAG/mutex  : "
        << matched.ApproxStorageBytes()
        << " bytes\n"
        << "  APC/Fabric slab            : "
        << apc.ApproxStorageBytes()
        << " bytes\n"
        << "  note: these are representation/backing bytes, not a heap-profiler total.\n\n"
        << "MEDIAN COST PER OPERATION ("
        << MEASURED_RUNS
        << " measured runs after warm-up)\n";

    std::cout
        << std::left
        << std::setw(28) << "metric"
        << std::right
        << std::setw(13) << "forest-LB"
        << std::setw(13) << "vector-DAG"
        << std::setw(13) << "APC"
        << std::setw(13) << "APC/DAG"
        << '\n';

    for (
        std::size_t index = 0u;
        index < METRIC_COUNT;
        ++index
    )
    {
        const double lower_ns =
            Median(
                lower_samples[index]
            );

        const double matched_ns =
            Median(
                matched_samples[index]
            );

        const double apc_ns =
            Median(
                apc_samples[index]
            );

        std::cout
            << std::left
            << std::setw(28)
            << MetricName(
                static_cast<Metric>(
                    index
                )
            )
            << std::right
            << std::setw(10)
            << std::fixed
            << std::setprecision(2)
            << lower_ns
            << " ns"
            << std::setw(10)
            << matched_ns
            << " ns"
            << std::setw(10)
            << apc_ns
            << " ns"
            << std::setw(10)
            << Ratio(
                apc_ns,
                matched_ns
            )
            << "x\n";
    }

    std::cout
        << "\nPUBLIC READ OUTCOMES\n";

    PrintReadCounts(
        "vector forest lower bound",
        lower_reads
    );

    PrintReadCounts(
        "vector K-parent DAG",
        matched_reads
    );

    PrintReadCounts(
        "APC/Fabric DAG",
        apc_reads
    );

    const bool ok =
        lower_proof.Passed() &&
        matched_final_proof.Passed() &&
        apc_final_proof.Passed() &&
        lower_reads.BadContract == 0u &&
        matched_reads.BadContract == 0u &&
        apc_reads.BadContract == 0u &&
        apc_reads.Retry == 0u;

    std::cout
        << "\nTEST 1 OVERALL: "
        << (ok ? "PASS" : "FAIL")
        << '\n';

    return ok
        ? Result::PASS
        : Result::FAIL;
}
} // namespace Test01_Baseline

// -----------------------------------------------------------------------------
// Test 2: shared-parent contention sweep against the global-mutex forest.
// -----------------------------------------------------------------------------

namespace Test02_Contention
{
constexpr std::size_t NODE_COUNT = 40u;
constexpr std::uint8_t K = 4u;
constexpr std::size_t FIRST_CHILD = 8u;
constexpr std::uint32_t OPS_PER_THREAD = 2'000u;
constexpr std::uint32_t MEASURED_RUNS = 5u;

using MatchedBackend =
    VectorLockedDAG<
        NODE_COUNT,
        1u,
        K
    >;

using APCBackend =
    APCFabricBackend<
        NODE_COUNT,
        1u,
        K
    >;

template <typename Backend>
bool Build(
    Backend& backend,
    std::size_t workers
)
{
    if (!backend.Initialize())
    {
        return false;
    }

    for (
        std::size_t i = 0u;
        i < workers;
        ++i
    )
    {
        const std::size_t child =
            FIRST_CHILD + i;

        if (
            !backend.AddParent(
                0u,
                child,
                Axis::HORIZONTAL
            ) ||
            !backend.AddParent(
                2u,
                child,
                Axis::VERTICAL
            )
        )
        {
            return false;
        }
    }

    return true;
}

struct SweepResult
{
    bool Ok = false;
    double NsPerSuccess = 0.0;
    std::uint64_t Success = 0u;
    std::uint64_t Retries = 0u;
};

template <typename Backend>
SweepResult RunWorkers(
    Backend& backend,
    std::size_t workers
)
{
    std::barrier start(
        static_cast<std::ptrdiff_t>(
            workers + 1u
        )
    );

    std::atomic<bool> failed{false};

    std::atomic<std::uint64_t>
        success{0u};

    std::atomic<std::uint64_t>
        retries{0u};

    std::vector<std::thread>
        threads;

    threads.reserve(workers);

    for (
        std::size_t worker = 0u;
        worker < workers;
        ++worker
    )
    {
        threads.emplace_back(
            [&, worker]() noexcept
            {
                const std::size_t child =
                    FIRST_CHILD +
                    worker;

                std::size_t h_current = 0u;
                std::size_t v_current = 2u;

                std::uint64_t local_retries = 0u;
                std::uint64_t local_success = 0u;

                start.arrive_and_wait();

                for (
                    std::uint32_t i = 0u;
                    i < OPS_PER_THREAD;
                    ++i
                )
                {
                    const std::size_t h_next =
                        h_current == 0u
                            ? 1u
                            : 0u;

                    const std::size_t v_next =
                        v_current == 2u
                            ? 3u
                            : 2u;

                    if (
                        !RetryReplace(
                            backend,
                            h_current,
                            h_next,
                            child,
                            Axis::HORIZONTAL,
                            local_retries
                        ) ||
                        !RetryReplace(
                            backend,
                            v_current,
                            v_next,
                            child,
                            Axis::VERTICAL,
                            local_retries
                        )
                    )
                    {
                        failed.store(
                            true,
                            std::memory_order_release
                        );

                        break;
                    }

                    h_current = h_next;
                    v_current = v_next;
                    local_success += 2u;
                }

                success.fetch_add(
                    local_success,
                    std::memory_order_relaxed
                );

                retries.fetch_add(
                    local_retries,
                    std::memory_order_relaxed
                );
            }
        );
    }

    const auto begin =
        Clock::now();

    start.arrive_and_wait();

    for (
        std::thread& thread :
        threads
    )
    {
        thread.join();
    }

    const auto elapsed =
        std::chrono::duration_cast<
            std::chrono::nanoseconds
        >(
            Clock::now() - begin
        ).count();

    const std::uint64_t completed =
        success.load(
            std::memory_order_acquire
        );

    return {
        !failed.load(
            std::memory_order_acquire
        ) &&
            completed ==
                workers *
                OPS_PER_THREAD *
                2u,

        completed == 0u
            ? 0.0
            : static_cast<double>(
                elapsed
            ) /
                completed,

        completed,

        retries.load(
            std::memory_order_acquire
        )
    };
}

inline Result Run()
{
    Banner(
        "TEST 2 - SAME-K MUTABLE-DAG CONTENTION: GLOBAL MUTEX vs APC/FABRIC"
    );

    constexpr std::array<
        std::size_t,
        4u
    > WORKERS{
        1u,
        2u,
        4u,
        8u
    };

    bool all_ok = true;

    std::cout
        << "Both backends use K=4 and parent<child. Each worker owns one child;\n"
        << "all workers replace parents drawn from the same two H and two V parents.\n"
        << "The vector DAG serializes each ReplaceParent with one global mutex.\n"
        << "APC/Fabric performs bounded one-attempt transactions and retries at workload level.\n"
        << "Timing excludes thread creation because workers wait on a start barrier.\n\n";

    for (
        std::size_t workers :
        WORKERS
    )
    {
        std::array<
            double,
            MEASURED_RUNS
        > vector_ns{};

        std::array<
            double,
            MEASURED_RUNS
        > apc_ns{};

        std::array<
            double,
            MEASURED_RUNS
        > apc_retry_per_success{};

        bool row_ok = true;

        for (
            std::uint32_t run = 0u;
            run < MEASURED_RUNS;
            ++run
        )
        {
            MatchedBackend vector_backend{};
            APCBackend apc_backend{};

            if (
                !Build(
                    vector_backend,
                    workers
                ) ||
                !Build(
                    apc_backend,
                    workers
                )
            )
            {
                return Result::FAIL;
            }

            SweepResult vector_result{};
            SweepResult apc_result{};

            if ((run & 1u) == 0u)
            {
                vector_result =
                    RunWorkers(
                        vector_backend,
                        workers
                    );

                apc_result =
                    RunWorkers(
                        apc_backend,
                        workers
                    );
            }
            else
            {
                apc_result =
                    RunWorkers(
                        apc_backend,
                        workers
                    );

                vector_result =
                    RunWorkers(
                        vector_backend,
                        workers
                    );
            }

            const GraphProof vector_proof =
                ProveQuiescentCombinedDAG<
                    NODE_COUNT,
                    K
                >(vector_backend);

            const GraphProof apc_proof =
                ProveQuiescentCombinedDAG<
                    NODE_COUNT,
                    K
                >(apc_backend);

            row_ok =
                row_ok &&
                vector_result.Ok &&
                apc_result.Ok &&
                vector_proof.Passed() &&
                apc_proof.Passed();

            vector_ns[run] =
                vector_result.NsPerSuccess;

            apc_ns[run] =
                apc_result.NsPerSuccess;

            apc_retry_per_success[run] =
                apc_result.Success == 0u
                    ? 0.0
                    : static_cast<double>(
                        apc_result.Retries
                    ) /
                        static_cast<double>(
                            apc_result.Success
                        );
        }

        const double vector_median =
            Median(vector_ns);

        const double apc_median =
            Median(apc_ns);

        const double retry_rate =
            Median(
                apc_retry_per_success
            );

        const double vector_mops =
            vector_median > 0.0
                ? 1000.0 /
                    vector_median
                : 0.0;

        const double apc_mops =
            apc_median > 0.0
                ? 1000.0 /
                    apc_median
                : 0.0;

        all_ok =
            all_ok &&
            row_ok;

        std::cout
            << "  threads="
            << std::setw(2)
            << workers
            << "  vector-DAG="
            << std::setw(9)
            << std::fixed
            << std::setprecision(2)
            << vector_median
            << " ns/op ("
            << std::setw(6)
            << vector_mops
            << " Mops/s)"
            << "  APC="
            << std::setw(9)
            << apc_median
            << " ns/op ("
            << std::setw(6)
            << apc_mops
            << " Mops/s)"
            << "  APC/vector="
            << std::setw(6)
            << Ratio(
                apc_median,
                vector_median
            )
            << "x"
            << "  retries/success="
            << std::setw(8)
            << std::setprecision(4)
            << retry_rate
            << "  integrity="
            << (
                row_ok
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';
    }

    std::cout
        << "\nTEST 2 OVERALL: "
        << (
            all_ok
                ? "PASS"
                : "FAIL"
        )
        << '\n';

    return all_ok
        ? Result::PASS
        : Result::FAIL;
}
} // namespace Test02_Contention

// -----------------------------------------------------------------------------
// Test 3: public parent reader versus atomic ReplaceParent writer.
// -----------------------------------------------------------------------------

namespace Test03_ReaderWriter
{
inline Result Run()
{
    Banner("TEST 3 - PUBLIC PARENT READERS vs ATOMIC CROSS-PARENT WRITER");

    constexpr std::size_t N = 4u;
    constexpr std::uint8_t K = 2u;
    constexpr std::size_t CHILD = 3u;
    constexpr std::uint32_t WRITES = 50'000u;
    constexpr std::uint32_t READS = 80'000u;
    constexpr std::size_t READER_COUNT = 4u;

    APCFabricBackend<N, 1u, K> backend{};
    if (!backend.Initialize() || !backend.AddParent(0u, CHILD, Axis::HORIZONTAL))
    {
        return Result::FAIL;
    }

    std::barrier start(static_cast<std::ptrdiff_t>(READER_COUNT + 2u));
    std::atomic<bool> failed{false};
    std::atomic<std::uint64_t> found_a{0u};
    std::atomic<std::uint64_t> found_b{0u};
    std::atomic<std::uint64_t> retry{0u};
    std::atomic<std::uint64_t> none{0u};
    std::atomic<std::uint64_t> writer_retries{0u};

    std::thread writer([&]() noexcept
    {
        std::size_t current = 0u;
        std::uint64_t local_retries = 0u;
        start.arrive_and_wait();
        for (std::uint32_t i = 0u; i < WRITES; ++i)
        {
            const std::size_t next = current == 0u ? 1u : 0u;
            if (!RetryReplace(
                backend, current, next, CHILD,
                Axis::HORIZONTAL, local_retries
            ))
            {
                failed.store(true, std::memory_order_release);
                break;
            }
            current = next;
        }
        writer_retries.store(local_retries, std::memory_order_release);
    });

    std::vector<std::thread> readers;
    readers.reserve(READER_COUNT);
    for (std::size_t reader_index = 0u; reader_index < READER_COUNT; ++reader_index)
    {
        readers.emplace_back([&, reader_index]() noexcept
        {
            std::uint64_t local_a = 0u;
            std::uint64_t local_b = 0u;
            std::uint64_t local_retry = 0u;
            std::uint64_t local_none = 0u;
            start.arrive_and_wait();
            for (std::uint32_t i = 0u; i < READS; ++i)
            {
                const ReadResult read = backend.FindParent(
                    CHILD,
                    Axis::HORIZONTAL,
                    0u,
                    1u
                );
                if (!read.ContractValid())
                {
                    failed.store(true, std::memory_order_release);
                    break;
                }
                if (read.IsRetry()) ++local_retry;
                else if (read.IsNone()) ++local_none;
                else if (read.Node == 0u) ++local_a;
                else if (read.Node == 1u) ++local_b;
                else
                {
                    failed.store(true, std::memory_order_release);
                    break;
                }
                PerturbSchedule(i + reader_index);
            }
            found_a.fetch_add(local_a, std::memory_order_relaxed);
            found_b.fetch_add(local_b, std::memory_order_relaxed);
            retry.fetch_add(local_retry, std::memory_order_relaxed);
            none.fetch_add(local_none, std::memory_order_relaxed);
        });
    }

    start.arrive_and_wait();
    writer.join();
    for (std::thread& reader : readers) reader.join();

    const GraphProof proof = ProveQuiescentCombinedDAG<N, K>(backend);
    const bool ok = !failed.load(std::memory_order_acquire) &&
        none.load(std::memory_order_acquire) == 0u &&
        proof.Passed();

    std::cout
        << "  parent A observations : " << found_a.load() << '\n'
        << "  parent B observations : " << found_b.load() << '\n'
        << "  reader RETRY          : " << retry.load() << '\n'
        << "  reader NONE (illegal) : " << none.load() << '\n'
        << "  writer retries        : " << writer_retries.load() << '\n'
        << "\nTEST 3 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';

    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test03_ReaderWriter

// -----------------------------------------------------------------------------
// Test 4: API symmetry, multi-parent isolation, duplicate/full-row rejection.
// -----------------------------------------------------------------------------

namespace Test04_PublicMutationAPI
{
inline bool ParentSetEquals(
    APCFabricBackend<8u, 1u, 2u>& backend,
    std::size_t child,
    Axis axis,
    std::array<std::size_t, 2u> expected,
    std::size_t expected_count)
{
    std::array<bool, 8u> found{};
    std::size_t count = 0u;
    for (std::uint8_t ordinal = 0u; ordinal < 2u; ++ordinal)
    {
        const ReadResult read = backend.FindParent(child, axis, ordinal, DEFAULT_MAX_TRIES);
        if (!read.ContractValid() || read.IsRetry()) return false;
        if (read.IsFound())
        {
            if (read.Node >= found.size() || found[read.Node]) return false;
            found[read.Node] = true;
            ++count;
        }
    }
    if (count != expected_count) return false;
    for (std::size_t i = 0u; i < expected_count; ++i)
    {
        if (!found[expected[i]]) return false;
    }
    return true;
}

inline Result Run()
{
    Banner("TEST 4 - PUBLIC DAG MUTATION API PAIRS AND MULTI-PARENT ISOLATION");
    APCFabricBackend<8u, 1u, 2u> backend{};
    if (!backend.Initialize()) return Result::FAIL;

    bool ok = true;
    ok = backend.AddParent(0u, 5u, Axis::HORIZONTAL) && ok;
    ok = backend.Node(1u).AttachMyChild(
        backend.Node(5u),
        FabricSegments::VALUE_PARENT_EDGE_TABLE_H
    ) && ok;
    ok = ParentSetEquals(backend, 5u, Axis::HORIZONTAL, {0u, 1u}, 2u) && ok;

    const bool duplicate_rejected = !backend.AddParent(0u, 5u, Axis::HORIZONTAL);
    const bool third_rejected = !backend.AddParent(2u, 5u, Axis::HORIZONTAL);
    ok = duplicate_rejected && third_rejected && ok;

    ok = backend.RemoveParent(0u, 5u, Axis::HORIZONTAL) && ok;
    ok = ParentSetEquals(backend, 5u, Axis::HORIZONTAL, {1u, 0u}, 1u) && ok;
    ok = backend.AddParent(2u, 5u, Axis::HORIZONTAL) && ok;
    ok = backend.Node(1u).DetachMyChild(
        backend.Node(5u),
        FabricSegments::VALUE_PARENT_EDGE_TABLE_H
    ) && ok;
    ok = ParentSetEquals(backend, 5u, Axis::HORIZONTAL, {2u, 0u}, 1u) && ok;

    ok = backend.AddParent(0u, 5u, Axis::VERTICAL) && ok;
    ok = backend.AddParent(1u, 5u, Axis::VERTICAL) && ok;
    ok = backend.ReplaceParent(0u, 2u, 5u, Axis::VERTICAL) && ok;
    ok = ParentSetEquals(backend, 5u, Axis::VERTICAL, {1u, 2u}, 2u) && ok;

    const bool same_parent_replace_rejected =
        !backend.Node(5u).ReplaceParent(
            backend.Node(1u),
            backend.Node(1u),
            FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V
        );
    const bool invalid_table_rejected =
        !backend.Node(5u).AddParent(
            backend.Node(0u),
            FabricSegments::SEGMENT_POOL
        );

    const GraphProof proof = ProveQuiescentCombinedDAG<8u, 2u>(backend);
    ok = ok && same_parent_replace_rejected && invalid_table_rejected && proof.Passed();

    std::cout
        << "  child-side and parent-side API symmetry : " << (ok ? "PASS" : "FAIL") << '\n'
        << "  duplicate relation rejected             : " << (duplicate_rejected ? "PASS" : "FAIL") << '\n'
        << "  third parent at K=2 rejected            : " << (third_rejected ? "PASS" : "FAIL") << '\n'
        << "  H/V rows remain independent             : " << (proof.Passed() ? "PASS" : "FAIL") << '\n'
        << "\nTEST 4 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';

    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test04_PublicMutationAPI

// -----------------------------------------------------------------------------
// Test 5: direct proof of the fixed-order H-union-V DAG rule.
// -----------------------------------------------------------------------------

namespace Test05_CombinedAcyclicity
{
inline Result Run()
{
    Banner("TEST 5 - H-UNION-V ACYCLIC DAG, MULTI-PARENT CAPACITY, CYCLE REJECTION");
    constexpr std::size_t N = 7u;
    constexpr std::uint8_t K = 2u;
    APCFabricBackend<N, 1u, K> backend{};
    if (!backend.Initialize()) return Result::FAIL;

    bool valid_diamond =
        backend.AddParent(0u, 2u, Axis::HORIZONTAL) &&
        backend.AddParent(1u, 2u, Axis::VERTICAL) &&
        backend.AddParent(0u, 3u, Axis::VERTICAL) &&
        backend.AddParent(1u, 3u, Axis::HORIZONTAL) &&
        backend.AddParent(2u, 4u, Axis::HORIZONTAL) &&
        backend.AddParent(3u, 4u, Axis::VERTICAL);

    const bool backward_h_rejected = !backend.AddParent(4u, 0u, Axis::HORIZONTAL);
    const bool backward_v_rejected = !backend.AddParent(4u, 1u, Axis::VERTICAL);
    const bool self_h_rejected = !backend.AddParent(4u, 4u, Axis::HORIZONTAL);
    const bool self_v_rejected = !backend.AddParent(4u, 4u, Axis::VERTICAL);

    // If accepted, this would close 0 --H--> 2 --V--> 0.
    const bool cross_axis_cycle_rejected =
        !backend.AddParent(2u, 0u, Axis::VERTICAL);

    const bool h_capacity =
        backend.AddParent(0u, 6u, Axis::HORIZONTAL) &&
        backend.AddParent(1u, 6u, Axis::HORIZONTAL) &&
        !backend.AddParent(2u, 6u, Axis::HORIZONTAL);
    const bool v_capacity =
        backend.AddParent(0u, 6u, Axis::VERTICAL) &&
        backend.AddParent(1u, 6u, Axis::VERTICAL) &&
        !backend.AddParent(2u, 6u, Axis::VERTICAL);

    const bool bad_replace_rejected =
        !backend.ReplaceParent(0u, 6u, 2u, Axis::HORIZONTAL);
    const ReadResult preserved = backend.FindParent(2u, Axis::HORIZONTAL, 0u);
    const bool old_relation_preserved = preserved.IsFound() && preserved.Node == 0u;

    const GraphProof proof = ProveQuiescentCombinedDAG<N, K>(backend);
    const bool ok = valid_diamond &&
        backward_h_rejected && backward_v_rejected &&
        self_h_rejected && self_v_rejected &&
        cross_axis_cycle_rejected && h_capacity && v_capacity &&
        bad_replace_rejected && old_relation_preserved && proof.Passed();

    std::cout
        << "  legal mixed-axis diamond              : " << (valid_diamond ? "PASS" : "FAIL") << '\n'
        << "  every backward/self insertion rejected: "
        << ((backward_h_rejected && backward_v_rejected && self_h_rejected && self_v_rejected)
            ? "PASS" : "FAIL") << '\n'
        << "  attempted H-union-V cycle rejected    : "
        << (cross_axis_cycle_rejected ? "PASS" : "FAIL") << '\n'
        << "  K=2 independently enforced on H and V : "
        << ((h_capacity && v_capacity) ? "PASS" : "FAIL") << '\n'
        << "  failed replacement preserves old edge : "
        << ((bad_replace_rejected && old_relation_preserved) ? "PASS" : "FAIL") << '\n'
        << "  exhaustive public-read topological sort: "
        << (proof.Passed() ? "PASS" : "FAIL") << '\n'
        << "\nTEST 5 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';

    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test05_CombinedAcyclicity

// -----------------------------------------------------------------------------
// Test 6: fixed region schema, compact DEVICE_VIEW_TABLE rows, protocol storage,
// typed zero-copy views, and slot-reuse cleanup.
// -----------------------------------------------------------------------------

namespace Test06_RegionSchemaAndViews
{
using SD = SchemaDefinition;

constexpr std::uint32_t VIEW_WIDTH = 8u;

constexpr SD::FabricRegionConfig OneRegionConfig(
    std::uint32_t batch_capacity = VIEW_WIDTH
) noexcept
{
    return SD::FabricRegionConfig{
        ADS::RegionBit(MacroColumnOfAPC::BOTTOM_UP_SLOT),
        0u,
        batch_capacity
    };
}

class InspectableFabric final : public APCFinilizer
{
public:
    std::span<const SD::RegionSchemaRecord> SchemaRow(
        std::uint32_t slot
    ) noexcept
    {
        const std::span<SD::RegionSchemaRecord> row = MetrixViewRow_(slot);
        return std::span<const SD::RegionSchemaRecord>(row.data(), row.size());
    }

    std::optional<std::uint64_t> ReadAPCLocalCell(
        std::uint32_t slot,
        std::uint32_t local_cell
    ) noexcept
    {
        const ADS::RangeOfAPC range = GetSegmentPoolRange(slot);
        if (
            !range.IsValid ||
            local_cell >= range.EndIndex - range.BeginIndex ||
            range.BeginIndex + local_cell >= FabCache_->SlabCellCount_
        )
        {
            return std::nullopt;
        }

        return std::atomic_ref<const std::uint64_t>(
            SlabBasePtr_[range.BeginIndex + local_cell]
        ).load(std::memory_order_acquire);
    }

    const CoreOfFabricCoordinator::FabricCache*
    FabricHeader() const noexcept
    {
        return FabCache_;
    }

    std::optional<ADS::RangeOfAPC> TableRange(FabricSegments table) noexcept
    {
        RecordBookConf::FabricSegmentBounds bounds{};
        if (!GetRecordMapCarrierRanges_(table, bounds) || !bounds.IsValid)
        {
            return std::nullopt;
        }
        return ADS::RangeOfAPC{
            static_cast<std::size_t>(bounds.BeginIndex),
            static_cast<std::size_t>(bounds.EndIndex),
            true
        };
    }

    std::size_t MatrixViewBegin() const noexcept
    {
        return FabCache_->MatrixViewTableBeginIndex_;
    }

    std::uint16_t ActiveMask() const noexcept { return FabCache_->ActiveRegionMask_; }
    std::uint8_t ActiveCount() const noexcept { return FabCache_->ActiveRegionCount_; }
    std::uint16_t ViewRowCells() const noexcept { return FabCache_->MatrixViewRowCellCount_; }
    std::uint32_t BatchCapacity() const noexcept { return FabCache_->MatrixBatchCapacity_; }
};

inline bool MakeSchema(
    SD::RegionSchemaTable& table,
    MacroColumnOfAPC region,
    SD::DataTypeOfMacroColumn dtype,
    SD::SchemaProtocols protocol,
    std::uint32_t height,
    std::uint32_t width,
    std::uint32_t record_count = 0u,
    SD::SchemaFlags flags = SD::SchemaFlags::NONE
) noexcept
{
    SD::RegionSchemaRecord& schema = table[static_cast<std::size_t>(region)];
    schema.Region = region;
    schema.Dtype = dtype;
    schema.Protocol = protocol;
    schema.MatrixHeight = height;
    schema.MatrixWidth = width;
    schema.Flags = flags;

    return SD::SealDesiredSchema(
        schema,
        record_count
    );
}

inline bool SchemaABIAndGeometry() noexcept
{
    SD::RegionSchemaTable disabled{};
    SD::MakeDisabledSchemaTable(disabled);
    for (std::size_t i = 0u; i < disabled.size(); ++i)
    {
        if (
            disabled[i].Region != static_cast<MacroColumnOfAPC>(i) ||
            !SD::HasSchemaFlag(disabled[i].Flags, SD::SchemaFlags::REGION_DISABLED)
        )
        {
            return false;
        }
    }

    SD::RegionSchemaRecord ordinary{};
    ordinary.Region = MacroColumnOfAPC::BOTTOM_UP_SLOT;
    ordinary.Dtype = SD::DataTypeOfMacroColumn::UINT16_T;
    ordinary.Protocol = SD::SchemaProtocols::PRIVATE_REGION;
    ordinary.MatrixHeight = 3u;
    ordinary.MatrixWidth = 4u;
    ordinary.Flags = SD::SchemaFlags::BATCHED_LAST_DIM;

    if (!SD::SealDesiredSchema(
        ordinary,
        0u
    ))
    {
        return false;
    }

    ordinary.CellOffset = ADS::META_CELL_COUNT;
    const auto ordinary_bytes = SD::MatrixByteCount(ordinary);
    const auto ordinary_cells = SD::MatrixCellCount(ordinary);
    const auto ordinary_stride = SD::RecordStrideCells(ordinary);
    const auto ordinary_records = SD::LogicalRecordCount(ordinary);
    const bool ordinary_ok =
        ordinary_bytes == 24u &&
        ordinary_cells == 3u &&
        ordinary_stride == SD::REGION_ALIGNMENT_CELLS &&
        ordinary_records == 1u &&
        ordinary.CellCount == SD::REGION_ALIGNMENT_CELLS &&
        ordinary.EnqueuePosition == SD::NO_POSITION &&
        ordinary.DequeuePosition == SD::NO_POSITION &&
        SD::ValidateStortedRegionSchema(ordinary, MINIMUM_APC_CELL_COUNT, 4u) &&
        !SD::ValidateStortedRegionSchema(ordinary, MINIMUM_APC_CELL_COUNT, 8u);

    SD::RegionSchemaRecord double_buffer = ordinary;
    double_buffer.Region = MacroColumnOfAPC::AUX_SLOT;
    double_buffer.Protocol = SD::SchemaProtocols::DOUBLE_BUFFERED;

    const bool double_ok = SD::SealDesiredSchema(
        double_buffer,
        2u
    );
    double_buffer.CellOffset = ADS::META_CELL_COUNT;

    SD::RegionSchemaRecord queue{};
    queue = ordinary;
    queue.Region = MacroColumnOfAPC::ERROR_SLOT;
    queue.Protocol = SD::SchemaProtocols::MPMC_FIXED_RECORD_QUEUE;
    
    const bool queue_ok = SD::SealDesiredSchema(
        queue,
        4u
    );
    queue.CellOffset = ADS::META_CELL_COUNT;

    SD::RegionSchemaRecord invalid{};
    SD::RegionSchemaRecord invalid_1{};
    SD::RegionSchemaRecord invalid_2{};
    
    invalid_1.Protocol = SD::SchemaProtocols::MPMC_FIXED_RECORD_QUEUE;
    invalid_1.Flags = SD::SchemaFlags::BATCHED_LAST_DIM;

    invalid_2.Protocol = SD::SchemaProtocols::DOUBLE_BUFFERED;

    const bool invalid_shapes_rejected =
        !SD::SealDesiredSchema(
            invalid_1,
            3u
        ) &&
        !SD::SealDesiredSchema(
            invalid,
            2u
        ) &&
        !SD::SealDesiredSchema(
            invalid_2,
            2u
        );

    constexpr std::uint16_t sparse_mask =
        ADS::RegionBit(MacroColumnOfAPC::BOTTOM_UP_SLOT) |
        ADS::RegionBit(MacroColumnOfAPC::STATE_SLOT) |
        ADS::RegionBit(MacroColumnOfAPC::WEIGHT_SLOT) |
        ADS::RegionBit(MacroColumnOfAPC::EXTRA_SLOT);

    const bool compact_ok =
        ADS::CompactRegionIndex(
            sparse_mask, MacroColumnOfAPC::BOTTOM_UP_SLOT
        ) == 0u &&
        ADS::CompactRegionIndex(
            sparse_mask, MacroColumnOfAPC::STATE_SLOT
        ) == 1u &&
        ADS::CompactRegionIndex(
            sparse_mask, MacroColumnOfAPC::WEIGHT_SLOT
        ) == 2u &&
        ADS::CompactRegionIndex(
            sparse_mask, MacroColumnOfAPC::EXTRA_SLOT
        ) == 3u &&
        !ADS::CompactRegionIndex(
            sparse_mask, MacroColumnOfAPC::ERROR_SLOT
        ).has_value();

    return ordinary_ok &&
        double_ok &&
        SD::LogicalRecordCount(double_buffer) == 2u &&
        double_buffer.CellCount == 2u * SD::REGION_ALIGNMENT_CELLS &&
        double_buffer.EnqueuePosition == 1u &&
        double_buffer.DequeuePosition == 0u &&
        SD::ValidateStortedRegionSchema(
            double_buffer, MINIMUM_APC_CELL_COUNT, 4u
        ) &&
        queue_ok &&
        SD::LogicalRecordCount(queue) == 4u &&
        queue.CellCount == 4u * SD::REGION_ALIGNMENT_CELLS &&
        SD::HasSchemaFlag(queue.Flags, SD::SchemaFlags::REQUIRED_POW_OF_TWO) &&
        SD::HasSchemaFlag(queue.Flags, SD::SchemaFlags::HAS_PER_SLOT_SEQUENSE) &&
        SD::ValidateStortedRegionSchema(queue, MINIMUM_APC_CELL_COUNT, 4u) &&
        invalid_shapes_rejected &&
        compact_ok;
}

inline bool FabricConfigurationValidation() noexcept
{
    const auto accepts = [](
        std::uint32_t slot_count,
        std::uint32_t slot_cells,
        SD::FabricRegionConfig config,
        std::uint8_t parents
    ) noexcept
    {
        APCFinilizer fabric{};
        return fabric.InitializeFabric(
            slot_count,
            slot_cells,
            config,
            parents
        );
    };

    constexpr SD::FabricRegionConfig valid = OneRegionConfig();
    constexpr SD::FabricRegionConfig zero_mask{0u, 0u, VIEW_WIDTH};
    constexpr SD::FabricRegionConfig invalid_mask{
        static_cast<std::uint16_t>(
            std::uint16_t{1u} << ADS::CountOfMacroColumn()
        ),
        0u,
        VIEW_WIDTH
    };
    constexpr SD::FabricRegionConfig zero_batch{
        ADS::RegionBit(MacroColumnOfAPC::BOTTOM_UP_SLOT),
        0u,
        0u
    };

    return
        accepts(2u, MINIMUM_APC_CELL_COUNT, valid, 2u) &&
        !accepts(0u, MINIMUM_APC_CELL_COUNT, valid, 2u) &&
        !accepts(2u, MINIMUM_APC_CELL_COUNT + 1u, valid, 2u) &&
        !accepts(2u, MINIMUM_APC_CELL_COUNT, zero_mask, 2u) &&
        !accepts(2u, MINIMUM_APC_CELL_COUNT, invalid_mask, 2u) &&
        !accepts(2u, MINIMUM_APC_CELL_COUNT, zero_batch, 2u) &&
        !accepts(2u, MINIMUM_APC_CELL_COUNT, valid, 0u) &&
        !accepts(
            2u,
            MINIMUM_APC_CELL_COUNT,
            valid,
            static_cast<std::uint8_t>(
                ADS::COMPILED_MAX_DIRECT_PARENTS_PER_AXIS + 1u
            )
        );
}

inline bool CreationValidationAndRollback() noexcept
{
    APCFinilizer fabric{};
    if (!fabric.InitializeFabric(
        1u, MINIMUM_APC_CELL_COUNT, OneRegionConfig(), 2u
    ))
    {
        return false;
    }

    SD::RegionSchemaTable valid{};
    SD::MakeDisabledSchemaTable(valid);
    if (!MakeSchema(
        valid,
        MacroColumnOfAPC::BOTTOM_UP_SLOT,
        SD::DataTypeOfMacroColumn::UINT64_T,
        SD::SchemaProtocols::PRIVATE_REGION,
        1u,
        VIEW_WIDTH,
        0u,
        SD::SchemaFlags::BATCHED_LAST_DIM
    ))
    {
        return false;
    }

    AdaptivePackedCellContainer candidate{};

    SD::RegionSchemaTable missing_active{};
    SD::MakeDisabledSchemaTable(missing_active);
    const bool missing_rejected = !fabric.CreateAPC(candidate, missing_active);

    SD::RegionSchemaTable wrong_region = valid;
    wrong_region[0u].Region = MacroColumnOfAPC::STATE_SLOT;
    const bool wrong_region_rejected = !fabric.CreateAPC(candidate, wrong_region);

    SD::RegionSchemaTable wrong_batch{};
    SD::MakeDisabledSchemaTable(wrong_batch);
    const bool wrong_batch_defined = MakeSchema(
        wrong_batch,
        MacroColumnOfAPC::BOTTOM_UP_SLOT,
        SD::DataTypeOfMacroColumn::UINT64_T,
        SD::SchemaProtocols::PRIVATE_REGION,
        1u,
        VIEW_WIDTH / 2u,
        0u,
        SD::SchemaFlags::BATCHED_LAST_DIM
    );
    const bool wrong_batch_rejected =
        wrong_batch_defined && !fabric.CreateAPC(candidate, wrong_batch);

    SD::RegionSchemaTable oversized{};
    SD::MakeDisabledSchemaTable(oversized);
    const bool oversized_defined = MakeSchema(
        oversized,
        MacroColumnOfAPC::BOTTOM_UP_SLOT,
        SD::DataTypeOfMacroColumn::UINT64_T,
        SD::SchemaProtocols::PRIVATE_REGION,
        MINIMUM_APC_CELL_COUNT,
        VIEW_WIDTH,
        0u,
        SD::SchemaFlags::BATCHED_LAST_DIM
    );
    const bool oversized_rejected =
        oversized_defined && !fabric.CreateAPC(candidate, oversized);

    SD::RegionSchemaTable extra_region = valid;
    const bool extra_defined = MakeSchema(
        extra_region,
        MacroColumnOfAPC::STATE_SLOT,
        SD::DataTypeOfMacroColumn::UINT64_T,
        SD::SchemaProtocols::PRIVATE_REGION,
        1u,
        VIEW_WIDTH,
        0u,
        SD::SchemaFlags::BATCHED_LAST_DIM
    );
    const bool extra_rejected =
        extra_defined && !fabric.CreateAPC(candidate, extra_region);

    const bool candidate_remained_unbound =
        !candidate.IsActiveAPC() &&
        candidate.GetThisSlotIdx() == ADS::APC_INDEX_BOUND_SENTINAL;

    AdaptivePackedCellContainer valid_apc{};
    const bool slot_reusable =
        fabric.CreateAPC(valid_apc, valid) &&
        valid_apc.GetThisSlotIdx() == 0u &&
        valid_apc.Retire();

    return missing_rejected && wrong_region_rejected &&
        wrong_batch_rejected && oversized_rejected && extra_rejected &&
        candidate_remained_unbound && slot_reusable;
}

template <typename T>
constexpr T FirstValue() noexcept
{
    if constexpr (std::is_same_v<T, char>) return 'A';
    else if constexpr (std::is_floating_point_v<T>) return static_cast<T>(1.25);
    else if constexpr (std::is_signed_v<T>) return static_cast<T>(-7);
    else return static_cast<T>(7u);
}

template <typename T>
constexpr T SecondValue() noexcept
{
    if constexpr (std::is_same_v<T, char>) return 'Z';
    else if constexpr (std::is_floating_point_v<T>) return static_cast<T>(3.5);
    else return static_cast<T>(42);
}

template <typename T>
using WrongType = std::conditional_t<std::is_same_v<T, float>, std::uint32_t, float>;

template <typename T>
bool CreateTyped(
    APCFinilizer& fabric,
    AdaptivePackedCellContainer& apc,
    SD::SchemaProtocols region_protocol) noexcept
{
    constexpr auto dtype_value = SD::CppTypeToRegionDType<T>();
    static_assert(dtype_value.has_value());

    SD::RegionSchemaTable schemas{};
    SD::MakeDisabledSchemaTable(schemas);
    return MakeSchema(
        schemas,
        MacroColumnOfAPC::BOTTOM_UP_SLOT,
        dtype_value.value(),
        region_protocol,
        1u,
        VIEW_WIDTH,
        0u,
        SD::SchemaFlags::BATCHED_LAST_DIM
    ) && fabric.CreateAPC(apc, schemas);
}

template <typename T>
bool PrivateCase() noexcept
{
    APCFinilizer fabric{};
    AdaptivePackedCellContainer apc{};
    if (
        !fabric.InitializeFabric(
            2u, MINIMUM_APC_CELL_COUNT, OneRegionConfig(), 2u
        ) ||
        !CreateTyped<T>(fabric, apc, SD::SchemaProtocols::PRIVATE_REGION)
    )
    {
        return false;
    }

    auto view = apc.BuildAViewOverRegion<T>(MacroColumnOfAPC::BOTTOM_UP_SLOT);
    auto wrong = apc.BuildAViewOverRegion<WrongType<T>>(MacroColumnOfAPC::BOTTOM_UP_SLOT);
    if (
        !view.has_value() || !view->IsValid() || view->Size() != VIEW_WIDTH ||
        view->GetProtocol() != SD::SchemaProtocols::PRIVATE_REGION ||
        wrong.has_value() || view->RawMutableSpan().has_value() == false ||
        view->AtomicStore(0u, FirstValue<T>()) ||
        view->AtomicStore(view->Size(), FirstValue<T>()) ||
        apc.BuildAViewOverRegion<T>(
            MacroColumnOfAPC::BOTTOM_UP_SLOT, 1u
        ).has_value() ||
        apc.BuildAViewOverRegion<T>(
            MacroColumnOfAPC::ERROR_SLOT
        ).has_value()
    )
    {
        return false;
    }

    auto span = view->RawMutableSpan();
    span.value()[0u] = FirstValue<T>();
    span.value()[span->size() / 2u] = SecondValue<T>();
    span.value().back() = FirstValue<T>();
    if (!apc.ZeroARegion<T>(MacroColumnOfAPC::BOTTOM_UP_SLOT)) return false;
    return std::all_of(span->begin(), span->end(), [](T value) { return value == T{}; });
}

template <typename T>
bool AtomicCase() noexcept
{
    APCFinilizer fabric{};
    AdaptivePackedCellContainer apc{};
    if (
        !fabric.InitializeFabric(
            2u, MINIMUM_APC_CELL_COUNT, OneRegionConfig(), 2u
        ) ||
        !CreateTyped<T>(fabric, apc, SD::SchemaProtocols::ATOMIC_WORD_ARRAY)
    )
    {
        return false;
    }

    auto view = apc.BuildAViewOverRegion<T>(MacroColumnOfAPC::BOTTOM_UP_SLOT);
    auto wrong = apc.BuildAViewOverRegion<WrongType<T>>(MacroColumnOfAPC::BOTTOM_UP_SLOT);
    if (
        !view.has_value() || !view->IsValid() || view->Size() != VIEW_WIDTH ||
        view->GetProtocol() != SD::SchemaProtocols::ATOMIC_WORD_ARRAY ||
        view->RawMutableSpan().has_value() || wrong.has_value() ||
        view->AtomicStore(view->Size(), FirstValue<T>())
    )
    {
        return false;
    }

    const std::size_t middle = view->Size() / 2u;
    if (
        !view->AtomicStore(0u, FirstValue<T>(), std::memory_order_relaxed) ||
        !view->AtomicStore(middle, SecondValue<T>(), std::memory_order_release) ||
        view->AtomicLoad(0u, std::memory_order_relaxed) != FirstValue<T>() ||
        view->AtomicLoad(middle, std::memory_order_acquire) != SecondValue<T>()
    )
    {
        return false;
    }

    T expected = FirstValue<T>();
    if (
        !view->AtomicCompareExchangeStrong(
            0u,
            expected,
            SecondValue<T>(),
            std::memory_order_acq_rel,
            std::memory_order_acquire
        ) ||
        !apc.ZeroARegion<T>(MacroColumnOfAPC::BOTTOM_UP_SLOT)
    )
    {
        return false;
    }

    for (std::size_t i = 0u; i < view->Size(); ++i)
    {
        if (view->AtomicLoad(i, std::memory_order_relaxed) != T{}) return false;
    }
    return true;
}

template <typename T>
bool ImmutableCase() noexcept
{
    APCFinilizer fabric{};
    AdaptivePackedCellContainer apc{};
    if (
        !fabric.InitializeFabric(
            2u, MINIMUM_APC_CELL_COUNT, OneRegionConfig(), 2u
        ) ||
        !CreateTyped<T>(fabric, apc, SD::SchemaProtocols::IMMUTABLE_SNAPSHOT)
    )
    {
        return false;
    }

    auto view = apc.BuildAViewOverRegion<T>(
        MacroColumnOfAPC::BOTTOM_UP_SLOT
    );
    return
        view.has_value() &&
        view->IsValid() &&
        view->Size() == VIEW_WIDTH &&
        view->GetProtocol() == SD::SchemaProtocols::IMMUTABLE_SNAPSHOT &&
        !view->RawMutableSpan().has_value() &&
        !view->AtomicStore(0u, FirstValue<T>()) &&
        !apc.ZeroARegion<T>(MacroColumnOfAPC::BOTTOM_UP_SLOT);
}

template <typename T>
bool RunType(const char* name)
{
    const bool private_ok = PrivateCase<T>();
    const bool atomic_ok = AtomicCase<T>();
    const bool immutable_ok = ImmutableCase<T>();
    std::cout
        << "  " << std::left << std::setw(10) << name
        << " private=" << (private_ok ? "PASS" : "FAIL")
        << " atomic=" << (atomic_ok ? "PASS" : "FAIL")
        << " immutable=" << (immutable_ok ? "PASS" : "FAIL") << '\n';
    return private_ok && atomic_ok && immutable_ok;
}

inline bool DeviceViewAndProtocolStorage() noexcept
{
    constexpr std::uint32_t batch = 4u;
    constexpr std::uint32_t slot_count = 3u;
    constexpr std::uint16_t active_mask =
        ADS::RegionBit(MacroColumnOfAPC::BOTTOM_UP_SLOT) |
        ADS::RegionBit(MacroColumnOfAPC::STATE_SLOT) |
        ADS::RegionBit(MacroColumnOfAPC::ERROR_SLOT) |
        ADS::RegionBit(MacroColumnOfAPC::WEIGHT_SLOT) |
        ADS::RegionBit(MacroColumnOfAPC::AUX_SLOT);
    constexpr std::uint8_t active_count = 5u;
    constexpr std::uint16_t row_cells =
        active_count * static_cast<std::uint16_t>(SD::RegionSchemaCellCount());

    InspectableFabric fabric{};
    const SD::FabricRegionConfig config{active_mask, 0u, batch};
    if (!fabric.InitializeFabric(
        slot_count, MINIMUM_APC_CELL_COUNT, config, 2u
    ))
    {
        std::cout << "    detail: valid mixed-region Fabric initialization was rejected\n";
        return false;
    }

    const auto matrix_range =
        fabric.TableRange(
            FabricSegments::MATRIX_VIEW_TABLE
        );

    const CoreOfFabricCoordinator::FabricCache*
        header = fabric.FabricHeader();

    const bool fabric_metadata_ok =
        header != nullptr &&
        fabric.ActiveMask() == active_mask &&
        fabric.ActiveCount() == active_count &&
        fabric.ViewRowCells() == row_cells &&
        fabric.BatchCapacity() == batch &&
        matrix_range.has_value() &&
        matrix_range->EndIndex -
            matrix_range->BeginIndex ==
            static_cast<std::size_t>(
                slot_count
            ) * row_cells &&
        fabric.MatrixViewBegin() ==
            matrix_range->BeginIndex &&
        header->FormateVersion_ ==
            CoreOfFabricCoordinator::
                FORMAT_VERSION &&
        header->CountOfAPC_ ==
            slot_count &&
        header->ActiveRegionMask_ ==
            active_mask &&
        header->ActiveRegionCount_ ==
            active_count &&
        header->MatrixViewRowCellCount_ ==
            row_cells &&
        header->MatrixBatchCapacity_ ==
            batch &&
        header->RegionAlignmentCellCount_ ==
            SD::REGION_ALIGNMENT_CELLS;

    SD::RegionSchemaTable schemas{};
    SD::MakeDisabledSchemaTable(schemas);
    if (
        !MakeSchema(
            schemas,
            MacroColumnOfAPC::BOTTOM_UP_SLOT,
            SD::DataTypeOfMacroColumn::FLOAT32_T,
            SD::SchemaProtocols::PRIVATE_REGION,
            3u,
            batch,
            0u,
            SD::SchemaFlags::BATCHED_LAST_DIM
        ) ||
        !MakeSchema(
            schemas,
            MacroColumnOfAPC::STATE_SLOT,
            SD::DataTypeOfMacroColumn::FLOAT32_T,
            SD::SchemaProtocols::ATOMIC_WORD_ARRAY,
            2u,
            batch,
            0u,
            SD::SchemaFlags::BATCHED_LAST_DIM
        ) ||
        !MakeSchema(
            schemas,
            MacroColumnOfAPC::ERROR_SLOT,
            SD::DataTypeOfMacroColumn::FLOAT32_T,
            SD::SchemaProtocols::MPMC_FIXED_RECORD_QUEUE,
            1u,
            batch,
            4u,
            SD::SchemaFlags::BATCHED_LAST_DIM
        ) ||
        !MakeSchema(
            schemas,
            MacroColumnOfAPC::WEIGHT_SLOT,
            SD::DataTypeOfMacroColumn::FLOAT32_T,
            SD::SchemaProtocols::IMMUTABLE_SNAPSHOT,
            5u,
            3u
        ) ||
        !MakeSchema(
            schemas,
            MacroColumnOfAPC::AUX_SLOT,
            SD::DataTypeOfMacroColumn::UINT64_T,
            SD::SchemaProtocols::DOUBLE_BUFFERED,
            1u,
            batch,
            2u,
            SD::SchemaFlags::BATCHED_LAST_DIM
        )
    )
    {
        std::cout << "    detail: one or more valid RegionSchemaRecord definitions were rejected\n";
        return false;
    }

    AdaptivePackedCellContainer apc{};
    if (!fabric.CreateAPC(apc, schemas))
    {
        std::cout
            << "    detail: Fabric metadata precheck="
            << (fabric_metadata_ok ? "PASS" : "FAIL")
            << "; CreateAPC rejected the valid compact schema row\n";
        return false;
    }

    SD::RegionSchemaTable second_schemas = schemas;
    if (!MakeSchema(
        second_schemas,
        MacroColumnOfAPC::BOTTOM_UP_SLOT,
        SD::DataTypeOfMacroColumn::FLOAT32_T,
        SD::SchemaProtocols::PRIVATE_REGION,
        5u,
        batch,
        0u,
        SD::SchemaFlags::BATCHED_LAST_DIM
    ))
    {
        return false;
    }

    AdaptivePackedCellContainer second_apc{};
    if (!fabric.CreateAPC(second_apc, second_schemas))
    {
        std::cout << "    detail: second compact DEVICE_VIEW_TABLE row was rejected\n";
        return false;
    }

    const std::uint32_t slot = apc.GetThisSlotIdx();
    const std::span<const SD::RegionSchemaRecord> row = fabric.SchemaRow(slot);
    const std::span<const SD::RegionSchemaRecord> second_row =
        fabric.SchemaRow(second_apc.GetThisSlotIdx());
    constexpr std::array<MacroColumnOfAPC, active_count> expected_regions{
        MacroColumnOfAPC::BOTTOM_UP_SLOT,
        MacroColumnOfAPC::STATE_SLOT,
        MacroColumnOfAPC::ERROR_SLOT,
        MacroColumnOfAPC::WEIGHT_SLOT,
        MacroColumnOfAPC::AUX_SLOT
    };

    bool row_ok = row.size() == active_count;
    std::uint32_t expected_offset = ADS::META_CELL_COUNT;
    for (std::size_t i = 0u; row_ok && i < row.size(); ++i)
    {
        expected_offset = SD::AlignRegionCells(expected_offset);
        const SD::RegionSchemaRecord& record = row[i];
        row_ok =
            record.Region == expected_regions[i] &&
            record.CellOffset == expected_offset &&
            record.CellOffset % SD::REGION_ALIGNMENT_CELLS == 0u &&
            record.SeqLockCounter == 0u &&
            SD::FreshProtocolState(record) &&
            SD::ValidateStortedRegionSchema(
                record, MINIMUM_APC_CELL_COUNT, batch
            );
        expected_offset = record.CellOffset + record.CellCount;
    }
    row_ok = row_ok && expected_offset <= MINIMUM_APC_CELL_COUNT;
    const bool row_stride_and_isolation_ok =
        second_row.size() == active_count &&
        reinterpret_cast<std::uintptr_t>(second_row.data()) -
            reinterpret_cast<std::uintptr_t>(row.data()) ==
            static_cast<std::uintptr_t>(row_cells) * sizeof(std::uint64_t) &&
        row[0u].MatrixHeight == 3u &&
        second_row[0u].MatrixHeight == 5u &&
        row[0u].CellOffset == ADS::META_CELL_COUNT &&
        second_row[0u].CellOffset == ADS::META_CELL_COUNT;

    const bool header_ok =
        fabric.ReadAPCLocalCell(
            slot, static_cast<std::uint32_t>(ADS::HeaderIdentifierOfAPC::MAGIC_ID)
        ) == ADS::BRANCH_MAGIC &&
        fabric.ReadAPCLocalCell(
            slot, static_cast<std::uint32_t>(ADS::HeaderIdentifierOfAPC::APC_SLOT_IDX)
        ) == slot &&
        fabric.ReadAPCLocalCell(
            slot, static_cast<std::uint32_t>(ADS::HeaderIdentifierOfAPC::EOF_APC_HEADER)
        ) == ADS::EOF_HEADER;

    bool queue_storage_ok = row_ok;
    if (queue_storage_ok)
    {
        const SD::RegionSchemaRecord& queue = row[2u];
        const auto matrix_cells = SD::MatrixCellCount(queue);
        const auto stride_cells = SD::RecordStrideCells(queue);
        const auto record_count = SD::LogicalRecordCount(queue);
        queue_storage_ok =
            matrix_cells.has_value() &&
            stride_cells.has_value() &&
            record_count == 4u &&
            queue.EnqueuePosition == 0u &&
            queue.DequeuePosition == 0u;

        for (std::uint32_t i = 0u;
            queue_storage_ok && i < record_count.value();
            ++i)
        {
            const std::uint32_t sequence_cell =
                queue.CellOffset + i * stride_cells.value() + matrix_cells.value();
            queue_storage_ok = fabric.ReadAPCLocalCell(slot, sequence_cell) == i;
        }
    }

    auto feedforward = apc.BuildAViewOverRegion<float>(
        MacroColumnOfAPC::BOTTOM_UP_SLOT
    );
    auto state = apc.BuildAViewOverRegion<float>(MacroColumnOfAPC::STATE_SLOT);
    auto weight = apc.BuildAViewOverRegion<float>(MacroColumnOfAPC::WEIGHT_SLOT);
    auto queue_view = apc.BuildAViewOverRegion<float>(MacroColumnOfAPC::ERROR_SLOT);
    auto double_view = apc.BuildAViewOverRegion<std::uint64_t>(
        MacroColumnOfAPC::AUX_SLOT
    );
    auto wrong_type = apc.BuildAViewOverRegion<double>(
        MacroColumnOfAPC::BOTTOM_UP_SLOT
    );
    auto inactive = apc.BuildAViewOverRegion<float>(
        MacroColumnOfAPC::EXTRA_SLOT
    );
    auto second_feedforward = second_apc.BuildAViewOverRegion<float>(
        MacroColumnOfAPC::BOTTOM_UP_SLOT
    );

    const bool public_views_ok =
        feedforward.has_value() && feedforward->IsValid() &&
        feedforward->Size() == 3u * batch &&
        feedforward->GetProtocol() == SD::SchemaProtocols::PRIVATE_REGION &&
        feedforward->RawMutableSpan().has_value() &&
        state.has_value() && state->IsValid() &&
        state->Size() == 2u * batch &&
        state->GetProtocol() == SD::SchemaProtocols::ATOMIC_WORD_ARRAY &&
        !state->RawMutableSpan().has_value() &&
        weight.has_value() && weight->IsValid() &&
        weight->Size() == 15u &&
        weight->GetProtocol() == SD::SchemaProtocols::IMMUTABLE_SNAPSHOT &&
        !weight->RawMutableSpan().has_value() &&
        !queue_view.has_value() &&
        !double_view.has_value() &&
        !wrong_type.has_value() &&
        !inactive.has_value() &&
        second_feedforward.has_value() &&
        second_feedforward->Size() == 5u * batch;

    feedforward.reset();
    state.reset();
    weight.reset();
    queue_view.reset();
    double_view.reset();
    wrong_type.reset();
    inactive.reset();
    second_feedforward.reset();

    const bool retire_ok = apc.Retire() && second_apc.Retire();
    const bool ok = fabric_metadata_ok && row_ok &&
        row_stride_and_isolation_ok && header_ok && queue_storage_ok &&
        public_views_ok && retire_ok;
    if (!ok)
    {
        std::cout
            << "    detail: metadata=" << (fabric_metadata_ok ? "PASS" : "FAIL")
            << " row=" << (row_ok ? "PASS" : "FAIL")
            << " row-stride=" << (row_stride_and_isolation_ok ? "PASS" : "FAIL")
            << " header=" << (header_ok ? "PASS" : "FAIL")
            << " MPMC=" << (queue_storage_ok ? "PASS" : "FAIL")
            << " views=" << (public_views_ok ? "PASS" : "FAIL")
            << " retire=" << (retire_ok ? "PASS" : "FAIL") << '\n';
    }
    return ok;
}

inline bool SlotReuseClearsPayloadAndSchema() noexcept
{
    InspectableFabric fabric{};
    if (!fabric.InitializeFabric(
        1u, MINIMUM_APC_CELL_COUNT, OneRegionConfig(), 2u
    ))
    {
        std::cout << "    detail: single-slot Fabric initialization was rejected\n";
        return false;
    }

    SD::RegionSchemaTable first_schema{};
    SD::MakeDisabledSchemaTable(first_schema);
    if (!MakeSchema(
        first_schema,
        MacroColumnOfAPC::BOTTOM_UP_SLOT,
        SD::DataTypeOfMacroColumn::UINT64_T,
        SD::SchemaProtocols::PRIVATE_REGION,
        1u,
        VIEW_WIDTH,
        0u,
        SD::SchemaFlags::BATCHED_LAST_DIM
    ))
    {
        return false;
    }

    AdaptivePackedCellContainer first{};
    if (!fabric.CreateAPC(first, first_schema))
    {
        std::cout << "    detail: first APC creation was rejected\n";
        return false;
    }
    const std::uint32_t retired_slot = first.GetThisSlotIdx();
    {
        auto first_view = first.BuildAViewOverRegion<std::uint64_t>(
            MacroColumnOfAPC::BOTTOM_UP_SLOT
        );
        if (!first_view.has_value() || !first_view->RawMutableSpan().has_value())
        {
            return false;
        }
        const std::optional<std::span<std::uint64_t>> mutable_span =
            first_view->RawMutableSpan();
        for (std::uint64_t& value : mutable_span.value())
        {
            value = UINT64_MAX;
        }
    }
    if (!first.Retire() || first.IsActiveAPC())
    {
        return false;
    }

    SD::RegionSchemaTable replacement_schema{};
    SD::MakeDisabledSchemaTable(replacement_schema);
    if (!MakeSchema(
        replacement_schema,
        MacroColumnOfAPC::BOTTOM_UP_SLOT,
        SD::DataTypeOfMacroColumn::UINT32_T,
        SD::SchemaProtocols::ATOMIC_WORD_ARRAY,
        2u,
        VIEW_WIDTH,
        0u,
        SD::SchemaFlags::BATCHED_LAST_DIM
    ))
    {
        return false;
    }

    AdaptivePackedCellContainer replacement{};
    if (
        !fabric.CreateAPC(replacement, replacement_schema) ||
        replacement.GetThisSlotIdx() != retired_slot ||
        first.IsActiveAPC()
    )
    {
        std::cout << "    detail: retired-slot reclamation or replacement creation failed\n";
        return false;
    }

    auto replacement_view = replacement.BuildAViewOverRegion<std::uint32_t>(
        MacroColumnOfAPC::BOTTOM_UP_SLOT
    );
    auto stale_dtype = replacement.BuildAViewOverRegion<std::uint64_t>(
        MacroColumnOfAPC::BOTTOM_UP_SLOT
    );
    bool zeroed =
        replacement_view.has_value() &&
        replacement_view->Size() == 2u * VIEW_WIDTH &&
        replacement_view->GetProtocol() == SD::SchemaProtocols::ATOMIC_WORD_ARRAY &&
        !stale_dtype.has_value();
    for (std::size_t i = 0u;
        zeroed && i < replacement_view->Size();
        ++i)
    {
        zeroed = replacement_view->AtomicLoad(
            i, std::memory_order_relaxed
        ) == 0u;
    }

    replacement_view.reset();
    stale_dtype.reset();
    return zeroed && replacement.Retire();
}

inline Result Run()
{
    Banner("TEST 6 - REGION SCHEMA / DEVICE VIEW / PROTOCOL STORAGE / DTYPES");
    const bool abi_geometry_ok = SchemaABIAndGeometry();
    const bool config_ok = FabricConfigurationValidation();
    const bool rollback_ok = CreationValidationAndRollback();
    const bool device_view_ok = DeviceViewAndProtocolStorage();
    const bool reuse_ok = SlotReuseClearsPayloadAndSchema();

    std::cout
        << "  8-cell header + 40-byte schema ABI     : "
        << (abi_geometry_ok ? "PASS" : "FAIL") << '\n'
        << "  Fabric construction validation         : "
        << (config_ok ? "PASS" : "FAIL") << '\n'
        << "  invalid schema rollback + slot reuse   : "
        << (rollback_ok ? "PASS" : "FAIL") << '\n'
        << "  compact rows + metadata + MPMC sequence: "
        << (device_view_ok ? "PASS" : "FAIL") << '\n'
        << "  retirement/reuse clears data and schema: "
        << (reuse_ok ? "PASS" : "FAIL") << "\n\n";

    bool primitive_ok = true;
    primitive_ok = RunType<std::uint8_t>("uint8_t") && primitive_ok;
    primitive_ok = RunType<std::uint16_t>("uint16_t") && primitive_ok;
    primitive_ok = RunType<std::uint32_t>("uint32_t") && primitive_ok;
    primitive_ok = RunType<std::uint64_t>("uint64_t") && primitive_ok;
    primitive_ok = RunType<std::int8_t>("int8_t") && primitive_ok;
    primitive_ok = RunType<std::int16_t>("int16_t") && primitive_ok;
    primitive_ok = RunType<std::int32_t>("int32_t") && primitive_ok;
    primitive_ok = RunType<std::int64_t>("int64_t") && primitive_ok;
    primitive_ok = RunType<float>("float") && primitive_ok;
    primitive_ok = RunType<double>("double") && primitive_ok;
    primitive_ok = RunType<char>("char") && primitive_ok;

    const bool ok = abi_geometry_ok && config_ok && rollback_ok &&
        device_view_ok && reuse_ok && primitive_ok;
    std::cout << "\nTEST 6 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test06_RegionSchemaAndViews

// -----------------------------------------------------------------------------
// Test 7: concurrent mixed-axis mutation proof plus retirement/ABA lifecycle.
// -----------------------------------------------------------------------------

namespace Test07_ConcurrentDAGAndRetirement
{
inline bool FixedOrderRace()
{
    constexpr std::uint32_t ROUNDS = 10'000u;
    APCFabricBackend<2u, 1u, 2u> backend{};
    if (!backend.Initialize()) return false;

    std::barrier phase(3);
    std::atomic<std::uint32_t> valid_success{0u};
    std::atomic<std::uint32_t> invalid_success{0u};

    std::thread valid([&]() noexcept
    {
        for (std::uint32_t i = 0u; i < ROUNDS; ++i)
        {
            phase.arrive_and_wait();
            if (backend.AddParent(0u, 1u, Axis::HORIZONTAL))
            {
                valid_success.fetch_add(1u, std::memory_order_relaxed);
            }
            phase.arrive_and_wait();
        }
    });

    std::thread invalid([&]() noexcept
    {
        for (std::uint32_t i = 0u; i < ROUNDS; ++i)
        {
            phase.arrive_and_wait();
            if (backend.AddParent(1u, 0u, Axis::VERTICAL))
            {
                invalid_success.fetch_add(1u, std::memory_order_relaxed);
            }
            phase.arrive_and_wait();
        }
    });

    bool main_ok = true;
    for (std::uint32_t i = 0u; i < ROUNDS; ++i)
    {
        phase.arrive_and_wait();
        phase.arrive_and_wait();
        if (!backend.RemoveParent(0u, 1u, Axis::HORIZONTAL))
        {
            main_ok = false;
        }
    }

    valid.join();
    invalid.join();
    const GraphProof proof = ProveQuiescentCombinedDAG<2u, 2u>(backend);
    return main_ok &&
        valid_success.load() == ROUNDS &&
        invalid_success.load() == 0u &&
        proof.Passed();
}

inline bool MixedAxisStress(std::uint64_t& retries_out)
{
    constexpr std::size_t N = 32u;
    constexpr std::uint8_t K = 4u;
    constexpr std::size_t WORKERS = 8u;
    constexpr std::size_t FIRST_CHILD = 8u;
    constexpr std::uint32_t ROUNDS = 5'000u;

    APCFabricBackend<N, 1u, K> backend{};
    if (!backend.Initialize()) return false;
    for (std::size_t i = 0u; i < WORKERS; ++i)
    {
        if (
            !backend.AddParent(0u, FIRST_CHILD + i, Axis::HORIZONTAL) ||
            !backend.AddParent(2u, FIRST_CHILD + i, Axis::VERTICAL)
        )
        {
            return false;
        }
    }

    std::barrier start(static_cast<std::ptrdiff_t>(WORKERS + 1u));
    std::atomic<bool> failed{false};
    std::atomic<std::uint64_t> retries{0u};
    std::vector<std::thread> workers;
    workers.reserve(WORKERS);

    for (std::size_t worker = 0u; worker < WORKERS; ++worker)
    {
        workers.emplace_back([&, worker]() noexcept
        {
            const std::size_t child = FIRST_CHILD + worker;
            std::size_t h_current = 0u;
            std::size_t v_current = 2u;
            std::uint64_t local_retries = 0u;
            start.arrive_and_wait();

            for (std::uint32_t i = 0u; i < ROUNDS; ++i)
            {
                const std::size_t h_next = h_current == 0u ? 1u : 0u;
                const std::size_t v_next = v_current == 2u ? 3u : 2u;
                if (
                    !RetryReplace(
                        backend, h_current, h_next, child,
                        Axis::HORIZONTAL, local_retries
                    ) ||
                    !RetryReplace(
                        backend, v_current, v_next, child,
                        Axis::VERTICAL, local_retries
                    )
                )
                {
                    failed.store(true, std::memory_order_release);
                    break;
                }
                h_current = h_next;
                v_current = v_next;
            }
            retries.fetch_add(local_retries, std::memory_order_relaxed);
        });
    }

    start.arrive_and_wait();
    for (std::thread& worker : workers) worker.join();
    retries_out = retries.load(std::memory_order_acquire);

    const GraphProof proof = ProveQuiescentCombinedDAG<N, K>(backend);
    return !failed.load(std::memory_order_acquire) && proof.Passed();
}

constexpr SchemaDefinition::FabricRegionConfig AtomicRegionConfig() noexcept
{
    return SchemaDefinition::FabricRegionConfig{
        ADS::RegionBit(MacroColumnOfAPC::BOTTOM_UP_SLOT),
        0u,
        8u
    };
}

inline bool CreateAtomic(
    APCFinilizer& fabric,
    AdaptivePackedCellContainer& apc) noexcept
{
    SchemaDefinition::RegionSchemaTable schemas{};
    SchemaDefinition::MakeDisabledSchemaTable(schemas);

    SchemaDefinition::RegionSchemaRecord& schema = schemas[static_cast<std::size_t>(
            MacroColumnOfAPC::BOTTOM_UP_SLOT
    )];
    schema.Region = MacroColumnOfAPC::BOTTOM_UP_SLOT;
    schema.Dtype = SchemaDefinition::DataTypeOfMacroColumn::UINT64_T;
    schema.Protocol = SchemaDefinition::SchemaProtocols::ATOMIC_WORD_ARRAY;
    schema.MatrixHeight = 1u;
    schema.MatrixWidth = 8u;
    schema.Flags = SchemaDefinition::SchemaFlags::BATCHED_LAST_DIM;

    return SchemaDefinition::SealDesiredSchema(
        schema,
        0u
    ) && fabric.CreateAPC(apc, schemas);
}

inline bool RetirementAndABA()
{
    APCFinilizer fabric{};
    AdaptivePackedCellContainer parent{};
    AdaptivePackedCellContainer child{};
    AdaptivePackedCellContainer replacement{};

    if (
        !fabric.InitializeFabric(
            2u, MINIMUM_APC_CELL_COUNT, AtomicRegionConfig(), 2u
        ) ||
        !CreateAtomic(fabric, parent) ||
        !CreateAtomic(fabric, child)
    )
    {
        return false;
    }

    const std::uint32_t child_slot = child.GetThisSlotIdx();
    if (
        !child.AddParent(parent, FabricSegments::VALUE_PARENT_EDGE_TABLE_H) ||
        !child.AddParent(parent, FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V) ||
        parent.Retire() ||
        child.Retire() ||
        !child.RemoveParent(parent, FabricSegments::VALUE_PARENT_EDGE_TABLE_H) ||
        child.Retire() ||
        !child.RemoveParent(parent, FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V)
    )
    {
        return false;
    }

    auto held_view = child.BuildAViewOverRegion<std::uint64_t>(
        MacroColumnOfAPC::BOTTOM_UP_SLOT
    );
    if (!held_view.has_value() || child.Retire(1u))
    {
        return false;
    }
    held_view.reset();

    if (
        !child.Retire() ||
        child.IsActiveAPC() ||
        child.BuildAViewOverRegion<std::uint64_t>(
            MacroColumnOfAPC::BOTTOM_UP_SLOT
        ).has_value() ||
        !CreateAtomic(fabric, replacement) ||
        replacement.GetThisSlotIdx() != child_slot ||
        !replacement.IsActiveAPC() ||
        child.IsActiveAPC()
    )
    {
        return false;
    }

    return parent.Retire() && replacement.Retire();
}

inline bool ShutdownDrainsOutstandingView()
{
    ResolverTestFabric fabric{};
    TestAPC apc{};

    if (
        !fabric.InitializeFabric(
            1u, MINIMUM_APC_CELL_COUNT, AtomicRegionConfig(), 2u
        ) ||
        !CreateAtomic(fabric, apc)
    )
    {
        return false;
    }

    auto held_view = apc.BuildAViewOverRegion<std::uint64_t>(
        MacroColumnOfAPC::BOTTOM_UP_SLOT
    );
    if (!held_view.has_value())
    {
        return false;
    }

    std::atomic<bool> finished{false};
    std::thread shutdown([&]() noexcept
    {
        fabric.ShutDownFabric();
        finished.store(true, std::memory_order_release);
    });

    for (std::uint32_t spins = 0u; spins < 100'000u && fabric.IsFabricActive(); ++spins)
    {
        PerturbSchedule(spins);
    }

    const bool entered_shutdown = !fabric.IsFabricActive();
    const bool correctly_waiting =
        entered_shutdown && !finished.load(std::memory_order_acquire);

    held_view.reset();
    shutdown.join();

    return correctly_waiting &&
        finished.load(std::memory_order_acquire) &&
        !fabric.IsFabricActive();
}

inline bool DirectResolverAndABA()
{
    ResolverTestFabric fabric{};
    TestAPC original{};
    TestAPC replacement{};

    if (
        !fabric.InitializeFabric(
            1u, MINIMUM_APC_CELL_COUNT, AtomicRegionConfig(), 2u
        ) ||
        !CreateAtomic(fabric, original)
    )
    {
        return false;
    }

    const std::uint32_t slot = original.GetThisSlotIdx();
    const std::uint32_t generation_one = original.GenerationForTest();
    if (
        slot == ADS::APC_INDEX_BOUND_SENTINAL ||
        !HandleOfAPCStatic::IsGenerationValid(generation_one)
    )
    {
        return false;
    }

    {
        AdaptivePackedCellContainer current{};
        APCUseScope current_use{};
        if (
            !fabric.ResolveExistingForTest(slot, current, current_use) ||
            current.GetThisSlotIdx() != slot
        )
        {
            return false;
        }
    }

    {
        AdaptivePackedCellContainer exact{};
        APCUseScope exact_use{};
        if (
            !fabric.ResolveExistingForTest(
                slot, exact, exact_use, generation_one
            ) ||
            exact.GetThisSlotIdx() != slot
        )
        {
            return false;
        }
    }

    if (
        !original.Retire() ||
        !CreateAtomic(fabric, replacement) ||
        replacement.GetThisSlotIdx() != slot
    )
    {
        return false;
    }

    const std::uint32_t generation_two = replacement.GenerationForTest();
    if (
        !HandleOfAPCStatic::IsGenerationValid(generation_two) ||
        generation_two == generation_one
    )
    {
        return false;
    }

    {
        AdaptivePackedCellContainer stale{};
        APCUseScope stale_use{};
        if (fabric.ResolveExistingForTest(
            slot, stale, stale_use, generation_one
        ))
        {
            return false;
        }
    }

    {
        AdaptivePackedCellContainer current{};
        APCUseScope current_use{};
        if (
            !fabric.ResolveExistingForTest(slot, current, current_use) ||
            current.GetThisSlotIdx() != slot
        )
        {
            return false;
        }
    }

    {
        AdaptivePackedCellContainer exact{};
        APCUseScope exact_use{};
        if (
            !fabric.ResolveExistingForTest(
                slot, exact, exact_use, generation_two
            ) ||
            exact.GetThisSlotIdx() != slot
        )
        {
            return false;
        }
    }

    return replacement.Retire();
}


inline Result Run()
{
    Banner("TEST 7 - CONCURRENT H/V DAG MUTATION + RETIREMENT / ABA");
    std::uint64_t mixed_retries = 0u;
    const bool race_ok = FixedOrderRace();
    const bool stress_ok = MixedAxisStress(mixed_retries);
    const bool retirement_ok = RetirementAndABA();
    const bool resolver_ok = DirectResolverAndABA();
    const bool shutdown_ok = ShutdownDrainsOutstandingView();
    const bool ok = race_ok && stress_ok && retirement_ok && resolver_ok && shutdown_ok;

    std::cout
        << "  A--H-->B raced with illegal B--V-->A : " << (race_ok ? "PASS" : "FAIL") << '\n'
        << "  shared-parent mixed H/V stress       : " << (stress_ok ? "PASS" : "FAIL") << '\n'
        << "  transaction retries observed         : " << mixed_retries << '\n'
        << "  linked/pinned retirement + ABA reuse : " << (retirement_ok ? "PASS" : "FAIL") << '\n'
        << "  direct slot/generation resolver + ABA: " << (resolver_ok ? "PASS" : "FAIL") << '\n'
        << "  shutdown drains outstanding RegionView: " << (shutdown_ok ? "PASS" : "FAIL") << '\n'
        << "\nTEST 7 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test07_ConcurrentDAGAndRetirement



// -----------------------------------------------------------------------------
// Test 8: full-slab Save / Attach / Detach relocation.
//
// This is deliberately a Fabric/APC test, not a GHGF test. It proves that a
// quiescent slab image can move to a different host address without rebuilding
// APCs, schemas, payloads, or the H relation.
// -----------------------------------------------------------------------------

namespace Test08_FabricRelocation
{
using SD = SchemaDefinition;

constexpr std::uint32_t VIEW_WIDTH = 8u;
constexpr std::uint32_t SLOT_COUNT = 4u;
constexpr std::uint8_t PARENT_CAPACITY = 2u;

class RelocationFabric final : public APCFinilizer
{
public:
    bool ResolveExistingForTest(
        std::uint32_t slot,
        AdaptivePackedCellContainer& apc,
        APCUseScope& use,
        std::optional<std::uint32_t> expected_generation =
            std::nullopt
    ) noexcept
    {
        return GetExistingAPC_(
            slot,
            apc,
            use,
            expected_generation
        );
    }

    std::uint64_t* SlabAddressForTest() noexcept
    {
        return SlabBasePtr_;
    }

    std::uint64_t SlabCellCountForTest() const noexcept
    {
        return FabCache_
            ? FabCache_->SlabCellCount_
            : UNSIGNED_ZERO;
    }
};

constexpr SD::FabricRegionConfig
RegionConfig() noexcept
{
    return SD::FabricRegionConfig{
        ADS::RegionBit(
            MacroColumnOfAPC::BOTTOM_UP_SLOT
        ),
        0u,
        VIEW_WIDTH,
        false
    };
}

inline bool CreateAtomicNode(
    APCFinilizer& fabric,
    AdaptivePackedCellContainer& apc
) noexcept
{
    SD::RegionSchemaTable schemas{};
    SD::MakeDisabledSchemaTable(schemas);

    SD::RegionSchemaRecord& schema =
        schemas[static_cast<std::size_t>(
            MacroColumnOfAPC::BOTTOM_UP_SLOT
        )];

    schema.Region =
        MacroColumnOfAPC::BOTTOM_UP_SLOT;
    schema.Dtype =
        SD::DataTypeOfMacroColumn::UINT64_T;
    schema.Protocol =
        SD::SchemaProtocols::ATOMIC_WORD_ARRAY;
    schema.MatrixHeight = 1u;
    schema.MatrixWidth = VIEW_WIDTH;
    schema.Flags =
        SD::SchemaFlags::BATCHED_LAST_DIM;

    return
        SD::SealDesiredSchema(
            schema,
            0u
        ) &&
        fabric.CreateAPC(
            apc,
            schemas
        );
}

inline bool WritePayload(
    AdaptivePackedCellContainer& apc,
    std::uint64_t seed
) noexcept
{
    auto view =
        apc.BuildAViewOverRegion<std::uint64_t>(
            MacroColumnOfAPC::BOTTOM_UP_SLOT
        );

    if (
        !view.has_value() ||
        !view->IsValid() ||
        view->Size() != VIEW_WIDTH
    )
    {
        return false;
    }

    for (
        std::size_t i = 0u;
        i < VIEW_WIDTH;
        ++i
    )
    {
        if (!view->AtomicStore(
            i,
            seed +
                static_cast<std::uint64_t>(i),
            std::memory_order_release
        ))
        {
            return false;
        }
    }

    return true;
}

inline bool CheckPayload(
    AdaptivePackedCellContainer& apc,
    std::uint64_t seed
) noexcept
{
    auto view =
        apc.BuildAViewOverRegion<std::uint64_t>(
            MacroColumnOfAPC::BOTTOM_UP_SLOT
        );

    if (
        !view.has_value() ||
        !view->IsValid() ||
        view->Size() != VIEW_WIDTH
    )
    {
        return false;
    }

    for (
        std::size_t i = 0u;
        i < VIEW_WIDTH;
        ++i
    )
    {
        if (
            view->AtomicLoad(
                i,
                std::memory_order_acquire
            ) !=
            seed +
                static_cast<std::uint64_t>(i)
        )
        {
            return false;
        }
    }

    return true;
}

inline bool ResolveNode(
    RelocationFabric& fabric,
    std::uint32_t slot,
    std::uint32_t generation,
    AdaptivePackedCellContainer& node
) noexcept
{
    APCUseScope use{};

    if (!fabric.ResolveExistingForTest(
        slot,
        node,
        use,
        generation
    ))
    {
        return false;
    }

    use.Release();
    return node.IsActiveAPC();
}

inline bool CheckHParent(
    AdaptivePackedCellContainer& child,
    std::uint32_t expected_parent_slot
) noexcept
{
    AdaptivePackedCellContainer parent =
        child.FindParent(
            FabricSegments::
                VALUE_PARENT_EDGE_TABLE_H,
            0u
        );

    return
        parent.IsActiveAPC() &&
        parent.GetThisSlotIdx() ==
            expected_parent_slot;
}

inline Result Run()
{
    Banner(
        "TEST 8 - FULL-SLAB SAVE / ATTACH / DETACH RELOCATION"
    );

    static constexpr std::uint64_t
        PARENT_SEED = 0xA100u;

    static constexpr std::uint64_t
        CHILD_SEED = 0xB200u;

    RelocationFabric source{};
    TestAPC source_parent{};
    TestAPC source_child{};

    if (
        !source.InitializeFabric(
            SLOT_COUNT,
            MINIMUM_APC_CELL_COUNT,
            RegionConfig(),
            PARENT_CAPACITY
        ) ||
        !CreateAtomicNode(
            source,
            source_parent
        ) ||
        !CreateAtomicNode(
            source,
            source_child
        )
    )
    {
        std::cout
            << "  source construction                         FAIL\n"
            << "\nTEST 8 OVERALL: FAIL\n";
        return Result::FAIL;
    }

    const std::uint32_t parent_slot =
        source_parent.GetThisSlotIdx();

    const std::uint32_t child_slot =
        source_child.GetThisSlotIdx();

    const std::uint32_t parent_generation =
        source_parent.GenerationForTest();

    const std::uint32_t child_generation =
        source_child.GenerationForTest();

    const bool source_setup =
        ADS::IsValid32BitAPCUnit(parent_slot) &&
        ADS::IsValid32BitAPCUnit(child_slot) &&
        HandleOfAPCStatic::
            IsGenerationValid(
                parent_generation
            ) &&
        HandleOfAPCStatic::
            IsGenerationValid(
                child_generation
            ) &&
        source_child.AddParent(
            source_parent,
            FabricSegments::
                VALUE_PARENT_EDGE_TABLE_H
        ) &&
        WritePayload(
            source_parent,
            PARENT_SEED
        ) &&
        WritePayload(
            source_child,
            CHILD_SEED
        );

    if (!source_setup)
    {
        std::cout
            << "  source topology/payload                     FAIL\n"
            << "\nTEST 8 OVERALL: FAIL\n";
        return Result::FAIL;
    }

    const std::uint64_t cell_count =
        source.SlabCellCountForTest();

    std::uint64_t* const source_address =
        source.SlabAddressForTest();

    std::vector<std::uint64_t> snapshot(
        static_cast<std::size_t>(
            cell_count
        )
    );

    const bool save_ok =
        source_address != nullptr &&
        cell_count != UNSIGNED_ZERO &&
        source.SaveFabric(
            std::span<std::uint64_t>(
                snapshot.data(),
                snapshot.size()
            )
        );

    RelocationFabric wrong_count_target{};
    const bool wrong_count_rejected =
        save_ok &&
        cell_count > 1u &&
        !wrong_count_target.AttachFabric(
            snapshot.data(),
            cell_count - 1u,
            CoreOfFabricCoordinator::
                FabricBackigOwnership::
                    BORROWED
        ) &&
        !wrong_count_target.IsFabricActive();

    std::vector<std::uint64_t>
        bad_version_image = snapshot;

    bool bad_version_rejected = false;

    if (
        save_ok &&
        bad_version_image.size() *
            sizeof(std::uint64_t) >=
            sizeof(
                CoreOfFabricCoordinator::
                    FabricCache
            )
    )
    {
        auto* const cache =
            reinterpret_cast<
                CoreOfFabricCoordinator::
                    FabricCache*
            >(
                bad_version_image.data()
            );

        cache->FormateVersion_ =
            static_cast<std::uint32_t>(
                CoreOfFabricCoordinator::
                    FORMAT_VERSION
            ) +
            1u;

        RelocationFabric
            bad_version_target{};

        bad_version_rejected =
            !bad_version_target.AttachFabric(
                bad_version_image.data(),
                static_cast<std::uint64_t>(
                    bad_version_image.size()
                ),
                CoreOfFabricCoordinator::
                    FabricBackigOwnership::
                        BORROWED
            ) &&
            !bad_version_target.IsFabricActive();
    }

    const bool source_resumed =
        save_ok &&
        source.IsFabricActive() &&
        source_parent.IsActiveAPC() &&
        source_child.IsActiveAPC() &&
        CheckPayload(
            source_parent,
            PARENT_SEED
        ) &&
        CheckPayload(
            source_child,
            CHILD_SEED
        ) &&
        CheckHParent(
            source_child,
            parent_slot
        );

    std::vector<std::uint64_t>
        relocated_one = snapshot;

    const bool first_address_changed =
        !relocated_one.empty() &&
        relocated_one.data() !=
            source_address;

    RelocationFabric first_target{};

    const bool first_attach =
        save_ok &&
        first_address_changed &&
        first_target.AttachFabric(
            relocated_one.data(),
            static_cast<std::uint64_t>(
                relocated_one.size()
            ),
            CoreOfFabricCoordinator::
                FabricBackigOwnership::
                    BORROWED
        );

    AdaptivePackedCellContainer
        first_parent{};

    AdaptivePackedCellContainer
        first_child{};

    const bool first_resolve =
        first_attach &&
        ResolveNode(
            first_target,
            parent_slot,
            parent_generation,
            first_parent
        ) &&
        ResolveNode(
            first_target,
            child_slot,
            child_generation,
            first_child
        );

    const bool first_payload =
        first_resolve &&
        CheckPayload(
            first_parent,
            PARENT_SEED
        ) &&
        CheckPayload(
            first_child,
            CHILD_SEED
        );

    const bool first_topology =
        first_resolve &&
        CheckHParent(
            first_child,
            parent_slot
        );

    const CoreOfFabricCoordinator::
        DetachFabric detached =
            first_attach
                ? first_target.DetachFabric()
                : CoreOfFabricCoordinator::
                    DetachFabric{};

    const bool detach_ok =
        static_cast<bool>(detached) &&
        detached.Slab_ ==
            relocated_one.data() &&
        detached.CellCount_ ==
            cell_count &&
        detached.Ownership_ ==
            CoreOfFabricCoordinator::
                FabricBackigOwnership::
                    BORROWED &&
        !first_target.IsFabricActive();

    // Detach intentionally leaves a quiescent image. Copy that image to another
    // allocation to prove a second address can interpret the same Fabric.
    std::vector<std::uint64_t>
        relocated_two{};

    if (detach_ok)
    {
        relocated_two.assign(
            detached.Slab_,
            detached.Slab_ +
                detached.CellCount_
        );
    }

    const bool second_address_changed =
        !relocated_two.empty() &&
        relocated_two.data() !=
            detached.Slab_;

    RelocationFabric second_target{};

    const bool second_attach =
        detach_ok &&
        second_address_changed &&
        second_target.AttachFabric(
            relocated_two.data(),
            static_cast<std::uint64_t>(
                relocated_two.size()
            ),
            CoreOfFabricCoordinator::
                FabricBackigOwnership::
                    BORROWED
        );

    AdaptivePackedCellContainer
        second_parent{};

    AdaptivePackedCellContainer
        second_child{};

    const bool second_resolve =
        second_attach &&
        ResolveNode(
            second_target,
            parent_slot,
            parent_generation,
            second_parent
        ) &&
        ResolveNode(
            second_target,
            child_slot,
            child_generation,
            second_child
        );

    const bool second_payload =
        second_resolve &&
        CheckPayload(
            second_parent,
            PARENT_SEED
        ) &&
        CheckPayload(
            second_child,
            CHILD_SEED
        );

    const bool second_topology =
        second_resolve &&
        CheckHParent(
            second_child,
            parent_slot
        );

    const bool address_independent =
        first_address_changed &&
        second_address_changed &&
        relocated_two.data() !=
            source_address;

    // Leave second_target attached until its destructor. Because the backing is
    // BORROWED and relocated_two was declared before second_target, the backing
    // stays alive for the complete shutdown.
    const bool ok =
        save_ok &&
        source_resumed &&
        first_attach &&
        first_resolve &&
        first_payload &&
        first_topology &&
        detach_ok &&
        second_attach &&
        second_resolve &&
        second_payload &&
        second_topology &&
        address_independent &&
        wrong_count_rejected &&
        bad_version_rejected;

    std::cout
        << "  SaveFabric succeeds and source resumes       "
        << (source_resumed ? "PASS" : "FAIL")
        << '\n'
        << "  first attach uses different slab address     "
        << (
            first_attach &&
            first_address_changed
                ? "PASS"
                : "FAIL"
        )
        << '\n'
        << "  first relocated payload                      "
        << (first_payload ? "PASS" : "FAIL")
        << '\n'
        << "  first relocated H topology                   "
        << (first_topology ? "PASS" : "FAIL")
        << '\n'
        << "  DetachFabric preserves borrowed image        "
        << (detach_ok ? "PASS" : "FAIL")
        << '\n'
        << "  second attach uses another slab address      "
        << (
            second_attach &&
            second_address_changed
                ? "PASS"
                : "FAIL"
        )
        << '\n'
        << "  second relocated payload                     "
        << (second_payload ? "PASS" : "FAIL")
        << '\n'
        << "  second relocated H topology                  "
        << (second_topology ? "PASS" : "FAIL")
        << '\n'
        << "  address-independent slab image               "
        << (address_independent ? "PASS" : "FAIL")
        << '\n'
        << "  wrong supplied cell count rejected           "
        << (wrong_count_rejected ? "PASS" : "FAIL")
        << '\n'
        << "  incompatible format version rejected         "
        << (bad_version_rejected ? "PASS" : "FAIL")
        << '\n'
        << "  source address                               "
        << static_cast<const void*>(
            source_address
        )
        << '\n'
        << "  first relocated address                      "
        << static_cast<const void*>(
            relocated_one.data()
        )
        << '\n'
        << "  second relocated address                     "
        << static_cast<const void*>(
            relocated_two.data()
        )
        << '\n'
        << "\nTEST 8 OVERALL: "
        << (ok ? "PASS" : "FAIL")
        << '\n';

    return ok
        ? Result::PASS
        : Result::FAIL;
}

} // namespace Test08_FabricRelocation


inline int RunAll()
{
    PrintBenchmarkEnvironment();

    const std::array<std::pair<const char*, Result>, 8u> results{{
        {"Test 1 - adapter-free quiescent benchmark", Test01_Baseline::Run()},
        {"Test 2 - same-K contention benchmark", Test02_Contention::Run()},
        {"Test 3 - reader/writer atomicity", Test03_ReaderWriter::Run()},
        {"Test 4 - public mutation API", Test04_PublicMutationAPI::Run()},
        {"Test 5 - combined DAG proof", Test05_CombinedAcyclicity::Run()},
        {"Test 6 - region schema and views", Test06_RegionSchemaAndViews::Run()},
        {"Test 7 - concurrency and retirement", Test07_ConcurrentDAGAndRetirement::Run()},
        {"Test 8 - full-slab relocation", Test08_FabricRelocation::Run()}
    }};

    Banner("SUPERNOVA APC/FABRIC SYSTEMS TEST SUITE SUMMARY");
    std::uint32_t failures = 0u;
    for (const auto& [name, result] : results)
    {
        std::cout
            << "  " << std::left << std::setw(42) << name
            << ResultName(result) << '\n';
        if (result == Result::FAIL) ++failures;
    }
    std::cout
        << "\n  failures: " << failures
        << "\n================================================================================\n";
    return failures == 0u ? 0 : 1;
}

} // namespace APCDAGTests
