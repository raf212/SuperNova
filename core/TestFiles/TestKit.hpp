#pragma once

// APC/Fabric fixed-region-schema + dual-edge DAG test kit (C++20)
//
// Put this file in core/TestFiles and compile a tiny runner:
//
//   #include "TestKit.hpp"
//   int main() { return APCDAGTests::RunAll(); }
//
// Add core/headers to the compiler include path and link the production .cpp files.
// Tests 1-5 and 7 use only public APC/Fabric operations. Test 6 additionally uses
// a read-only derived Fabric probe to verify the protected DEVICE_VIEW_TABLE ABI,
// compact row geometry, Fabric metadata, and MPMC sequence-cell initialization.

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
    bool PointerPresent = false;

    bool IsFound() const noexcept
    {
        return Outcome == ReadOperation::FOUND &&
            PointerPresent &&
            Node != NO_NODE &&
            Locator != NO_LOCATOR;
    }

    bool IsNone() const noexcept
    {
        return Outcome == ReadOperation::NONE &&
            !PointerPresent &&
            Node == NO_NODE &&
            Locator == NO_LOCATOR;
    }

    bool IsRetry() const noexcept
    {
        return Outcome == ReadOperation::RETRY &&
            !PointerPresent &&
            Node == NO_NODE &&
            Locator == NO_LOCATOR;
    }

    bool ContractValid() const noexcept
    {
        return IsFound() || IsNone() || IsRetry();
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
// Public APC/Fabric adapter for the completed DAG API.
// -----------------------------------------------------------------------------

template <std::size_t NodeCount, std::size_t PayloadWords, std::uint8_t ParentCapacity>
class APCFabricBackend
{
public:
    using SD = SchemaDefinition;
    static constexpr std::uint32_t SLOT_WORDS = MINIMUM_APC_CELL_COUNT;
    static constexpr std::uint32_t FABRIC_SLOT_COUNT =
        static_cast<std::uint32_t>(NodeCount + 8u);
    static constexpr std::uint8_t PARENT_CAPACITY = ParentCapacity;

    bool Initialize() noexcept
    {
        Slots_.fill(ADS::APC_INDEX_BOUND_SENTINAL);

        constexpr std::uint32_t matrix_width =
            PayloadWords == 0u ? 1u : static_cast<std::uint32_t>(PayloadWords);

        const SD::FabricRegionConfig region_config{
            static_cast<std::uint16_t>(
                ADS::RegionBit(MacroColumnOfAPC::FEEDFORWARD_MESSAGE) |
                ADS::RegionBit(MacroColumnOfAPC::FEEDBACKWARD_MESSAGE)
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



        SD::RegionSchemaRecord& ff_schema_prop = schemas[static_cast<std::size_t>(MacroColumnOfAPC::FEEDFORWARD_MESSAGE)];
        SD::RegionSchemaRecord& fb_schema = schemas[static_cast<std::size_t>(MacroColumnOfAPC::FEEDBACKWARD_MESSAGE)];
        ff_schema_prop.Region = MacroColumnOfAPC::FEEDFORWARD_MESSAGE;
        ff_schema_prop.Dtype = SD::DataTypeOfMacroColumn::UINT64_T;
        ff_schema_prop.Protocol = SD::SchemaProtocols::PRIVATE_REGION;
        ff_schema_prop.MatrixHeight = 1u;
        ff_schema_prop.MatrixWidth = matrix_width;
        ff_schema_prop.Flags = SD::SchemaFlags::BATCHED_LAST_DIM;

        fb_schema = ff_schema_prop;
        fb_schema.Region = MacroColumnOfAPC::FEEDBACKWARD_MESSAGE;
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
                    MacroColumnOfAPC::FEEDFORWARD_MESSAGE
                );
                auto atomic = Nodes_[i].template BuildAViewOverRegion<std::uint64_t>(
                    MacroColumnOfAPC::FEEDBACKWARD_MESSAGE
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
        return Convert_(Nodes_[child].FindParent(
            EdgeTableForAxis(axis),
            ordinal,
            max_tries
        ));
    }

    ReadResult FindFirstChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t max_tries = 1u) noexcept
    {
        return parent < NodeCount
            ? Convert_(Nodes_[parent].FindFirstChild(EdgeTableForAxis(axis), max_tries))
            : ReadResult{};
    }

    ReadResult FindLastChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t max_tries = 1u) noexcept
    {
        return parent < NodeCount
            ? Convert_(Nodes_[parent].FindLastChild(EdgeTableForAxis(axis), max_tries))
            : ReadResult{};
    }

    ReadResult FindNextChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator,
        std::uint32_t max_tries = 1u) noexcept
    {
        return parent < NodeCount
            ? Convert_(Nodes_[parent].FindNextChild(
                EdgeTableForAxis(axis),
                locator,
                max_tries
            ))
            : ReadResult{};
    }

    ReadResult FindPreviousChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator,
        std::uint32_t max_tries = 1u) noexcept
    {
        return parent < NodeCount
            ? Convert_(Nodes_[parent].FindPreviousChild(
                EdgeTableForAxis(axis),
                locator,
                max_tries
            ))
            : ReadResult{};
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

    VagueTemoraryPremativeFabric Fabric_{};

private:
    std::array<AdaptivePackedCellContainer, NodeCount> Nodes_{};
    std::array<std::uint32_t, NodeCount> Slots_{};
    std::array<RegionView<std::uint64_t>, NodeCount> DirectViews_{};
    std::array<RegionView<std::uint64_t>, NodeCount> AtomicViews_{};

    std::size_t IndexOf_(AdaptivePackedCellContainer* ptr) const noexcept
    {
        if (!ptr)
        {
            return ReadResult::NO_NODE;
        }
        const AdaptivePackedCellContainer* first = Nodes_.data();
        const AdaptivePackedCellContainer* last = first + Nodes_.size();
        return ptr >= first && ptr < last
            ? static_cast<std::size_t>(ptr - first)
            : ReadResult::NO_NODE;
    }

    template <typename Operation>
    ReadResult Convert_(const Operation& operation) const noexcept
    {
        return ReadResult{
            IndexOf_(operation.APC_),
            operation.RelationLocator_,
            operation.MutationOP_,
            operation.Use_
        };
    }
};

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
constexpr std::uint32_t MEASURED_RUNS = 5u;

using VectorBackend = VectorLockedForest<NODE_COUNT, PAYLOAD_WORDS>;
using APCBackend = APCFabricBackend<NODE_COUNT, PAYLOAD_WORDS, PARENT_CAPACITY>;

constexpr std::size_t Reverse6(std::size_t value) noexcept
{
    std::size_t result = 0u;
    for (std::size_t bit = 0u; bit < 6u; ++bit)
    {
        result = (result << 1u) | ((value >> bit) & 1u);
    }
    return result;
}

constexpr std::array<std::size_t, CHAIN_LENGTH> MakeVerticalOrder() noexcept
{
    std::array<std::size_t, CHAIN_LENGTH> order{};
    for (std::size_t i = 0u; i < CHAIN_LENGTH; ++i)
    {
        order[i] = CHAIN_BEGIN + Reverse6(i);
    }
    return order;
}

constexpr auto VERTICAL_ORDER = MakeVerticalOrder();

template <typename Backend>
bool Build(Backend& backend)
{
    if (!backend.Initialize())
    {
        return false;
    }

    for (std::size_t child = CHAIN_BEGIN + 1u; child <= CHAIN_END; ++child)
    {
        if (!backend.AddParent(child - 1u, child, Axis::HORIZONTAL))
        {
            return false;
        }
    }
    for (std::size_t child : VERTICAL_ORDER)
    {
        if (!backend.AddParent(MAIN_V_PARENT, child, Axis::VERTICAL))
        {
            return false;
        }
    }
    if (!backend.AddParent(AUX_V_PARENT, AUX_ANCHOR, Axis::VERTICAL))
    {
        return false;
    }

    for (std::size_t node = 0u; node < NODE_COUNT; ++node)
    {
        for (std::uint32_t word = 0u; word < PAYLOAD_WORDS; ++word)
        {
            const std::uint64_t value =
                (static_cast<std::uint64_t>(node + 1u) << 32u) | word;
            if (
                !backend.StorePayload(node, word, value, false) ||
                !backend.StorePayload(node, word, value, true)
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
            : static_cast<double>(ElapsedNs) / static_cast<double>(Operations);
    }
};

template <typename Backend>
Timing HorizontalForward(Backend& backend)
{
    Timing timing{};
    const auto begin = Clock::now();
    for (std::uint32_t round = 0u; round < TRAVERSAL_ROUNDS; ++round)
    {
        for (std::size_t parent = CHAIN_BEGIN; parent < CHAIN_END; ++parent)
        {
            const ReadResult read = backend.FindFirstChild(parent, Axis::HORIZONTAL);
            timing.Reads.Observe(read);
            if (!read.IsFound() || read.Node != parent + 1u)
            {
                return {};
            }
            timing.Checksum += read.Node + 1u;
        }
    }
    timing.ElapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - begin
    ).count();
    timing.Operations = static_cast<std::uint64_t>(TRAVERSAL_ROUNDS) *
        (CHAIN_LENGTH - 1u);
    timing.Ok = true;
    return timing;
}

template <typename Backend>
Timing HorizontalBackward(Backend& backend)
{
    Timing timing{};
    const auto begin = Clock::now();
    for (std::uint32_t round = 0u; round < TRAVERSAL_ROUNDS; ++round)
    {
        for (std::size_t child = CHAIN_END; child > CHAIN_BEGIN; --child)
        {
            const ReadResult read = backend.FindParent(child, Axis::HORIZONTAL, 0u);
            timing.Reads.Observe(read);
            if (!read.IsFound() || read.Node != child - 1u)
            {
                return {};
            }
            timing.Checksum += read.Node + 1u;
        }
    }
    timing.ElapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - begin
    ).count();
    timing.Operations = static_cast<std::uint64_t>(TRAVERSAL_ROUNDS) *
        (CHAIN_LENGTH - 1u);
    timing.Ok = true;
    return timing;
}

template <typename Backend>
Timing VerticalForward(Backend& backend, bool with_payload)
{
    Timing timing{};
    const std::uint32_t rounds = with_payload
        ? GRAPH_PAYLOAD_ROUNDS
        : TRAVERSAL_ROUNDS;
    const auto begin = Clock::now();

    for (std::uint32_t round = 0u; round < rounds; ++round)
    {
        ReadResult read = backend.FindFirstChild(MAIN_V_PARENT, Axis::VERTICAL);
        timing.Reads.Observe(read);
        for (std::size_t i = 0u; i < CHAIN_LENGTH; ++i)
        {
            if (!read.IsFound() || read.Node != VERTICAL_ORDER[i])
            {
                return {};
            }

            if (with_payload)
            {
                std::uint64_t value = 0u;
                if (!backend.LoadPayload(
                    read.Node,
                    static_cast<std::uint32_t>(i % PAYLOAD_WORDS),
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
                timing.Checksum += read.Node + 1u;
            }

            const std::uint32_t cursor = read.Locator;
            read = backend.FindNextChild(MAIN_V_PARENT, Axis::VERTICAL, cursor);
            timing.Reads.Observe(read);
        }
        if (!read.IsNone())
        {
            return {};
        }
    }

    timing.ElapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - begin
    ).count();
    timing.Operations = static_cast<std::uint64_t>(rounds) * CHAIN_LENGTH;
    timing.Ok = true;
    return timing;
}

template <typename Backend>
Timing VerticalBackward(Backend& backend)
{
    Timing timing{};
    const auto begin = Clock::now();
    for (std::uint32_t round = 0u; round < TRAVERSAL_ROUNDS; ++round)
    {
        ReadResult read = backend.FindLastChild(MAIN_V_PARENT, Axis::VERTICAL);
        timing.Reads.Observe(read);
        for (std::size_t i = CHAIN_LENGTH; i-- > 0u;)
        {
            if (!read.IsFound() || read.Node != VERTICAL_ORDER[i])
            {
                return {};
            }
            timing.Checksum += read.Node + 1u;
            const std::uint32_t cursor = read.Locator;
            read = backend.FindPreviousChild(MAIN_V_PARENT, Axis::VERTICAL, cursor);
            timing.Reads.Observe(read);
        }
        if (!read.IsNone())
        {
            return {};
        }
    }
    timing.ElapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - begin
    ).count();
    timing.Operations = static_cast<std::uint64_t>(TRAVERSAL_ROUNDS) * CHAIN_LENGTH;
    timing.Ok = true;
    return timing;
}

template <typename Backend>
Timing PayloadRead(Backend& backend, bool atomic)
{
    Timing timing{};
    const auto begin = Clock::now();
    for (std::uint32_t round = 0u; round < PAYLOAD_ROUNDS; ++round)
    {
        for (std::size_t node = CHAIN_BEGIN; node <= CHAIN_END; ++node)
        {
            for (std::uint32_t word = 0u; word < PAYLOAD_WORDS; ++word)
            {
                std::uint64_t value = 0u;
                if (!backend.LoadPayload(node, word, value, atomic))
                {
                    return {};
                }
                timing.Checksum += value;
            }
        }
    }
    timing.ElapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - begin
    ).count();
    timing.Operations = static_cast<std::uint64_t>(PAYLOAD_ROUNDS) *
        CHAIN_LENGTH * PAYLOAD_WORDS;
    timing.Ok = true;
    return timing;
}

template <typename Backend>
Timing ParentReplacement(Backend& backend, Axis axis)
{
    const std::size_t child = CHAIN_END;
    const std::size_t parent_a = axis == Axis::HORIZONTAL
        ? CHAIN_END - 1u
        : MAIN_V_PARENT;
    const std::size_t parent_b = axis == Axis::HORIZONTAL
        ? CHAIN_END - 2u
        : AUX_V_PARENT;

    Timing timing{};
    std::size_t current = parent_a;
    const auto begin = Clock::now();
    for (std::uint32_t i = 0u; i < MUTATION_ROUNDS; ++i)
    {
        const std::size_t next = current == parent_a ? parent_b : parent_a;
        if (!backend.ReplaceParent(current, next, child, axis))
        {
            return {};
        }
        current = next;
        ++timing.Operations;
    }
    timing.ElapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - begin
    ).count();
    timing.Checksum = timing.Operations;
    timing.Ok = current == parent_a;
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

constexpr std::size_t METRIC_COUNT = static_cast<std::size_t>(Metric::COUNT);

constexpr const char* MetricName(Metric metric) noexcept
{
    switch (metric)
    {
    case Metric::H_FORWARD: return "H forward sequential";
    case Metric::H_BACKWARD: return "H backward sequential";
    case Metric::V_FORWARD: return "V forward scrambled";
    case Metric::V_BACKWARD: return "V backward scrambled";
    case Metric::PAYLOAD_DIRECT: return "payload direct read";
    case Metric::PAYLOAD_ATOMIC: return "payload atomic read";
    case Metric::GRAPH_PAYLOAD: return "scrambled graph+payload";
    case Metric::REPLACE_H_PARENT: return "atomic H parent replace";
    case Metric::REPLACE_V_PARENT: return "atomic V parent replace";
    default: return "unknown";
    }
}

template <typename Backend>
Timing RunMetric(Backend& backend, Metric metric)
{
    switch (metric)
    {
    case Metric::H_FORWARD: return HorizontalForward(backend);
    case Metric::H_BACKWARD: return HorizontalBackward(backend);
    case Metric::V_FORWARD: return VerticalForward(backend, false);
    case Metric::V_BACKWARD: return VerticalBackward(backend);
    case Metric::PAYLOAD_DIRECT: return PayloadRead(backend, false);
    case Metric::PAYLOAD_ATOMIC: return PayloadRead(backend, true);
    case Metric::GRAPH_PAYLOAD: return VerticalForward(backend, true);
    case Metric::REPLACE_H_PARENT: return ParentReplacement(backend, Axis::HORIZONTAL);
    case Metric::REPLACE_V_PARENT: return ParentReplacement(backend, Axis::VERTICAL);
    default: return {};
    }
}

inline Result Run()
{
    Banner("TEST 1 - DAG TRAVERSAL / PAYLOAD / ATOMIC-PARENT-REPLACE BASELINE");

    const auto vector_build_begin = Clock::now();
    VectorBackend vector_backend{};
    if (!Build(vector_backend)) return Result::FAIL;
    const auto vector_build_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - vector_build_begin
    ).count();

    const auto apc_build_begin = Clock::now();
    APCBackend apc_backend{};
    if (!Build(apc_backend)) return Result::FAIL;
    const auto apc_build_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - apc_build_begin
    ).count();

    const GraphProof initial_proof =
        ProveQuiescentCombinedDAG<NODE_COUNT, PARENT_CAPACITY>(apc_backend);
    if (!initial_proof.Passed())
    {
        return Result::FAIL;
    }

    std::array<std::array<double, MEASURED_RUNS>, METRIC_COUNT> vector_samples{};
    std::array<std::array<double, MEASURED_RUNS>, METRIC_COUNT> apc_samples{};
    ReadCounts vector_reads{};
    ReadCounts apc_reads{};

    for (std::uint32_t run = 0u; run < MEASURED_RUNS; ++run)
    {
        for (std::size_t index = 0u; index < METRIC_COUNT; ++index)
        {
            const Metric metric = static_cast<Metric>(index);
            Timing vector_timing{};
            Timing apc_timing{};

            if ((run & 1u) == 0u)
            {
                vector_timing = RunMetric(vector_backend, metric);
                apc_timing = RunMetric(apc_backend, metric);
            }
            else
            {
                apc_timing = RunMetric(apc_backend, metric);
                vector_timing = RunMetric(vector_backend, metric);
            }

            if (
                !vector_timing.Ok ||
                !apc_timing.Ok ||
                vector_timing.Checksum != apc_timing.Checksum
            )
            {
                std::cout << "  failed metric: " << MetricName(metric) << '\n';
                return Result::FAIL;
            }

            vector_samples[index][run] = vector_timing.NsPerOperation();
            apc_samples[index][run] = apc_timing.NsPerOperation();
            vector_reads.Add(vector_timing.Reads);
            apc_reads.Add(apc_timing.Reads);
        }
    }

    const GraphProof final_proof =
        ProveQuiescentCombinedDAG<NODE_COUNT, PARENT_CAPACITY>(apc_backend);

    std::cout
        << "Correctness: explicit parent reads, cursor child traversal, and H-union-V proof: "
        << (final_proof.Passed() ? "PASS" : "FAIL") << "\n\n"
        << "CONSTRUCTION\n"
        << "  vector+locked forest: " << vector_build_ns / 1000 << " us\n"
        << "  APC+Fabric DAG       : " << apc_build_ns / 1000 << " us\n\n"
        << "MEDIAN COST PER OPERATION\n";

    for (std::size_t index = 0u; index < METRIC_COUNT; ++index)
    {
        const double vector_ns = Median(vector_samples[index]);
        const double apc_ns = Median(apc_samples[index]);
        std::cout
            << std::left << std::setw(29) << MetricName(static_cast<Metric>(index))
            << " vector-base=" << std::right << std::setw(10)
            << std::fixed << std::setprecision(2) << vector_ns << " ns/op"
            << "  APC=" << std::setw(10) << apc_ns << " ns/op"
            << "  ratio=" << std::setw(8) << Ratio(apc_ns, vector_ns) << "x\n";
    }

    std::cout << "\nPUBLIC READ OUTCOMES\n";
    PrintReadCounts("vector locked forest", vector_reads);
    PrintReadCounts("APC/Fabric DAG", apc_reads);

    const bool ok = final_proof.Passed() &&
        vector_reads.BadContract == 0u &&
        apc_reads.BadContract == 0u &&
        apc_reads.Retry == 0u;

    std::cout << "\nTEST 1 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? Result::PASS : Result::FAIL;
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

template <typename Backend>
bool Build(Backend& backend, std::size_t workers)
{
    if (!backend.Initialize()) return false;
    for (std::size_t i = 0u; i < workers; ++i)
    {
        const std::size_t child = FIRST_CHILD + i;
        if (
            !backend.AddParent(0u, child, Axis::HORIZONTAL) ||
            !backend.AddParent(2u, child, Axis::VERTICAL)
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
SweepResult RunWorkers(Backend& backend, std::size_t workers)
{
    std::barrier start(static_cast<std::ptrdiff_t>(workers + 1u));
    std::atomic<bool> failed{false};
    std::atomic<std::uint64_t> success{0u};
    std::atomic<std::uint64_t> retries{0u};
    std::vector<std::thread> threads;
    threads.reserve(workers);

    for (std::size_t worker = 0u; worker < workers; ++worker)
    {
        threads.emplace_back([&, worker]() noexcept
        {
            const std::size_t child = FIRST_CHILD + worker;
            std::size_t h_current = 0u;
            std::size_t v_current = 2u;
            std::uint64_t local_retries = 0u;
            std::uint64_t local_success = 0u;
            start.arrive_and_wait();

            for (std::uint32_t i = 0u; i < OPS_PER_THREAD; ++i)
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
                local_success += 2u;
            }

            success.fetch_add(local_success, std::memory_order_relaxed);
            retries.fetch_add(local_retries, std::memory_order_relaxed);
        });
    }

    const auto begin = Clock::now();
    start.arrive_and_wait();
    for (std::thread& thread : threads) thread.join();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - begin
    ).count();

    const std::uint64_t completed = success.load(std::memory_order_acquire);
    return {
        !failed.load(std::memory_order_acquire) &&
            completed == workers * OPS_PER_THREAD * 2u,
        completed == 0u ? 0.0 : static_cast<double>(elapsed) / completed,
        completed,
        retries.load(std::memory_order_acquire)
    };
}

inline Result Run()
{
    Banner("TEST 2 - GLOBAL-MUTEX VECTOR FOREST vs APC/FABRIC DAG CONTENTION");
    constexpr std::array<std::size_t, 4u> WORKERS{1u, 2u, 4u, 8u};
    bool all_ok = true;

    std::cout
        << "Each worker owns one child; all workers contend on the same two H and two V parents.\n"
        << "APC retries one-attempt transactions at workload level; vector serializes each move.\n\n";

    for (std::size_t workers : WORKERS)
    {
        VectorLockedForest<NODE_COUNT, 1u> vector_backend{};
        APCFabricBackend<NODE_COUNT, 1u, K> apc_backend{};
        if (!Build(vector_backend, workers) || !Build(apc_backend, workers))
        {
            return Result::FAIL;
        }

        const SweepResult vector_result = RunWorkers(vector_backend, workers);
        const SweepResult apc_result = RunWorkers(apc_backend, workers);
        const GraphProof proof = ProveQuiescentCombinedDAG<NODE_COUNT, K>(apc_backend);
        const bool row_ok = vector_result.Ok && apc_result.Ok && proof.Passed();
        all_ok = all_ok && row_ok;

        std::cout
            << "  threads=" << std::setw(2) << workers
            << " vector=" << std::setw(10) << std::fixed << std::setprecision(2)
            << vector_result.NsPerSuccess << " ns/op"
            << " APC=" << std::setw(10) << apc_result.NsPerSuccess << " ns/op"
            << " APC-retries=" << std::setw(10) << apc_result.Retries
            << " integrity=" << (row_ok ? "PASS" : "FAIL") << '\n';
    }

    std::cout << "\nTEST 2 OVERALL: " << (all_ok ? "PASS" : "FAIL") << '\n';
    return all_ok ? Result::PASS : Result::FAIL;
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
        ADS::RegionBit(MacroColumnOfAPC::FEEDFORWARD_MESSAGE),
        0u,
        batch_capacity
    };
}

class InspectableFabric final : public VagueTemoraryPremativeFabric
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
            range.BeginIndex + local_cell >= SlabCellCount_
        )
        {
            return std::nullopt;
        }

        return std::atomic_ref<const std::uint64_t>(
            SlabBasePtr_[range.BeginIndex + local_cell]
        ).load(std::memory_order_acquire);
    }

    std::optional<std::uint64_t> FabricMeta(
        CoreOfFabricCoordinator::FabricMetaIndicies index
    ) noexcept
    {
        if (!SlabBasePtr_)
        {
            return std::nullopt;
        }
        return std::atomic_ref<const std::uint64_t>(
            SlabBasePtr_[static_cast<std::size_t>(index)]
        ).load(std::memory_order_acquire);
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
        return MatrixViewTableBeginIndex_;
    }

    std::uint16_t ActiveMask() const noexcept { return ActiveRegionMask_; }
    std::uint8_t ActiveCount() const noexcept { return ActiveRegionCount_; }
    std::uint16_t ViewRowCells() const noexcept { return MatrixViewRowCellCount_; }
    std::uint32_t BatchCapacity() const noexcept { return MatrixBatchCapacity_; }
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
    ordinary.Region = MacroColumnOfAPC::FEEDFORWARD_MESSAGE;
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
        ADS::RegionBit(MacroColumnOfAPC::FEEDFORWARD_MESSAGE) |
        ADS::RegionBit(MacroColumnOfAPC::STATE_SLOT) |
        ADS::RegionBit(MacroColumnOfAPC::WEIGHT_SLOT) |
        ADS::RegionBit(MacroColumnOfAPC::HETEROGENOUS_PTR);

    const bool compact_ok =
        ADS::CompactRegionIndex(
            sparse_mask, MacroColumnOfAPC::FEEDFORWARD_MESSAGE
        ) == 0u &&
        ADS::CompactRegionIndex(
            sparse_mask, MacroColumnOfAPC::STATE_SLOT
        ) == 1u &&
        ADS::CompactRegionIndex(
            sparse_mask, MacroColumnOfAPC::WEIGHT_SLOT
        ) == 2u &&
        ADS::CompactRegionIndex(
            sparse_mask, MacroColumnOfAPC::HETEROGENOUS_PTR
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
        VagueTemoraryPremativeFabric fabric{};
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
        ADS::RegionBit(MacroColumnOfAPC::FEEDFORWARD_MESSAGE),
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
    VagueTemoraryPremativeFabric fabric{};
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE,
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE,
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE,
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
    VagueTemoraryPremativeFabric& fabric,
    AdaptivePackedCellContainer& apc,
    SD::SchemaProtocols region_protocol) noexcept
{
    constexpr auto dtype_value = SD::CppTypeToRegionDType<T>();
    static_assert(dtype_value.has_value());

    SD::RegionSchemaTable schemas{};
    SD::MakeDisabledSchemaTable(schemas);
    return MakeSchema(
        schemas,
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE,
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
    VagueTemoraryPremativeFabric fabric{};
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

    auto view = apc.BuildAViewOverRegion<T>(MacroColumnOfAPC::FEEDFORWARD_MESSAGE);
    auto wrong = apc.BuildAViewOverRegion<WrongType<T>>(MacroColumnOfAPC::FEEDFORWARD_MESSAGE);
    if (
        !view.has_value() || !view->IsValid() || view->Size() != VIEW_WIDTH ||
        view->GetProtocol() != SD::SchemaProtocols::PRIVATE_REGION ||
        wrong.has_value() || view->RawMutableSpan().has_value() == false ||
        view->AtomicStore(0u, FirstValue<T>()) ||
        view->AtomicStore(view->Size(), FirstValue<T>()) ||
        apc.BuildAViewOverRegion<T>(
            MacroColumnOfAPC::FEEDFORWARD_MESSAGE, 1u
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
    if (!apc.ZeroARegion<T>(MacroColumnOfAPC::FEEDFORWARD_MESSAGE)) return false;
    return std::all_of(span->begin(), span->end(), [](T value) { return value == T{}; });
}

template <typename T>
bool AtomicCase() noexcept
{
    VagueTemoraryPremativeFabric fabric{};
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

    auto view = apc.BuildAViewOverRegion<T>(MacroColumnOfAPC::FEEDFORWARD_MESSAGE);
    auto wrong = apc.BuildAViewOverRegion<WrongType<T>>(MacroColumnOfAPC::FEEDFORWARD_MESSAGE);
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
        !apc.ZeroARegion<T>(MacroColumnOfAPC::FEEDFORWARD_MESSAGE)
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
    VagueTemoraryPremativeFabric fabric{};
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE
    );
    return
        view.has_value() &&
        view->IsValid() &&
        view->Size() == VIEW_WIDTH &&
        view->GetProtocol() == SD::SchemaProtocols::IMMUTABLE_SNAPSHOT &&
        !view->RawMutableSpan().has_value() &&
        !view->AtomicStore(0u, FirstValue<T>()) &&
        !apc.ZeroARegion<T>(MacroColumnOfAPC::FEEDFORWARD_MESSAGE);
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
        ADS::RegionBit(MacroColumnOfAPC::FEEDFORWARD_MESSAGE) |
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

    using FMI = CoreOfFabricCoordinator::FabricMetaIndicies;
    const auto matrix_range = fabric.TableRange(FabricSegments::MATRIX_VIEW_TABLE);
    const bool fabric_metadata_ok =
        fabric.ActiveMask() == active_mask &&
        fabric.ActiveCount() == active_count &&
        fabric.ViewRowCells() == row_cells &&
        fabric.BatchCapacity() == batch &&
        matrix_range.has_value() &&
        matrix_range->EndIndex - matrix_range->BeginIndex ==
            static_cast<std::size_t>(slot_count) * row_cells &&
        fabric.MatrixViewBegin() == matrix_range->BeginIndex &&
        fabric.FabricMeta(FMI::ACTIVE_REGION_MASK) == active_mask &&
        fabric.FabricMeta(FMI::ACTIVE_REGION_COUNT) == active_count &&
        fabric.FabricMeta(FMI::REGION_SCHEMA_RECORD_CELL_COUNT) ==
            SD::RegionSchemaCellCount() &&
        fabric.FabricMeta(FMI::DEVICE_VIEW_ROW_CELL_COUNT) == row_cells &&
        fabric.FabricMeta(FMI::MATRIC_BATCH_CAPACITY) == batch &&
        fabric.FabricMeta(FMI::REGION_ALLIGNMENT_CELL_COUNT) ==
            SD::REGION_ALIGNMENT_CELLS;

    SD::RegionSchemaTable schemas{};
    SD::MakeDisabledSchemaTable(schemas);
    if (
        !MakeSchema(
            schemas,
            MacroColumnOfAPC::FEEDFORWARD_MESSAGE,
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE,
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE,
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE
    );
    auto state = apc.BuildAViewOverRegion<float>(MacroColumnOfAPC::STATE_SLOT);
    auto weight = apc.BuildAViewOverRegion<float>(MacroColumnOfAPC::WEIGHT_SLOT);
    auto queue_view = apc.BuildAViewOverRegion<float>(MacroColumnOfAPC::ERROR_SLOT);
    auto double_view = apc.BuildAViewOverRegion<std::uint64_t>(
        MacroColumnOfAPC::AUX_SLOT
    );
    auto wrong_type = apc.BuildAViewOverRegion<double>(
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE
    );
    auto inactive = apc.BuildAViewOverRegion<float>(
        MacroColumnOfAPC::HETEROGENOUS_PTR
    );
    auto second_feedforward = second_apc.BuildAViewOverRegion<float>(
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE,
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
            MacroColumnOfAPC::FEEDFORWARD_MESSAGE
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE,
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE
    );
    auto stale_dtype = replacement.BuildAViewOverRegion<std::uint64_t>(
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE
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
        ADS::RegionBit(MacroColumnOfAPC::FEEDFORWARD_MESSAGE),
        0u,
        8u
    };
}

inline bool CreateAtomic(
    VagueTemoraryPremativeFabric& fabric,
    AdaptivePackedCellContainer& apc) noexcept
{
    SchemaDefinition::RegionSchemaTable schemas{};
    SchemaDefinition::MakeDisabledSchemaTable(schemas);

    SchemaDefinition::RegionSchemaRecord& schema = schemas[static_cast<std::size_t>(
            MacroColumnOfAPC::FEEDFORWARD_MESSAGE
    )];
    schema.Region = MacroColumnOfAPC::FEEDFORWARD_MESSAGE;
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
    VagueTemoraryPremativeFabric fabric{};
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
        MacroColumnOfAPC::FEEDFORWARD_MESSAGE
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
            MacroColumnOfAPC::FEEDFORWARD_MESSAGE
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

inline Result Run()
{
    Banner("TEST 7 - CONCURRENT H/V DAG MUTATION + RETIREMENT / ABA");
    std::uint64_t mixed_retries = 0u;
    const bool race_ok = FixedOrderRace();
    const bool stress_ok = MixedAxisStress(mixed_retries);
    const bool retirement_ok = RetirementAndABA();
    const bool ok = race_ok && stress_ok && retirement_ok;

    std::cout
        << "  A--H-->B raced with illegal B--V-->A : " << (race_ok ? "PASS" : "FAIL") << '\n'
        << "  shared-parent mixed H/V stress       : " << (stress_ok ? "PASS" : "FAIL") << '\n'
        << "  transaction retries observed         : " << mixed_retries << '\n'
        << "  linked/pinned retirement + ABA reuse : " << (retirement_ok ? "PASS" : "FAIL") << '\n'
        << "\nTEST 7 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';

    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test07_ConcurrentDAGAndRetirement

inline int RunAll()
{
    const std::array<std::pair<const char*, Result>, 7u> results{{
        {"Test 1 - baseline and benchmark", Test01_Baseline::Run()},
        {"Test 2 - contention sweep", Test02_Contention::Run()},
        {"Test 3 - reader/writer atomicity", Test03_ReaderWriter::Run()},
        {"Test 4 - public mutation API", Test04_PublicMutationAPI::Run()},
        {"Test 5 - combined DAG proof", Test05_CombinedAcyclicity::Run()},
        {"Test 6 - region schema and views", Test06_RegionSchemaAndViews::Run()},
        {"Test 7 - concurrency and retirement", Test07_ConcurrentDAGAndRetirement::Run()}
    }};

    Banner("APC DUAL-EDGE DAG TEST SUITE SUMMARY");
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
