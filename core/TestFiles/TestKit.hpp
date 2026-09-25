#pragma once

// SuperNova APC/Fabric paper-quality systems test kit (C++20)
//
// The benchmark sections deliberately distinguish:
//   1) a compact fixed-capacity bidirectional vector DAG for Test 1,
//   2) a feature-matched row-local-lock vector DAG for Test 2,
//   3) the same row-local DAG with shared-reader/exclusive-writer locks for Test 3,
//   4) APC/Fabric with public generation/transaction/read contracts.
//
// Tests 1-3 share one runtime (N,K) case matrix and reusable backend adapters.
// Test 1 uses one 128 x 8-byte payload region per node and minimum valid Fabric
// geometry. Tests 2A/2B sweep every writer count through usable_threads-2; Tests
// 3A/3B keep exactly two active writers and sweep every reader count through the
// remaining usable threads. Baselines do not emulate Fabric relocation, generation
// handles, retirement, or schema semantics; those properties are tested separately.
//
// Put this file in core/TestFiles and compile a tiny runner:
//
//   #include "TestKit.hpp"
//   int main() { return APCDAGTests::Run(); }
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
#include <shared_mutex>
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

namespace ConcurrencyConfig
{
constexpr std::size_t BENCHMARK_PAYLOAD_WORDS = 1u;
constexpr std::size_t READER_WRITER_COUNT = 2u;
constexpr std::uint32_t MUTATIONS_PER_WRITER = 2'000u;
constexpr std::uint32_t STABLE_READS_PER_READER = 20'000u;
constexpr std::uint32_t WRITER_WARMUP_MUTATIONS = 256u;
constexpr std::size_t MEASURED_RUNS = 4u;
constexpr std::uint32_t TRANSACTION_ATTEMPT_LIMIT = 100'000u;

// Stable readers use an explicit TestKit-local budget rather than inheriting
// DEFAULT_MAX_TRIES from the production API. A bounded optimistic read can
// legitimately observe RETRY while a writer owns the row; exhausting one
// budget is therefore a progress/starvation event, not by itself corruption.
constexpr std::uint32_t STABLE_READ_ATTEMPT_LIMIT =
    TRANSACTION_ATTEMPT_LIMIT;
constexpr std::uint32_t STABLE_READ_STARVATION_ROUND_LIMIT = 16u;

constexpr std::uint64_t OPERATIONS_PER_MUTATION_STEP = 2u;
constexpr double MILLION_OPERATIONS_PER_SECOND_FROM_NS = 1000.0;
constexpr std::uint64_t RANDOM_SEED = 0x9E3779B97F4A7C15ull;
constexpr std::uint64_t RANDOM_STREAM_STEP = 0xD1B54A32D192ED03ull;
constexpr std::uint32_t RANDOM_LEFT_SHIFT_A = 13u;
constexpr std::uint32_t RANDOM_RIGHT_SHIFT = 7u;
constexpr std::uint32_t RANDOM_LEFT_SHIFT_B = 17u;
constexpr std::size_t DISTRIBUTED_PARENT_LIMIT = 100u;
} // namespace ConcurrencyConfig

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
    if constexpr ((N & 1u) != 0u)
    {
        return samples[N / 2u];
    }
    else
    {
        return (samples[N / 2u - 1u] + samples[N / 2u]) * 0.5;
    }
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
// Reusable runtime vector-DAG storage used by Tests 1-3.
//
// One compact relation representation is shared by:
//   * CompactVectorDAG       : one global mutation mutex, raw quiescent reads.
//   * RowLockedVectorDAG     : one row lock per node/axis; Test 3 optionally
//                              takes shared read ownership.
//
// Child identity is derived from relation-locator / K and Parent==NIL denotes
// vacancy, so the baseline stores no redundant child or occupancy field.
// -----------------------------------------------------------------------------

class RuntimeVectorDAGStorage
{
public:
    static constexpr std::uint32_t NIL = UINT32_MAX;

    bool Initialize(
        std::size_t node_count,
        std::size_t payload_words,
        std::uint8_t parent_capacity)
    {
        if (
            node_count == 0u ||
            node_count > UINT32_MAX ||
            parent_capacity == 0u ||
            parent_capacity > ADS::COMPILED_MAX_DIRECT_PARENTS_PER_AXIS ||
            payload_words > UINT32_MAX ||
            node_count > UINT32_MAX / static_cast<std::size_t>(parent_capacity)
        )
        {
            return false;
        }

        NodeCount_ = node_count;
        PayloadWords_ = payload_words;
        ParentCapacity_ = parent_capacity;
        RelationCount_ = node_count * static_cast<std::size_t>(parent_capacity);

        Nodes_.assign(NodeCount_, Node{});
        HRelations_.assign(RelationCount_, Relation{});
        VRelations_.assign(RelationCount_, Relation{});
        Payload_.assign(NodeCount_ * PayloadWords_, 0u);
        return true;
    }

    std::size_t NodeCount() const noexcept { return NodeCount_; }
    std::size_t PayloadWords() const noexcept { return PayloadWords_; }
    std::uint8_t ParentCapacity() const noexcept { return ParentCapacity_; }

    bool AddUnlocked(std::size_t parent, std::size_t child, Axis axis) noexcept
    {
        if (
            parent >= NodeCount_ || child >= NodeCount_ || parent >= child ||
            FindRelation_(child, parent, axis) != NIL
        )
        {
            return false;
        }

        const std::uint32_t locator = FindVacancy_(child, axis);
        if (locator == NIL)
        {
            return false;
        }

        Relation& relation = Relations_(axis)[locator];
        AxisState& parent_axis = Axis_(parent, axis);
        relation.Parent = static_cast<std::uint32_t>(parent);
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
        return true;
    }

    bool RemoveUnlocked(std::size_t parent, std::size_t child, Axis axis) noexcept
    {
        const std::uint32_t locator = FindRelation_(child, parent, axis);
        if (locator == NIL)
        {
            return false;
        }

        Unlink_(locator, axis);
        Relations_(axis)[locator] = Relation{};
        return true;
    }

    bool ReplaceUnlocked(
        std::size_t old_parent,
        std::size_t new_parent,
        std::size_t child,
        Axis axis) noexcept
    {
        if (
            old_parent >= NodeCount_ || new_parent >= NodeCount_ ||
            child >= NodeCount_ || old_parent == new_parent || new_parent >= child
        )
        {
            return false;
        }

        const std::uint32_t locator = FindRelation_(child, old_parent, axis);
        if (locator == NIL || FindRelation_(child, new_parent, axis) != NIL)
        {
            return false;
        }

        Relation& relation = Relations_(axis)[locator];
        Unlink_(locator, axis);

        AxisState& new_parent_axis = Axis_(new_parent, axis);
        relation.Parent = static_cast<std::uint32_t>(new_parent);
        relation.Previous = new_parent_axis.LastChildRelation;
        relation.Next = NIL;

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
        std::uint8_t ordinal) const noexcept
    {
        if (child >= NodeCount_ || ordinal >= ParentCapacity_)
        {
            return {};
        }

        const std::uint32_t locator = Locator_(child, ordinal);
        const Relation& relation = Relations_(axis)[locator];
        return relation.Parent == NIL
            ? ReadResult{}
            : ReadResult{relation.Parent, locator, ReadOperation::FOUND, true};
    }

    ReadResult FindFirstChild(std::size_t parent, Axis axis) const noexcept
    {
        return parent < NodeCount_
            ? ChildResult_(Axis_(parent, axis).FirstChildRelation, axis)
            : ReadResult{};
    }

    ReadResult FindLastChild(std::size_t parent, Axis axis) const noexcept
    {
        return parent < NodeCount_
            ? ChildResult_(Axis_(parent, axis).LastChildRelation, axis)
            : ReadResult{};
    }

    ReadResult FindNextChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator) const noexcept
    {
        if (parent >= NodeCount_ || locator >= RelationCount_)
        {
            return {};
        }

        const Relation& relation = Relations_(axis)[locator];
        return relation.Parent == parent
            ? ChildResult_(relation.Next, axis)
            : ReadResult{};
    }

    ReadResult FindPreviousChild(
        std::size_t parent,
        Axis axis,
        std::uint32_t locator) const noexcept
    {
        if (parent >= NodeCount_ || locator >= RelationCount_)
        {
            return {};
        }

        const Relation& relation = Relations_(axis)[locator];
        return relation.Parent == parent
            ? ChildResult_(relation.Previous, axis)
            : ReadResult{};
    }

    bool StorePayload(
        std::size_t node,
        std::uint32_t word,
        std::uint64_t value,
        bool atomic) noexcept
    {
        if (node >= NodeCount_ || word >= PayloadWords_)
        {
            return false;
        }

        std::uint64_t& destination = Payload_[node * PayloadWords_ + word];
        if (atomic)
        {
            std::atomic_ref<std::uint64_t>(destination).store(
                value, std::memory_order_release);
        }
        else
        {
            destination = value;
        }
        return true;
    }

    bool LoadPayload(
        std::size_t node,
        std::uint32_t word,
        std::uint64_t& value,
        bool atomic) noexcept
    {
        if (node >= NodeCount_ || word >= PayloadWords_)
        {
            return false;
        }

        std::uint64_t& source = Payload_[node * PayloadWords_ + word];
        value = atomic
            ? std::atomic_ref<std::uint64_t>(source).load(std::memory_order_acquire)
            : source;
        return true;
    }

    std::size_t ApproxStorageBytes() const noexcept
    {
        return
            Nodes_.size() * sizeof(Node) +
            HRelations_.size() * sizeof(Relation) +
            VRelations_.size() * sizeof(Relation) +
            Payload_.size() * sizeof(std::uint64_t);
    }

private:
    struct AxisState
    {
        std::uint32_t FirstChildRelation = NIL;
        std::uint32_t LastChildRelation = NIL;
    };

    struct Node
    {
        AxisState H{};
        AxisState V{};
    };

    struct Relation
    {
        std::uint32_t Parent = NIL;
        std::uint32_t Previous = NIL;
        std::uint32_t Next = NIL;
    };

    std::size_t NodeCount_ = 0u;
    std::size_t PayloadWords_ = 0u;
    std::size_t RelationCount_ = 0u;
    std::uint8_t ParentCapacity_ = 0u;
    std::vector<Node> Nodes_{};
    std::vector<Relation> HRelations_{};
    std::vector<Relation> VRelations_{};
    std::vector<std::uint64_t> Payload_{};

    std::uint32_t Locator_(std::size_t child, std::uint8_t ordinal) const noexcept
    {
        return static_cast<std::uint32_t>(
            child * static_cast<std::size_t>(ParentCapacity_) + ordinal);
    }

    std::size_t Child_(std::uint32_t locator) const noexcept
    {
        return static_cast<std::size_t>(locator) /
            static_cast<std::size_t>(ParentCapacity_);
    }

    AxisState& Axis_(std::size_t node, Axis axis) noexcept
    {
        return axis == Axis::HORIZONTAL ? Nodes_[node].H : Nodes_[node].V;
    }

    const AxisState& Axis_(std::size_t node, Axis axis) const noexcept
    {
        return axis == Axis::HORIZONTAL ? Nodes_[node].H : Nodes_[node].V;
    }

    std::vector<Relation>& Relations_(Axis axis) noexcept
    {
        return axis == Axis::HORIZONTAL ? HRelations_ : VRelations_;
    }

    const std::vector<Relation>& Relations_(Axis axis) const noexcept
    {
        return axis == Axis::HORIZONTAL ? HRelations_ : VRelations_;
    }

    ReadResult ChildResult_(std::uint32_t locator, Axis axis) const noexcept
    {
        if (locator == NIL || locator >= RelationCount_)
        {
            return {};
        }

        const Relation& relation = Relations_(axis)[locator];
        return relation.Parent == NIL
            ? ReadResult{}
            : ReadResult{Child_(locator), locator, ReadOperation::FOUND, true};
    }

    std::uint32_t FindRelation_(
        std::size_t child,
        std::size_t parent,
        Axis axis) const noexcept
    {
        if (child >= NodeCount_ || parent >= NodeCount_)
        {
            return NIL;
        }

        for (std::uint8_t ordinal = 0u; ordinal < ParentCapacity_; ++ordinal)
        {
            const std::uint32_t locator = Locator_(child, ordinal);
            if (Relations_(axis)[locator].Parent == parent)
            {
                return locator;
            }
        }
        return NIL;
    }

    std::uint32_t FindVacancy_(std::size_t child, Axis axis) const noexcept
    {
        for (std::uint8_t ordinal = 0u; ordinal < ParentCapacity_; ++ordinal)
        {
            const std::uint32_t locator = Locator_(child, ordinal);
            if (Relations_(axis)[locator].Parent == NIL)
            {
                return locator;
            }
        }
        return NIL;
    }

    void Unlink_(std::uint32_t locator, Axis axis) noexcept
    {
        Relation& relation = Relations_(axis)[locator];
        AxisState& parent_axis = Axis_(relation.Parent, axis);

        if (relation.Previous == NIL)
            parent_axis.FirstChildRelation = relation.Next;
        else
            Relations_(axis)[relation.Previous].Next = relation.Next;

        if (relation.Next == NIL)
            parent_axis.LastChildRelation = relation.Previous;
        else
            Relations_(axis)[relation.Next].Previous = relation.Previous;

        relation.Previous = NIL;
        relation.Next = NIL;
    }
};

class CompactVectorDAG
{
public:
    bool Initialize(
        std::size_t node_count,
        std::size_t payload_words,
        std::uint8_t parent_capacity)
    {
        return Storage_.Initialize(node_count, payload_words, parent_capacity);
    }

    bool AddParent(std::size_t p, std::size_t c, Axis a, std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        std::lock_guard<std::mutex> lock(MutationMutex_);
        return Storage_.AddUnlocked(p, c, a);
    }

    bool RemoveParent(std::size_t p, std::size_t c, Axis a, std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        std::lock_guard<std::mutex> lock(MutationMutex_);
        return Storage_.RemoveUnlocked(p, c, a);
    }

    bool ReplaceParent(std::size_t old_p, std::size_t new_p, std::size_t c, Axis a, std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        std::lock_guard<std::mutex> lock(MutationMutex_);
        return Storage_.ReplaceUnlocked(old_p, new_p, c, a);
    }

    ReadResult FindParent(std::size_t c, Axis a, std::uint8_t o, std::uint32_t = 1u) noexcept
    { return Storage_.FindParent(c, a, o); }
    ReadResult StableFindParent(std::size_t c, Axis a, std::uint8_t o, std::uint32_t = 1u) noexcept
    { std::lock_guard<std::mutex> lock(MutationMutex_); return Storage_.FindParent(c, a, o); }
    ReadResult FindFirstChild(std::size_t p, Axis a, std::uint32_t = 1u) noexcept
    { return Storage_.FindFirstChild(p, a); }
    ReadResult FindLastChild(std::size_t p, Axis a, std::uint32_t = 1u) noexcept
    { return Storage_.FindLastChild(p, a); }
    ReadResult FindNextChild(std::size_t p, Axis a, std::uint32_t l, std::uint32_t = 1u) noexcept
    { return Storage_.FindNextChild(p, a, l); }
    ReadResult FindPreviousChild(std::size_t p, Axis a, std::uint32_t l, std::uint32_t = 1u) noexcept
    { return Storage_.FindPreviousChild(p, a, l); }
    bool StorePayload(std::size_t n, std::uint32_t w, std::uint64_t v, bool atomic) noexcept
    { return Storage_.StorePayload(n, w, v, atomic); }
    bool LoadPayload(std::size_t n, std::uint32_t w, std::uint64_t& v, bool atomic) noexcept
    { return Storage_.LoadPayload(n, w, v, atomic); }
    std::size_t ApproxStorageBytes() const noexcept { return Storage_.ApproxStorageBytes(); }

private:
    RuntimeVectorDAGStorage Storage_{};
    std::mutex MutationMutex_{};
};

template <typename RowMutex, bool SharedReaders>
class RowLockedVectorDAG
{
    static_assert(!SharedReaders || std::is_same_v<RowMutex, std::shared_mutex>);

public:
    bool Initialize(
        std::size_t node_count,
        std::size_t payload_words,
        std::uint8_t parent_capacity)
    {
        if (!Storage_.Initialize(node_count, payload_words, parent_capacity))
        {
            return false;
        }
        HLocks_ = std::make_unique<RowMutex[]>(node_count);
        VLocks_ = std::make_unique<RowMutex[]>(node_count);
        return true;
    }

    bool AddParent(std::size_t p, std::size_t c, Axis a, std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        if (!ValidPair_(p, c)) return false;
        std::scoped_lock lock(Lock_(p, a), Lock_(c, a));
        return Storage_.AddUnlocked(p, c, a);
    }

    bool RemoveParent(std::size_t p, std::size_t c, Axis a, std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        if (!ValidPair_(p, c)) return false;
        std::scoped_lock lock(Lock_(p, a), Lock_(c, a));
        return Storage_.RemoveUnlocked(p, c, a);
    }

    bool ReplaceParent(std::size_t old_p, std::size_t new_p, std::size_t c, Axis a, std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        if (
            old_p == new_p || !ValidPair_(old_p, c) || !ValidPair_(new_p, c)
        )
        {
            return false;
        }
        std::scoped_lock lock(Lock_(c, a), Lock_(old_p, a), Lock_(new_p, a));
        return Storage_.ReplaceUnlocked(old_p, new_p, c, a);
    }

    ReadResult FindParent(std::size_t c, Axis a, std::uint8_t o, std::uint32_t = 1u) noexcept
    { return Storage_.FindParent(c, a, o); }

    ReadResult StableFindParent(std::size_t c, Axis a, std::uint8_t o, std::uint32_t = 1u) noexcept
    {
        if (c >= Storage_.NodeCount()) return {};
        if constexpr (SharedReaders)
        {
            std::shared_lock<RowMutex> lock(Lock_(c, a));
            return Storage_.FindParent(c, a, o);
        }
        else
        {
            std::lock_guard<RowMutex> lock(Lock_(c, a));
            return Storage_.FindParent(c, a, o);
        }
    }

    ReadResult FindFirstChild(std::size_t p, Axis a, std::uint32_t = 1u) noexcept
    { return Storage_.FindFirstChild(p, a); }
    ReadResult FindLastChild(std::size_t p, Axis a, std::uint32_t = 1u) noexcept
    { return Storage_.FindLastChild(p, a); }
    ReadResult FindNextChild(std::size_t p, Axis a, std::uint32_t l, std::uint32_t = 1u) noexcept
    { return Storage_.FindNextChild(p, a, l); }
    ReadResult FindPreviousChild(std::size_t p, Axis a, std::uint32_t l, std::uint32_t = 1u) noexcept
    { return Storage_.FindPreviousChild(p, a, l); }
    bool StorePayload(std::size_t n, std::uint32_t w, std::uint64_t v, bool atomic) noexcept
    { return Storage_.StorePayload(n, w, v, atomic); }
    bool LoadPayload(std::size_t n, std::uint32_t w, std::uint64_t& v, bool atomic) noexcept
    { return Storage_.LoadPayload(n, w, v, atomic); }

private:
    RuntimeVectorDAGStorage Storage_{};
    std::unique_ptr<RowMutex[]> HLocks_{};
    std::unique_ptr<RowMutex[]> VLocks_{};

    bool ValidPair_(std::size_t parent, std::size_t child) const noexcept
    {
        return parent < Storage_.NodeCount() && child < Storage_.NodeCount() && parent < child;
    }

    RowMutex& Lock_(std::size_t node, Axis axis) noexcept
    {
        return axis == Axis::HORIZONTAL ? HLocks_[node] : VLocks_[node];
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

template <
    std::size_t NodeCount,
    std::size_t PayloadWords,
    std::uint8_t ParentCapacity,
    bool SinglePayloadRegion = false
>
class APCFabricBackend
{
public:
    using SD = SchemaDefinition;
    static_assert(PayloadWords <= UINT32_MAX);

    static constexpr std::uint32_t FABRIC_SLOT_COUNT =
        static_cast<std::uint32_t>(NodeCount);
    static constexpr std::uint8_t PARENT_CAPACITY = ParentCapacity;

    static constexpr std::uint32_t MINIMAL_SINGLE_REGION_SLOT_WORDS = []() constexpr
    {
        const std::uint32_t payload_words =
            PayloadWords == 0u ? 1u : static_cast<std::uint32_t>(PayloadWords);

        const std::uint32_t payload_cells =
            SD::AlignRegionCells(payload_words);

        const std::uint32_t required =
            SD::AlignRegionCells(
                SD::AlignRegionCells(ADS::META_CELL_COUNT) +
                payload_cells
            );

        return required < MINIMUM_APC_CELL_COUNT
            ? static_cast<std::uint32_t>(MINIMUM_APC_CELL_COUNT)
            : required;
    }();

    static constexpr std::uint32_t SLOT_WORDS =
        SinglePayloadRegion
            ? MINIMAL_SINGLE_REGION_SLOT_WORDS
            : static_cast<std::uint32_t>(MINIMUM_APC_CELL_COUNT);

    bool Initialize() noexcept
    {
        Slots_.fill(ADS::APC_INDEX_BOUND_SENTINAL);

        constexpr std::uint32_t matrix_width =
            PayloadWords == 0u ? 1u : static_cast<std::uint32_t>(PayloadWords);

        constexpr std::uint16_t active_region_mask =
            static_cast<std::uint16_t>(
                ADS::RegionBit(MacroColumnOfAPC::BOTTOM_UP_SLOT) |
                (
                    SinglePayloadRegion
                        ? 0u
                        : ADS::RegionBit(MacroColumnOfAPC::TOP_DOWN_SLOT)
                )
            );

        const SD::FabricRegionConfig region_config{
            active_region_mask,
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



        SD::RegionSchemaRecord& ff_schema_prop =
            schemas[static_cast<std::size_t>(MacroColumnOfAPC::BOTTOM_UP_SLOT)];

        ff_schema_prop.Region = MacroColumnOfAPC::BOTTOM_UP_SLOT;
        ff_schema_prop.Dtype = SD::DataTypeOfMacroColumn::UINT64_T;
        ff_schema_prop.Protocol = SD::SchemaProtocols::PRIVATE_REGION;
        ff_schema_prop.MatrixHeight = 1u;
        ff_schema_prop.MatrixWidth = matrix_width;
        ff_schema_prop.Flags = SD::SchemaFlags::BATCHED_LAST_DIM;

        if (!SD::SealDesiredSchema(ff_schema_prop, 0u))
        {
            return false;
        }

        if constexpr (!SinglePayloadRegion)
        {
            SD::RegionSchemaRecord& fb_schema =
                schemas[static_cast<std::size_t>(MacroColumnOfAPC::TOP_DOWN_SLOT)];

            fb_schema = ff_schema_prop;
            fb_schema.Region = MacroColumnOfAPC::TOP_DOWN_SLOT;
            fb_schema.Protocol = SD::SchemaProtocols::ATOMIC_WORD_ARRAY;

            if (!SD::SealDesiredSchema(fb_schema, 0u))
            {
                return false;
            }
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

                if (
                    !direct.has_value() ||
                    direct->Size() < PayloadWords ||
                    !direct->RawMutableSpan().has_value()
                )
                {
                    return false;
                }

                DirectViews_[i] = std::move(direct.value());

                if constexpr (!SinglePayloadRegion)
                {
                    auto atomic = Nodes_[i].template BuildAViewOverRegion<std::uint64_t>(
                        MacroColumnOfAPC::TOP_DOWN_SLOT
                    );

                    if (
                        !atomic.has_value() ||
                        atomic->Size() < PayloadWords
                    )
                    {
                        return false;
                    }

                    AtomicViews_[i] = std::move(atomic.value());
                }
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

    ReadResult StableFindParent(
        std::size_t child,
        Axis axis,
        std::uint8_t ordinal,
        std::uint32_t max_tries = 1u) noexcept
    {
        return FindParent(child, axis, ordinal, max_tries);
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
                if constexpr (SinglePayloadRegion)
                {
                    auto span = DirectViews_[node].RawMutableSpan();
                    if (!span.has_value())
                    {
                        return false;
                    }

                    std::atomic_ref<std::uint64_t>(span.value()[word]).store(
                        value,
                        std::memory_order_release
                    );
                    return true;
                }
                else
                {
                    return AtomicViews_[node].AtomicStore(
                        word,
                        value,
                        std::memory_order_release
                    );
                }
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
                if constexpr (SinglePayloadRegion)
                {
                    auto span = DirectViews_[node].RawMutableSpan();
                    if (!span.has_value())
                    {
                        return false;
                    }

                    value = std::atomic_ref<std::uint64_t>(span.value()[word]).load(
                        std::memory_order_acquire
                    );
                    return true;
                }
                else
                {
                    value = AtomicViews_[node].AtomicLoad(
                        word,
                        std::memory_order_acquire
                    );
                    return true;
                }
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
// Runtime APC/Fabric adapter used by configurable Tests 1-3.
// The fixed-size APCFabricBackend below remains available to Tests 4-8.
// -----------------------------------------------------------------------------

class RuntimeAPCFabricBackend
{
public:
    using SD = SchemaDefinition;

    bool Initialize(
        std::size_t node_count,
        std::size_t payload_words,
        std::uint8_t parent_capacity,
        bool single_payload_region)
    {
        if (
            node_count == 0u || node_count > UINT32_MAX ||
            payload_words == 0u || payload_words > UINT32_MAX ||
            parent_capacity == 0u ||
            parent_capacity > ADS::COMPILED_MAX_DIRECT_PARENTS_PER_AXIS
        )
        {
            return false;
        }

        NodeCount_ = node_count;
        PayloadWords_ = payload_words;
        SinglePayloadRegion_ = single_payload_region;
        Nodes_ = std::make_unique<TestAPC[]>(NodeCount_);
        Slots_ = std::make_unique<std::uint32_t[]>(NodeCount_);
        DirectViews_ = std::make_unique<RegionView<std::uint64_t>[]>(NodeCount_);
        if (!SinglePayloadRegion_)
        {
            AtomicViews_ = std::make_unique<RegionView<std::uint64_t>[]>(NodeCount_);
        }

        const std::uint32_t matrix_width = static_cast<std::uint32_t>(PayloadWords_);
        const std::uint16_t active_mask = static_cast<std::uint16_t>(
            ADS::RegionBit(MacroColumnOfAPC::BOTTOM_UP_SLOT) |
            (SinglePayloadRegion_ ? 0u : ADS::RegionBit(MacroColumnOfAPC::TOP_DOWN_SLOT)));

        const SD::FabricRegionConfig config{active_mask, 0u, matrix_width};
        if (!Fabric_.InitializeFabric(
            static_cast<std::uint32_t>(NodeCount_),
            SlotWords_(matrix_width, SinglePayloadRegion_),
            config,
            parent_capacity))
        {
            return false;
        }

        SD::RegionSchemaTable schemas{};
        SD::MakeDisabledSchemaTable(schemas);

        SD::RegionSchemaRecord& direct =
            schemas[static_cast<std::size_t>(MacroColumnOfAPC::BOTTOM_UP_SLOT)];
        direct.Region = MacroColumnOfAPC::BOTTOM_UP_SLOT;
        direct.Dtype = SD::DataTypeOfMacroColumn::UINT64_T;
        direct.Protocol = SD::SchemaProtocols::PRIVATE_REGION;
        direct.MatrixHeight = 1u;
        direct.MatrixWidth = matrix_width;
        direct.Flags = SD::SchemaFlags::BATCHED_LAST_DIM;
        if (!SD::SealDesiredSchema(direct, 0u)) return false;

        if (!SinglePayloadRegion_)
        {
            SD::RegionSchemaRecord& atomic =
                schemas[static_cast<std::size_t>(MacroColumnOfAPC::TOP_DOWN_SLOT)];
            atomic = direct;
            atomic.Region = MacroColumnOfAPC::TOP_DOWN_SLOT;
            atomic.Protocol = SD::SchemaProtocols::ATOMIC_WORD_ARRAY;
            if (!SD::SealDesiredSchema(atomic, 0u)) return false;
        }

        for (std::size_t node = 0u; node < NodeCount_; ++node)
        {
            if (!Fabric_.CreateAPC(Nodes_[node], schemas)) return false;

            const std::uint32_t slot = Nodes_[node].GetThisSlotIdx();
            if (slot != node) return false;
            Slots_[node] = slot;

            auto direct_view = Nodes_[node].BuildAViewOverRegion<std::uint64_t>(
                MacroColumnOfAPC::BOTTOM_UP_SLOT);
            if (
                !direct_view.has_value() ||
                direct_view->Size() < PayloadWords_ ||
                !direct_view->RawMutableSpan().has_value()
            )
            {
                return false;
            }
            DirectViews_[node] = std::move(direct_view.value());

            if (!SinglePayloadRegion_)
            {
                auto atomic_view = Nodes_[node].BuildAViewOverRegion<std::uint64_t>(
                    MacroColumnOfAPC::TOP_DOWN_SLOT);
                if (!atomic_view.has_value() || atomic_view->Size() < PayloadWords_)
                {
                    return false;
                }
                AtomicViews_[node] = std::move(atomic_view.value());
            }
        }
        return true;
    }

    bool AddParent(std::size_t p, std::size_t c, Axis a, std::uint32_t tries = DEFAULT_MAX_TRIES) noexcept
    {
        return p < NodeCount_ && c < NodeCount_ &&
            Nodes_[c].AddParent(Nodes_[p], EdgeTableForAxis(a), tries);
    }

    bool RemoveParent(std::size_t p, std::size_t c, Axis a, std::uint32_t tries = DEFAULT_MAX_TRIES) noexcept
    {
        return p < NodeCount_ && c < NodeCount_ &&
            Nodes_[c].RemoveParent(Nodes_[p], EdgeTableForAxis(a), tries);
    }

    bool ReplaceParent(std::size_t old_p, std::size_t new_p, std::size_t c, Axis a, std::uint32_t tries = DEFAULT_MAX_TRIES) noexcept
    {
        return old_p < NodeCount_ && new_p < NodeCount_ && c < NodeCount_ &&
            Nodes_[c].ReplaceParent(Nodes_[old_p], Nodes_[new_p], EdgeTableForAxis(a), tries);
    }

    ReadResult FindParent(std::size_t c, Axis a, std::uint8_t ordinal, std::uint32_t tries = 1u) noexcept
    {
        if (c >= NodeCount_) return {};
        TestAPC::RelationOperationForTest op{};
        AdaptivePackedCellContainer found = Nodes_[c].FindParent(
            EdgeTableForAxis(a), ordinal, &op, tries);
        return Convert_(found, op);
    }

    ReadResult StableFindParent(std::size_t c, Axis a, std::uint8_t ordinal, std::uint32_t tries = 1u) noexcept
    {
        return FindParent(c, a, ordinal, tries);
    }

    ReadResult FindFirstChild(std::size_t p, Axis a, std::uint32_t tries = 1u) noexcept
    {
        return ChildRead_(p, a, tries, 0u, ChildOperation::FIRST);
    }

    ReadResult FindLastChild(std::size_t p, Axis a, std::uint32_t tries = 1u) noexcept
    {
        return ChildRead_(p, a, tries, 0u, ChildOperation::LAST);
    }

    ReadResult FindNextChild(std::size_t p, Axis a, std::uint32_t locator, std::uint32_t tries = 1u) noexcept
    {
        return ChildRead_(p, a, tries, locator, ChildOperation::NEXT);
    }

    ReadResult FindPreviousChild(std::size_t p, Axis a, std::uint32_t locator, std::uint32_t tries = 1u) noexcept
    {
        return ChildRead_(p, a, tries, locator, ChildOperation::PREVIOUS);
    }

    BenchmarkReadResult BenchmarkFindParent(std::size_t c, Axis a, std::uint8_t ordinal, std::uint32_t tries = 1u) noexcept
    {
        if (c >= NodeCount_) return {};
        TestAPC::RelationOperationForTest op{};
        AdaptivePackedCellContainer found = Nodes_[c].FindParent(
            EdgeTableForAxis(a), ordinal, &op, tries);
        (void)found;
        return BenchmarkConvert_(op, BenchmarkReadResult::NO_NODE);
    }

    BenchmarkReadResult BenchmarkFindFirstChild(std::size_t p, Axis a, std::uint32_t tries = 1u) noexcept
    { return BenchmarkChildRead_(p, a, tries, 0u, ChildOperation::FIRST); }
    BenchmarkReadResult BenchmarkFindLastChild(std::size_t p, Axis a, std::uint32_t tries = 1u) noexcept
    { return BenchmarkChildRead_(p, a, tries, 0u, ChildOperation::LAST); }
    BenchmarkReadResult BenchmarkFindNextChild(std::size_t p, Axis a, std::uint32_t l, std::uint32_t tries = 1u) noexcept
    { return BenchmarkChildRead_(p, a, tries, l, ChildOperation::NEXT); }
    BenchmarkReadResult BenchmarkFindPreviousChild(std::size_t p, Axis a, std::uint32_t l, std::uint32_t tries = 1u) noexcept
    { return BenchmarkChildRead_(p, a, tries, l, ChildOperation::PREVIOUS); }

    bool StorePayload(std::size_t node, std::uint32_t word, std::uint64_t value, bool atomic) noexcept
    {
        if (node >= NodeCount_ || word >= PayloadWords_) return false;
        if (atomic && !SinglePayloadRegion_)
        {
            return AtomicViews_[node].AtomicStore(word, value, std::memory_order_release);
        }

        auto span = DirectViews_[node].RawMutableSpan();
        if (!span.has_value()) return false;
        if (atomic)
            std::atomic_ref<std::uint64_t>(span.value()[word]).store(value, std::memory_order_release);
        else
            span.value()[word] = value;
        return true;
    }

    bool LoadPayload(std::size_t node, std::uint32_t word, std::uint64_t& value, bool atomic) noexcept
    {
        if (node >= NodeCount_ || word >= PayloadWords_) return false;
        if (atomic && !SinglePayloadRegion_)
        {
            value = AtomicViews_[node].AtomicLoad(word, std::memory_order_acquire);
            return true;
        }

        auto span = DirectViews_[node].RawMutableSpan();
        if (!span.has_value()) return false;
        value = atomic
            ? std::atomic_ref<std::uint64_t>(span.value()[word]).load(std::memory_order_acquire)
            : span.value()[word];
        return true;
    }

    std::size_t ApproxStorageBytes() const noexcept { return Fabric_.SlabBytesForTest(); }

private:
    enum class ChildOperation : std::uint8_t { FIRST, LAST, NEXT, PREVIOUS };

    std::size_t NodeCount_ = 0u;
    std::size_t PayloadWords_ = 0u;
    bool SinglePayloadRegion_ = false;
    ResolverTestFabric Fabric_{};
    std::unique_ptr<TestAPC[]> Nodes_{};
    std::unique_ptr<std::uint32_t[]> Slots_{};
    std::unique_ptr<RegionView<std::uint64_t>[]> DirectViews_{};
    std::unique_ptr<RegionView<std::uint64_t>[]> AtomicViews_{};

    static std::uint32_t SlotWords_(std::uint32_t payload_words, bool single) noexcept
    {
        if (!single) return static_cast<std::uint32_t>(MINIMUM_APC_CELL_COUNT);
        const std::uint32_t payload_cells = SD::AlignRegionCells(payload_words);
        const std::uint32_t required = SD::AlignRegionCells(
            SD::AlignRegionCells(ADS::META_CELL_COUNT) + payload_cells);
        return std::max(required, static_cast<std::uint32_t>(MINIMUM_APC_CELL_COUNT));
    }

    std::size_t IndexOfSlot_(std::uint32_t slot) const noexcept
    {
        return slot < NodeCount_ && Slots_[slot] == slot
            ? static_cast<std::size_t>(slot)
            : ReadResult::NO_NODE;
    }

    ReadResult Convert_(
        AdaptivePackedCellContainer& found,
        const TestAPC::RelationOperationForTest& op) const noexcept
    {
        const std::size_t node = IndexOfSlot_(found.GetThisSlotIdx());
        return ReadResult{
            node,
            op.RelationLocator_,
            op.MutationOP_,
            node != ReadResult::NO_NODE
        };
    }

    static BenchmarkReadResult BenchmarkConvert_(
        const TestAPC::RelationOperationForTest& op,
        std::size_t node_hint) noexcept
    {
        const bool present = op.MutationOP_ == ReadOperation::FOUND;
        return BenchmarkReadResult{
            present ? node_hint : BenchmarkReadResult::NO_NODE,
            op.RelationLocator_, op.MutationOP_, present};
    }

    static BenchmarkReadResult BenchmarkChildConvert_(
        const TestAPC::RelationOperationForTest& op) noexcept
    {
        const std::size_t node_hint =
            op.MutationOP_ == ReadOperation::FOUND && op.RelationLocator_ != UINT32_MAX
                ? static_cast<std::size_t>(EdgeBuilder::RelationSlot(op.RelationLocator_))
                : BenchmarkReadResult::NO_NODE;
        return BenchmarkConvert_(op, node_hint);
    }

    ReadResult ChildRead_(
        std::size_t parent,
        Axis axis,
        std::uint32_t tries,
        std::uint32_t locator,
        ChildOperation operation) noexcept
    {
        if (parent >= NodeCount_) return {};
        TestAPC::RelationOperationForTest op{};
        AdaptivePackedCellContainer found{};
        switch (operation)
        {
        case ChildOperation::FIRST:
            found = Nodes_[parent].FindFirstChild(EdgeTableForAxis(axis), &op, tries); break;
        case ChildOperation::LAST:
            found = Nodes_[parent].FindLastChild(EdgeTableForAxis(axis), &op, tries); break;
        case ChildOperation::NEXT:
            found = Nodes_[parent].FindNextChild(EdgeTableForAxis(axis), locator, &op, tries); break;
        case ChildOperation::PREVIOUS:
            found = Nodes_[parent].FindPreviousChild(EdgeTableForAxis(axis), locator, &op, tries); break;
        }
        return Convert_(found, op);
    }

    BenchmarkReadResult BenchmarkChildRead_(
        std::size_t parent,
        Axis axis,
        std::uint32_t tries,
        std::uint32_t locator,
        ChildOperation operation) noexcept
    {
        if (parent >= NodeCount_) return {};
        TestAPC::RelationOperationForTest op{};
        AdaptivePackedCellContainer found{};
        switch (operation)
        {
        case ChildOperation::FIRST:
            found = Nodes_[parent].FindFirstChild(EdgeTableForAxis(axis), &op, tries); break;
        case ChildOperation::LAST:
            found = Nodes_[parent].FindLastChild(EdgeTableForAxis(axis), &op, tries); break;
        case ChildOperation::NEXT:
            found = Nodes_[parent].FindNextChild(EdgeTableForAxis(axis), locator, &op, tries); break;
        case ChildOperation::PREVIOUS:
            found = Nodes_[parent].FindPreviousChild(EdgeTableForAxis(axis), locator, &op, tries); break;
        }
        (void)found;
        return BenchmarkChildConvert_(op);
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
    std::uint32_t attempt_limit =
        ConcurrencyConfig::TRANSACTION_ATTEMPT_LIMIT) noexcept
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
// Configurable benchmark core for Tests 1-3.
// -----------------------------------------------------------------------------

struct BenchmarkCase
{
    std::size_t NodeCount = 0u;
    std::uint8_t ParentCapacity = 0u;
};

inline std::array<BenchmarkCase, 4u> MakeBenchmarkCases(
    std::size_t lower_nodes,
    std::size_t higher_nodes,
    std::uint8_t lower_k,
    std::uint8_t higher_k) noexcept
{
    return {{
        {lower_nodes, lower_k},
        {lower_nodes, higher_k},
        {higher_nodes, lower_k},
        {higher_nodes, higher_k}
    }};
}

namespace BenchmarkCore
{
constexpr std::size_t TEST1_PAYLOAD_WORDS = 128u;
constexpr std::uint64_t TARGET_TRAVERSAL_CALLS = 2'000'000u;
constexpr std::uint64_t TARGET_PAYLOAD_CALLS = 4'000'000u;
constexpr std::uint64_t TARGET_GRAPH_PAYLOAD_CALLS = 1'500'000u;
constexpr std::uint32_t TEST1_MUTATION_ROUNDS = 100'000u;

struct PairTiming
{
    double Vector = 0.0;
    double Fabric = 0.0;
};

inline std::atomic<std::uint64_t>& BenchmarkSink() noexcept
{
    static std::atomic<std::uint64_t> sink{0u};
    return sink;
}

inline void Consume(std::uint64_t value) noexcept
{
    BenchmarkSink().store(value, std::memory_order_relaxed);
}

inline std::uint32_t RoundsForTarget(
    std::uint64_t target,
    std::uint64_t per_round) noexcept
{
    if (per_round == 0u) return 1u;
    const std::uint64_t rounds = (target + per_round - 1u) / per_round;
    return static_cast<std::uint32_t>(std::max<std::uint64_t>(1u, rounds));
}

inline std::uint64_t EdgeCountPerAxis(
    std::size_t node_count,
    std::uint8_t k) noexcept
{
    if (node_count <= 1u) return 0u;
    const std::uint64_t n = static_cast<std::uint64_t>(node_count - 1u);
    const std::uint64_t capacity = static_cast<std::uint64_t>(k);
    const std::uint64_t ramp = std::min(n, capacity);
    return ramp * (ramp + 1u) / 2u + (n - ramp) * ramp;
}

inline std::uint64_t NextRandom(std::uint64_t& state) noexcept
{
    state ^= state << ConcurrencyConfig::RANDOM_LEFT_SHIFT_A;
    state ^= state >> ConcurrencyConfig::RANDOM_RIGHT_SHIFT;
    state ^= state << ConcurrencyConfig::RANDOM_LEFT_SHIFT_B;
    return state;
}

inline std::size_t DifferentRandomParent(
    std::uint64_t& state,
    std::size_t parent_count,
    std::size_t current) noexcept
{
    std::size_t next = static_cast<std::size_t>(NextRandom(state) % parent_count);
    if (next == current) next = (next + 1u) % parent_count;
    return next;
}

template <typename Backend>
bool InitializeBackend(
    Backend& backend,
    const BenchmarkCase& config,
    std::size_t payload_words,
    bool single_payload_region = false)
{
    if constexpr (requires {
        backend.Initialize(
            config.NodeCount,
            payload_words,
            config.ParentCapacity,
            single_payload_region);
    })
    {
        return backend.Initialize(
            config.NodeCount,
            payload_words,
            config.ParentCapacity,
            single_payload_region);
    }
    else
    {
        (void)single_payload_region;
        return backend.Initialize(
            config.NodeCount,
            payload_words,
            config.ParentCapacity);
    }
}

template <typename Backend>
bool InitializePayload(Backend& backend, const BenchmarkCase& config, std::size_t words)
{
    for (std::size_t node = 0u; node < config.NodeCount; ++node)
    {
        for (std::uint32_t word = 0u; word < words; ++word)
        {
            const std::uint64_t value =
                (static_cast<std::uint64_t>(node + 1u) << 32u) ^
                static_cast<std::uint64_t>(word + 1u);
            if (!backend.StorePayload(node, word, value, false)) return false;
        }
    }
    return true;
}

template <typename Backend>
bool BuildFullTest1Graph(Backend& backend, const BenchmarkCase& config)
{
    if (
        !InitializeBackend(
            backend, config, TEST1_PAYLOAD_WORDS, true) ||
        !InitializePayload(backend, config, TEST1_PAYLOAD_WORDS)
    )
    {
        return false;
    }

    for (std::size_t child = 1u; child < config.NodeCount; ++child)
    {
        const std::size_t count = std::min<std::size_t>(config.ParentCapacity, child);
        for (std::uint8_t ordinal = 0u; ordinal < count; ++ordinal)
        {
            const std::size_t h_parent = child - 1u - ordinal;
            const std::size_t v_parent = ordinal;
            if (
                !backend.AddParent(h_parent, child, Axis::HORIZONTAL) ||
                !backend.AddParent(v_parent, child, Axis::VERTICAL)
            )
            {
                return false;
            }
        }
    }
    return true;
}

template <typename Fn>
double MeasureNsPerOperation(std::uint64_t operations, Fn&& fn)
{
    const auto begin = Clock::now();
    const std::uint64_t checksum = fn();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - begin).count();
    Consume(checksum);
    return operations == 0u
        ? 0.0
        : static_cast<double>(elapsed) / static_cast<double>(operations);
}

template <typename VectorFn, typename FabricFn>
PairTiming MeasurePair(VectorFn&& vector_fn, FabricFn&& fabric_fn)
{
    std::array<double, ConcurrencyConfig::MEASURED_RUNS> vector_samples{};
    std::array<double, ConcurrencyConfig::MEASURED_RUNS> fabric_samples{};

    for (std::size_t run = 0u; run < ConcurrencyConfig::MEASURED_RUNS; ++run)
    {
        if ((run & 1u) == 0u)
        {
            vector_samples[run] = vector_fn();
            fabric_samples[run] = fabric_fn();
        }
        else
        {
            fabric_samples[run] = fabric_fn();
            vector_samples[run] = vector_fn();
        }
    }
    return {Median(vector_samples), Median(fabric_samples)};
}

inline void PrintPairRow(const char* name, const PairTiming& timing)
{
    std::cout
        << "  " << std::left << std::setw(27) << name
        << " vector=" << std::right << std::setw(10)
        << std::fixed << std::setprecision(2) << timing.Vector << " ns/op"
        << "  Fabric=" << std::setw(10) << timing.Fabric << " ns/op"
        << "  Fabric/vector=" << std::setw(7)
        << Ratio(timing.Fabric, timing.Vector) << "x\n";
}

template <typename Backend>
bool HasParentInFlat(
    const std::vector<std::uint32_t>& flat,
    std::size_t child,
    std::size_t parent,
    std::uint8_t k) noexcept
{
    const std::size_t base = child * static_cast<std::size_t>(k);
    for (std::uint8_t ordinal = 0u; ordinal < k; ++ordinal)
    {
        if (flat[base + ordinal] == parent) return true;
    }
    return false;
}

template <typename Backend>
GraphProof ProveRuntimeCombinedDAG(
    Backend& backend,
    const BenchmarkCase& config)
{
    GraphProof proof{};
    const std::size_t relation_slots =
        config.NodeCount * static_cast<std::size_t>(config.ParentCapacity);

    for (const Axis axis : {Axis::HORIZONTAL, Axis::VERTICAL})
    {
        std::vector<std::uint32_t> flat(
            relation_slots, RuntimeVectorDAGStorage::NIL);
        std::vector<std::uint32_t> expected_children(config.NodeCount, 0u);

        for (std::size_t child = 0u; child < config.NodeCount; ++child)
        {
            for (std::uint8_t ordinal = 0u; ordinal < config.ParentCapacity; ++ordinal)
            {
                const ReadResult read = backend.FindParent(
                    child, axis, ordinal, DEFAULT_MAX_TRIES);
                proof.ReadContracts = proof.ReadContracts && read.ContractValid();
                if (read.IsRetry())
                {
                    proof.ReadContracts = false;
                    continue;
                }
                if (!read.IsFound()) continue;

                if (read.Node >= child || read.Node >= config.NodeCount)
                    proof.ParentOrder = false;

                const std::size_t base = child * config.ParentCapacity;
                for (std::uint8_t prior = 0u; prior < ordinal; ++prior)
                {
                    if (flat[base + prior] == read.Node) proof.NoDuplicates = false;
                }
                flat[base + ordinal] = static_cast<std::uint32_t>(read.Node);
                if (read.Node < config.NodeCount) ++expected_children[read.Node];
            }
        }

        std::vector<std::uint32_t> seen(config.NodeCount, 0u);
        std::uint32_t stamp = 1u;
        for (std::size_t parent = 0u; parent < config.NodeCount; ++parent, ++stamp)
        {
            std::size_t count = 0u;
            ReadResult read = backend.FindFirstChild(parent, axis, DEFAULT_MAX_TRIES);
            proof.ReadContracts = proof.ReadContracts && read.ContractValid();
            while (read.IsFound())
            {
                if (
                    read.Node >= config.NodeCount ||
                    seen[read.Node] == stamp ||
                    !HasParentInFlat<Backend>(
                        flat, read.Node, parent, config.ParentCapacity)
                )
                {
                    proof.ReverseLists = false;
                    break;
                }
                seen[read.Node] = stamp;
                ++count;
                if (count > expected_children[parent])
                {
                    proof.ReverseLists = false;
                    break;
                }
                read = backend.FindNextChild(
                    parent, axis, read.Locator, DEFAULT_MAX_TRIES);
                proof.ReadContracts = proof.ReadContracts && read.ContractValid();
            }
            if (read.IsRetry()) proof.ReadContracts = false;
            if (count != expected_children[parent]) proof.ReverseLists = false;
        }

        std::fill(seen.begin(), seen.end(), 0u);
        stamp = 1u;
        for (std::size_t parent = 0u; parent < config.NodeCount; ++parent, ++stamp)
        {
            std::size_t count = 0u;
            ReadResult read = backend.FindLastChild(parent, axis, DEFAULT_MAX_TRIES);
            proof.ReadContracts = proof.ReadContracts && read.ContractValid();
            while (read.IsFound())
            {
                if (
                    read.Node >= config.NodeCount ||
                    seen[read.Node] == stamp ||
                    !HasParentInFlat<Backend>(
                        flat, read.Node, parent, config.ParentCapacity)
                )
                {
                    proof.ReverseLists = false;
                    break;
                }
                seen[read.Node] = stamp;
                ++count;
                if (count > expected_children[parent])
                {
                    proof.ReverseLists = false;
                    break;
                }
                read = backend.FindPreviousChild(
                    parent, axis, read.Locator, DEFAULT_MAX_TRIES);
                proof.ReadContracts = proof.ReadContracts && read.ContractValid();
            }
            if (read.IsRetry()) proof.ReadContracts = false;
            if (count != expected_children[parent]) proof.ReverseLists = false;
        }
    }

    proof.CombinedAcyclic = proof.ParentOrder;
    return proof;
}

template <typename Backend>
bool ReverseContains(
    Backend& backend,
    std::size_t parent,
    std::size_t child,
    Axis axis,
    std::size_t node_count)
{
    ReadResult read = backend.FindFirstChild(parent, axis, DEFAULT_MAX_TRIES);
    std::size_t steps = 0u;
    while (read.IsFound() && steps++ <= node_count)
    {
        if (read.Node == child) return true;
        read = backend.FindNextChild(parent, axis, read.Locator, DEFAULT_MAX_TRIES);
    }
    return false;
}

// -------------------------------------------------------------------------
// Test 2 shared mutation scenario.
// -------------------------------------------------------------------------

enum class MutationLocality : std::uint8_t { HOTSPOT, DISTRIBUTED };

struct MutationStep
{
    std::uint32_t HParent = 0u;
    std::uint32_t VParent = 0u;
};

struct MutationScenario
{
    BenchmarkCase Config{};
    MutationLocality Locality = MutationLocality::HOTSPOT;
    std::size_t MaxWriters = 0u;
    std::size_t FirstChild = 0u;
    std::size_t ParentCount = 0u;

    std::size_t Child(std::size_t writer) const noexcept
    { return FirstChild + writer; }

    std::size_t InitialH(std::size_t writer) const noexcept
    {
        return Locality == MutationLocality::HOTSPOT
            ? 0u
            : writer % ParentCount;
    }

    std::size_t InitialV(std::size_t writer) const noexcept
    {
        return Locality == MutationLocality::HOTSPOT
            ? 2u
            : (writer + ParentCount / 2u) % ParentCount;
    }
};

inline std::optional<MutationScenario> MakeMutationScenario(
    const BenchmarkCase& config,
    MutationLocality locality,
    std::size_t max_writers)
{
    MutationScenario scenario{config, locality, max_writers};
    if (locality == MutationLocality::HOTSPOT)
    {
        scenario.ParentCount = 4u;
        scenario.FirstChild = 4u;
        if (config.NodeCount < scenario.FirstChild + max_writers) return std::nullopt;
    }
    else
    {
        if (config.NodeCount <= max_writers + 1u) return std::nullopt;
        scenario.FirstChild = config.NodeCount - max_writers;
        scenario.ParentCount = std::min(
            ConcurrencyConfig::DISTRIBUTED_PARENT_LIMIT,
            scenario.FirstChild);
        if (scenario.ParentCount < 2u) return std::nullopt;
    }
    return scenario;
}

using MutationSchedule = std::vector<std::vector<MutationStep>>;

inline MutationSchedule BuildMutationSchedule(const MutationScenario& scenario)
{
    MutationSchedule schedule(
        scenario.MaxWriters,
        std::vector<MutationStep>(ConcurrencyConfig::MUTATIONS_PER_WRITER));

    for (std::size_t writer = 0u; writer < scenario.MaxWriters; ++writer)
    {
        std::size_t h_current = scenario.InitialH(writer);
        std::size_t v_current = scenario.InitialV(writer);
        std::uint64_t state = ConcurrencyConfig::RANDOM_SEED ^
            (static_cast<std::uint64_t>(writer + 1u) *
             ConcurrencyConfig::RANDOM_STREAM_STEP);

        for (std::uint32_t step = 0u; step < ConcurrencyConfig::MUTATIONS_PER_WRITER; ++step)
        {
            std::size_t h_next = 0u;
            std::size_t v_next = 0u;
            if (scenario.Locality == MutationLocality::HOTSPOT)
            {
                h_next = h_current == 0u ? 1u : 0u;
                v_next = v_current == 2u ? 3u : 2u;
            }
            else
            {
                h_next = DifferentRandomParent(state, scenario.ParentCount, h_current);
                v_next = DifferentRandomParent(state, scenario.ParentCount, v_current);
            }
            schedule[writer][step] = {
                static_cast<std::uint32_t>(h_next),
                static_cast<std::uint32_t>(v_next)};
            h_current = h_next;
            v_current = v_next;
        }
    }
    return schedule;
}

template <typename Backend>
bool BuildMutationBackend(
    Backend& backend,
    const MutationScenario& scenario,
    std::size_t writer_count)
{
    if (
        writer_count == 0u || writer_count > scenario.MaxWriters ||
        !InitializeBackend(
            backend,
            scenario.Config,
            ConcurrencyConfig::BENCHMARK_PAYLOAD_WORDS,
            false)
    )
    {
        return false;
    }

    for (std::size_t writer = 0u; writer < writer_count; ++writer)
    {
        const std::size_t child = scenario.Child(writer);
        if (
            !backend.AddParent(scenario.InitialH(writer), child, Axis::HORIZONTAL) ||
            !backend.AddParent(scenario.InitialV(writer), child, Axis::VERTICAL)
        )
        {
            return false;
        }
    }
    return true;
}

struct MutationSweepResult
{
    bool Ok = false;
    double NsPerSuccess = 0.0;
    std::uint64_t Success = 0u;
    std::uint64_t Retries = 0u;
};

template <typename Backend>
MutationSweepResult RunMutationWorkers(
    Backend& backend,
    const MutationScenario& scenario,
    const MutationSchedule& schedule,
    std::size_t writer_count)
{
    Clock::time_point begin{};
    Clock::time_point end{};
    auto begin_phase = [&]() noexcept { begin = Clock::now(); };
    auto end_phase = [&]() noexcept { end = Clock::now(); };
    std::barrier start(
        static_cast<std::ptrdiff_t>(writer_count + 1u), begin_phase);
    std::barrier finish(
        static_cast<std::ptrdiff_t>(writer_count + 1u), end_phase);

    std::atomic<bool> failed{false};
    std::atomic<std::uint64_t> success{0u};
    std::atomic<std::uint64_t> retries{0u};
    std::vector<std::thread> writers;
    writers.reserve(writer_count);

    for (std::size_t writer = 0u; writer < writer_count; ++writer)
    {
        writers.emplace_back([&, writer]() noexcept
        {
            const std::size_t child = scenario.Child(writer);
            std::size_t h_current = scenario.InitialH(writer);
            std::size_t v_current = scenario.InitialV(writer);
            std::uint64_t local_success = 0u;
            std::uint64_t local_retries = 0u;

            start.arrive_and_wait();
            for (std::uint32_t step = 0u; step < ConcurrencyConfig::MUTATIONS_PER_WRITER; ++step)
            {
                const MutationStep mutation = schedule[writer][step];
                if (
                    !RetryReplace(
                        backend, h_current, mutation.HParent, child,
                        Axis::HORIZONTAL, local_retries) ||
                    !RetryReplace(
                        backend, v_current, mutation.VParent, child,
                        Axis::VERTICAL, local_retries)
                )
                {
                    failed.store(true, std::memory_order_release);
                    break;
                }
                h_current = mutation.HParent;
                v_current = mutation.VParent;
                local_success += ConcurrencyConfig::OPERATIONS_PER_MUTATION_STEP;
            }
            success.fetch_add(local_success, std::memory_order_relaxed);
            retries.fetch_add(local_retries, std::memory_order_relaxed);
            finish.arrive_and_wait();
        });
    }

    start.arrive_and_wait();
    finish.arrive_and_wait();
    for (std::thread& writer : writers) writer.join();

    const std::uint64_t completed = success.load(std::memory_order_acquire);
    const std::uint64_t expected =
        writer_count * ConcurrencyConfig::MUTATIONS_PER_WRITER *
        ConcurrencyConfig::OPERATIONS_PER_MUTATION_STEP;
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - begin).count();

    return {
        !failed.load(std::memory_order_acquire) && completed == expected,
        completed == 0u ? 0.0 :
            static_cast<double>(elapsed) / static_cast<double>(completed),
        completed,
        retries.load(std::memory_order_acquire)
    };
}

template <typename Backend>
bool VerifyMutationScenario(
    Backend& backend,
    const MutationScenario& scenario,
    const MutationSchedule& schedule,
    std::size_t writer_count)
{
    for (std::size_t writer = 0u; writer < writer_count; ++writer)
    {
        const std::size_t child = scenario.Child(writer);
        const MutationStep expected = schedule[writer].back();
        for (const auto [axis, parent] : std::array<std::pair<Axis, std::size_t>, 2u>{{
            {Axis::HORIZONTAL, expected.HParent},
            {Axis::VERTICAL, expected.VParent}}})
        {
            const ReadResult read = backend.FindParent(child, axis, 0u, DEFAULT_MAX_TRIES);
            if (!read.IsFound() || read.Node != parent) return false;
            if (!ReverseContains(
                backend, parent, child, axis, scenario.Config.NodeCount)) return false;

            for (std::uint8_t ordinal = 1u; ordinal < scenario.Config.ParentCapacity; ++ordinal)
            {
                const ReadResult empty = backend.FindParent(
                    child, axis, ordinal, DEFAULT_MAX_TRIES);
                if (!empty.IsNone()) return false;
            }
        }
    }
    return true;
}

// -------------------------------------------------------------------------
// Test 3 shared reader/writer scenario.
// -------------------------------------------------------------------------

struct WriterSpec
{
    Axis RelationAxis = Axis::HORIZONTAL;
    std::size_t Child = 0u;
    std::size_t InitialParent = 0u;
};

struct ReaderScenario
{
    BenchmarkCase Config{};
    bool Distributed = false;
    std::size_t ParentCount = 0u;
    std::array<WriterSpec, ConcurrencyConfig::READER_WRITER_COUNT> Writers{};

    bool ParentAllowed(std::size_t writer, std::size_t parent) const noexcept
    {
        if (Distributed) return parent < ParentCount;
        return writer == 0u ? (parent == 0u || parent == 1u)
                            : (parent == 2u || parent == 3u);
    }
};

inline std::optional<ReaderScenario> MakeReaderScenario(
    const BenchmarkCase& config,
    bool distributed)
{
    ReaderScenario scenario{};
    scenario.Config = config;
    scenario.Distributed = distributed;

    if (!distributed)
    {
        if (config.NodeCount < 5u) return std::nullopt;
        const std::size_t child = config.NodeCount - 1u;
        scenario.ParentCount = 4u;
        scenario.Writers = {{
            {Axis::HORIZONTAL, child, 0u},
            {Axis::VERTICAL, child, 2u}
        }};
    }
    else
    {
        if (config.NodeCount < 4u) return std::nullopt;
        scenario.ParentCount = std::min(
            ConcurrencyConfig::DISTRIBUTED_PARENT_LIMIT,
            config.NodeCount - ConcurrencyConfig::READER_WRITER_COUNT);
        if (scenario.ParentCount < 2u) return std::nullopt;
        scenario.Writers = {{
            {Axis::HORIZONTAL, config.NodeCount - 2u, 0u},
            {Axis::VERTICAL, config.NodeCount - 1u, scenario.ParentCount / 2u}
        }};
    }
    return scenario;
}

using ParentSchedule = std::array<
    std::vector<std::uint32_t>,
    ConcurrencyConfig::READER_WRITER_COUNT>;

inline ParentSchedule BuildParentSchedule(const ReaderScenario& scenario)
{
    ParentSchedule schedule{};
    for (std::size_t writer = 0u; writer < schedule.size(); ++writer)
    {
        schedule[writer].resize(ConcurrencyConfig::MUTATIONS_PER_WRITER);
        std::size_t current = scenario.Writers[writer].InitialParent;
        std::uint64_t state = ConcurrencyConfig::RANDOM_SEED ^
            (static_cast<std::uint64_t>(writer + 17u) *
             ConcurrencyConfig::RANDOM_STREAM_STEP);

        for (std::uint32_t step = 0u; step < ConcurrencyConfig::MUTATIONS_PER_WRITER; ++step)
        {
            std::size_t next = 0u;
            if (!scenario.Distributed)
            {
                const std::size_t first = writer == 0u ? 0u : 2u;
                const std::size_t second = writer == 0u ? 1u : 3u;
                next = current == first ? second : first;
            }
            else
            {
                next = DifferentRandomParent(state, scenario.ParentCount, current);
            }
            schedule[writer][step] = static_cast<std::uint32_t>(next);
            current = next;
        }
    }
    return schedule;
}

template <typename Backend>
bool BuildReaderBackend(Backend& backend, const ReaderScenario& scenario)
{
    if (!InitializeBackend(
        backend,
        scenario.Config,
        ConcurrencyConfig::BENCHMARK_PAYLOAD_WORDS,
        false))
    {
        return false;
    }

    for (const WriterSpec& writer : scenario.Writers)
    {
        if (!backend.AddParent(
            writer.InitialParent, writer.Child, writer.RelationAxis))
        {
            return false;
        }
    }
    return true;
}

enum class StableReadStatus : std::uint8_t
{
    SUCCESS,
    RETRY_LIMIT,
    BAD_CONTRACT,
    INVALID_PARENT
};

template <typename Backend>
StableReadStatus StableReadOne(
    Backend& backend,
    const ReaderScenario& scenario,
    std::size_t writer,
    std::uint64_t& retries) noexcept
{
    const WriterSpec& spec = scenario.Writers[writer];

    for (
        std::uint32_t attempt = 0u;
        attempt < ConcurrencyConfig::STABLE_READ_ATTEMPT_LIMIT;
        ++attempt
    )
    {
        const ReadResult read = backend.StableFindParent(
            spec.Child, spec.RelationAxis, 0u, 1u);

        if (!read.ContractValid())
        {
            return StableReadStatus::BAD_CONTRACT;
        }

        if (read.IsRetry())
        {
            ++retries;
            PerturbSchedule(attempt);
            continue;
        }

        if (
            !read.IsFound() ||
            !scenario.ParentAllowed(writer, read.Node)
        )
        {
            return StableReadStatus::INVALID_PARENT;
        }

        return StableReadStatus::SUCCESS;
    }

    return StableReadStatus::RETRY_LIMIT;
}

struct ReaderSweepResult
{
    bool Ok = false;
    bool CorrectnessOk = false;
    bool ProgressOk = false;
    bool FinalVerificationOk = false;
    double NsPerStableRead = 0.0;
    double ElapsedNs = 0.0;
    std::uint64_t StableReads = 0u;
    std::uint64_t ReaderRetries = 0u;
    std::uint64_t ReaderStarvations = 0u;
    std::uint64_t WriterSuccess = 0u;
    std::uint64_t WriterRetries = 0u;
    std::uint64_t WriterExhaustions = 0u;
};

template <typename Backend>
bool VerifyReaderScenario(Backend& backend, const ReaderScenario& scenario)
{
    for (std::size_t writer = 0u; writer < scenario.Writers.size(); ++writer)
    {
        const WriterSpec& spec = scenario.Writers[writer];
        const ReadResult read = backend.FindParent(
            spec.Child, spec.RelationAxis, 0u, DEFAULT_MAX_TRIES);
        if (!read.IsFound() || !scenario.ParentAllowed(writer, read.Node)) return false;
        if (!ReverseContains(
            backend, read.Node, spec.Child, spec.RelationAxis,
            scenario.Config.NodeCount)) return false;
    }
    return true;
}

template <typename Backend>
ReaderSweepResult RunReadersWithWriters(
    Backend& backend,
    const ReaderScenario& scenario,
    const ParentSchedule& schedule,
    std::size_t reader_count)
{
    std::barrier writer_start(static_cast<std::ptrdiff_t>(
        ConcurrencyConfig::READER_WRITER_COUNT + 1u));

    Clock::time_point begin{};
    Clock::time_point end{};
    std::atomic<bool> measure_writers{false};
    auto begin_phase = [&]() noexcept
    {
        measure_writers.store(true, std::memory_order_release);
        begin = Clock::now();
    };
    auto end_phase = [&]() noexcept
    {
        end = Clock::now();
        measure_writers.store(false, std::memory_order_release);
    };
    std::barrier reader_start(
        static_cast<std::ptrdiff_t>(reader_count + 1u), begin_phase);
    std::barrier reader_finish(
        static_cast<std::ptrdiff_t>(reader_count + 1u), end_phase);

    // Keep correctness and progress failures separate. A malformed/illegal
    // stable read is a correctness failure. Exhausting a retry budget is a
    // progress event; the TestKit retries that logical read for a bounded
    // number of starvation rounds before declaring the measurement incomplete.
    std::atomic<bool> correctness_failed{false};
    std::atomic<bool> progress_failed{false};
    std::atomic<bool> stop_writers{false};
    std::atomic<std::size_t> warmed_writers{0u};
    std::atomic<std::uint64_t> stable_reads{0u};
    std::atomic<std::uint64_t> reader_retries{0u};
    std::atomic<std::uint64_t> reader_starvations{0u};
    std::atomic<std::uint64_t> writer_success{0u};
    std::atomic<std::uint64_t> writer_retries{0u};
    std::atomic<std::uint64_t> writer_exhaustions{0u};

    const auto abort_requested = [&]() noexcept
    {
        return
            correctness_failed.load(std::memory_order_acquire) ||
            progress_failed.load(std::memory_order_acquire);
    };

    std::vector<std::thread> writers;
    writers.reserve(ConcurrencyConfig::READER_WRITER_COUNT);
    for (std::size_t writer = 0u; writer < ConcurrencyConfig::READER_WRITER_COUNT; ++writer)
    {
        writers.emplace_back([&, writer]() noexcept
        {
            const WriterSpec spec = scenario.Writers[writer];
            std::size_t current = spec.InitialParent;
            std::size_t schedule_index = 0u;

            auto mutate_once = [&]() noexcept -> bool
            {
                for (std::size_t guard = 0u; guard < schedule[writer].size(); ++guard)
                {
                    const std::size_t target =
                        schedule[writer][schedule_index++ % schedule[writer].size()];
                    if (target == current) continue;

                    std::uint64_t local_retries = 0u;
                    if (!RetryReplace(
                        backend, current, target, spec.Child,
                        spec.RelationAxis, local_retries))
                    {
                        writer_exhaustions.fetch_add(1u, std::memory_order_relaxed);
                        return false;
                    }

                    current = target;
                    if (measure_writers.load(std::memory_order_acquire))
                    {
                        writer_success.fetch_add(1u, std::memory_order_relaxed);
                        writer_retries.fetch_add(local_retries, std::memory_order_relaxed);
                    }
                    return true;
                }
                return false;
            };

            writer_start.arrive_and_wait();

            for (
                std::uint32_t i = 0u;
                i < ConcurrencyConfig::WRITER_WARMUP_MUTATIONS &&
                !abort_requested();
                ++i
            )
            {
                if (!mutate_once())
                {
                    progress_failed.store(true, std::memory_order_release);
                    break;
                }
            }

            warmed_writers.fetch_add(1u, std::memory_order_release);

            while (
                !stop_writers.load(std::memory_order_acquire) &&
                !abort_requested()
            )
            {
                if (!mutate_once())
                {
                    progress_failed.store(true, std::memory_order_release);
                    break;
                }
            }
        });
    }

    std::vector<std::thread> readers;
    readers.reserve(reader_count);
    for (std::size_t reader = 0u; reader < reader_count; ++reader)
    {
        readers.emplace_back([&, reader]() noexcept
        {
            std::uint64_t local_reads = 0u;
            std::uint64_t local_retries = 0u;
            std::uint64_t local_starvations = 0u;
            const std::size_t writer =
                reader % ConcurrencyConfig::READER_WRITER_COUNT;

            reader_start.arrive_and_wait();

            if (!abort_requested())
            {
                for (
                    std::uint32_t i = 0u;
                    i < ConcurrencyConfig::STABLE_READS_PER_READER &&
                    !abort_requested();
                    ++i
                )
                {
                    std::uint32_t starvation_rounds = 0u;

                    for (;;)
                    {
                        const StableReadStatus status = StableReadOne(
                            backend, scenario, writer, local_retries);

                        if (status == StableReadStatus::SUCCESS)
                        {
                            ++local_reads;
                            break;
                        }

                        if (status == StableReadStatus::RETRY_LIMIT)
                        {
                            ++local_starvations;
                            ++starvation_rounds;

                            if (
                                starvation_rounds >=
                                ConcurrencyConfig::STABLE_READ_STARVATION_ROUND_LIMIT
                            )
                            {
                                progress_failed.store(
                                    true, std::memory_order_release);
                                break;
                            }

                            // Give a continuously publishing writer a chance to
                            // leave RESERVED and let this logical read resume.
                            std::this_thread::yield();
                            continue;
                        }

                        // BAD_CONTRACT or INVALID_PARENT are actual correctness
                        // failures and must never be masked as scheduler noise.
                        correctness_failed.store(true, std::memory_order_release);
                        break;
                    }
                }
            }

            stable_reads.fetch_add(local_reads, std::memory_order_relaxed);
            reader_retries.fetch_add(local_retries, std::memory_order_relaxed);
            reader_starvations.fetch_add(
                local_starvations, std::memory_order_relaxed);
            reader_finish.arrive_and_wait();
        });
    }

    writer_start.arrive_and_wait();
    while (
        warmed_writers.load(std::memory_order_acquire) <
            ConcurrencyConfig::READER_WRITER_COUNT &&
        !abort_requested()
    )
    {
        std::this_thread::yield();
    }

    if (abort_requested())
    {
        stop_writers.store(true, std::memory_order_release);
    }

    reader_start.arrive_and_wait();
    reader_finish.arrive_and_wait();
    stop_writers.store(true, std::memory_order_release);

    for (std::thread& reader : readers) reader.join();
    for (std::thread& writer : writers) writer.join();

    const std::uint64_t completed = stable_reads.load(std::memory_order_acquire);
    const std::uint64_t expected =
        reader_count * ConcurrencyConfig::STABLE_READS_PER_READER;
    const double elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());

    const bool final_verification = VerifyReaderScenario(backend, scenario);
    const bool correctness_ok =
        !correctness_failed.load(std::memory_order_acquire) &&
        final_verification;
    const bool progress_ok =
        !progress_failed.load(std::memory_order_acquire) &&
        completed == expected &&
        writer_exhaustions.load(std::memory_order_acquire) == 0u;

    return {
        correctness_ok && progress_ok,
        correctness_ok,
        progress_ok,
        final_verification,
        completed == 0u ? 0.0 : elapsed / static_cast<double>(completed),
        elapsed,
        completed,
        reader_retries.load(std::memory_order_acquire),
        reader_starvations.load(std::memory_order_acquire),
        writer_success.load(std::memory_order_acquire),
        writer_retries.load(std::memory_order_acquire),
        writer_exhaustions.load(std::memory_order_acquire)
    };
}

} // namespace BenchmarkCore

// -----------------------------------------------------------------------------
// Test 1: fair scaled bidirectional DAG + 1 KiB payload comparison.
// -----------------------------------------------------------------------------

namespace Test01_Baseline
{
using namespace BenchmarkCore;

inline bool RunScenario(const BenchmarkCase& config, std::size_t case_index)
{
    std::cout
        << "\n  CASE " << case_index << "/4"
        << "  N=" << config.NodeCount
        << "  K=" << static_cast<unsigned>(config.ParentCapacity) << '\n';

    bool construction_ok = true;
    auto construction = MeasurePair(
        [&]()
        {
            return MeasureNsPerOperation(1u, [&]() -> std::uint64_t
            {
                CompactVectorDAG backend{};
                const bool ok = BuildFullTest1Graph(backend, config);
                construction_ok = construction_ok && ok;
                return ok ? backend.ApproxStorageBytes() : 0u;
            });
        },
        [&]()
        {
            return MeasureNsPerOperation(1u, [&]() -> std::uint64_t
            {
                RuntimeAPCFabricBackend backend{};
                const bool ok = BuildFullTest1Graph(backend, config);
                construction_ok = construction_ok && ok;
                return ok ? backend.ApproxStorageBytes() : 0u;
            });
        });

    CompactVectorDAG vector_backend{};
    RuntimeAPCFabricBackend fabric_backend{};
    if (
        !construction_ok ||
        !BuildFullTest1Graph(vector_backend, config) ||
        !BuildFullTest1Graph(fabric_backend, config)
    )
    {
        std::cout << "    build: FAIL\n";
        return false;
    }

    const GraphProof vector_proof = ProveRuntimeCombinedDAG(vector_backend, config);
    const GraphProof fabric_proof = ProveRuntimeCombinedDAG(fabric_backend, config);
    bool ok = vector_proof.Passed() && fabric_proof.Passed();

    const std::uint64_t edge_count = EdgeCountPerAxis(
        config.NodeCount, config.ParentCapacity);
    const std::uint64_t parent_calls =
        config.NodeCount * static_cast<std::uint64_t>(config.ParentCapacity);
    const std::uint64_t reverse_calls = edge_count + config.NodeCount;
    const std::uint64_t payload_calls =
        config.NodeCount * TEST1_PAYLOAD_WORDS;

    auto parent_scan = [&](auto& backend, Axis axis)
    {
        const std::uint32_t rounds = RoundsForTarget(
            TARGET_TRAVERSAL_CALLS, parent_calls);
        return MeasureNsPerOperation(parent_calls * rounds, [&]()
        {
            std::uint64_t checksum = 0u;
            for (std::uint32_t r = 0u; r < rounds; ++r)
                for (std::size_t child = 0u; child < config.NodeCount; ++child)
                    for (std::uint8_t ordinal = 0u; ordinal < config.ParentCapacity; ++ordinal)
                    {
                        const BenchmarkReadResult read = BenchmarkFindParentCall(
                            backend, child, axis, ordinal, 1u);
                        checksum += read.Locator + static_cast<std::uint64_t>(read.Outcome);
                    }
            return checksum;
        });
    };

    auto reverse_scan = [&](auto& backend, Axis axis, bool payload)
    {
        const std::uint32_t rounds = RoundsForTarget(
            payload ? TARGET_GRAPH_PAYLOAD_CALLS : TARGET_TRAVERSAL_CALLS,
            reverse_calls);
        return MeasureNsPerOperation(reverse_calls * rounds, [&]()
        {
            std::uint64_t checksum = 0u;
            for (std::uint32_t r = 0u; r < rounds; ++r)
            {
                for (std::size_t parent = 0u; parent < config.NodeCount; ++parent)
                {
                    BenchmarkReadResult read = BenchmarkFindFirstChildCall(
                        backend, parent, axis, 1u);
                    while (read.IsFound())
                    {
                        checksum += read.Locator;
                        if (payload && read.HasNodeHint())
                        {
                            std::uint64_t value = 0u;
                            const std::uint32_t word = static_cast<std::uint32_t>(
                                read.NodeHint % TEST1_PAYLOAD_WORDS);
                            if (!backend.LoadPayload(read.NodeHint, word, value, false))
                                checksum ^= UINT64_MAX;
                            checksum ^= value;
                        }
                        read = BenchmarkFindNextChildCall(
                            backend, parent, axis, read.Locator, 1u);
                    }
                    checksum += static_cast<std::uint64_t>(read.Outcome);
                }
            }
            return checksum;
        });
    };

    auto payload_scan = [&](auto& backend, bool atomic)
    {
        const std::uint32_t rounds = RoundsForTarget(
            TARGET_PAYLOAD_CALLS, payload_calls);
        return MeasureNsPerOperation(payload_calls * rounds, [&]()
        {
            std::uint64_t checksum = 0u;
            for (std::uint32_t r = 0u; r < rounds; ++r)
                for (std::size_t node = 0u; node < config.NodeCount; ++node)
                    for (std::uint32_t word = 0u; word < TEST1_PAYLOAD_WORDS; ++word)
                    {
                        std::uint64_t value = 0u;
                        if (!backend.LoadPayload(node, word, value, atomic))
                            checksum ^= UINT64_MAX;
                        checksum += value;
                    }
            return checksum;
        });
    };

    auto replace_scan = [&](auto& backend, Axis axis)
    {
        const std::size_t child = static_cast<std::size_t>(config.ParentCapacity) + 1u;
        const std::size_t old_parent = axis == Axis::HORIZONTAL ? child - 1u : 0u;
        const std::size_t alternate = axis == Axis::HORIZONTAL
            ? child - static_cast<std::size_t>(config.ParentCapacity) - 1u
            : static_cast<std::size_t>(config.ParentCapacity);

        return MeasureNsPerOperation(
            static_cast<std::uint64_t>(TEST1_MUTATION_ROUNDS) * 2u,
            [&]()
            {
                std::uint64_t checksum = 0u;
                for (std::uint32_t r = 0u; r < TEST1_MUTATION_ROUNDS; ++r)
                {
                    const bool a = backend.ReplaceParent(
                        old_parent, alternate, child, axis, DEFAULT_MAX_TRIES);
                    const bool b = backend.ReplaceParent(
                        alternate, old_parent, child, axis, DEFAULT_MAX_TRIES);
                    checksum += static_cast<std::uint64_t>(a) +
                        static_cast<std::uint64_t>(b);
                }
                return checksum;
            });
    };

    const PairTiming h_parent = MeasurePair(
        [&] { return parent_scan(vector_backend, Axis::HORIZONTAL); },
        [&] { return parent_scan(fabric_backend, Axis::HORIZONTAL); });
    const PairTiming v_parent = MeasurePair(
        [&] { return parent_scan(vector_backend, Axis::VERTICAL); },
        [&] { return parent_scan(fabric_backend, Axis::VERTICAL); });
    const PairTiming h_reverse = MeasurePair(
        [&] { return reverse_scan(vector_backend, Axis::HORIZONTAL, false); },
        [&] { return reverse_scan(fabric_backend, Axis::HORIZONTAL, false); });
    const PairTiming v_reverse = MeasurePair(
        [&] { return reverse_scan(vector_backend, Axis::VERTICAL, false); },
        [&] { return reverse_scan(fabric_backend, Axis::VERTICAL, false); });
    const PairTiming direct = MeasurePair(
        [&] { return payload_scan(vector_backend, false); },
        [&] { return payload_scan(fabric_backend, false); });
    const PairTiming atomic = MeasurePair(
        [&] { return payload_scan(vector_backend, true); },
        [&] { return payload_scan(fabric_backend, true); });
    const PairTiming graph_payload = MeasurePair(
        [&] { return reverse_scan(vector_backend, Axis::HORIZONTAL, true); },
        [&] { return reverse_scan(fabric_backend, Axis::HORIZONTAL, true); });
    const PairTiming h_replace = MeasurePair(
        [&] { return replace_scan(vector_backend, Axis::HORIZONTAL); },
        [&] { return replace_scan(fabric_backend, Axis::HORIZONTAL); });
    const PairTiming v_replace = MeasurePair(
        [&] { return replace_scan(vector_backend, Axis::VERTICAL); },
        [&] { return replace_scan(fabric_backend, Axis::VERTICAL); });

    std::cout
        << "    edges/axis=" << edge_count
        << "  payload/node=" << TEST1_PAYLOAD_WORDS * sizeof(std::uint64_t)
        << " B\n"
        << "    storage vector/Fabric="
        << vector_backend.ApproxStorageBytes() << "/"
        << fabric_backend.ApproxStorageBytes() << " B"
        << "  Fabric/vector="
        << std::fixed << std::setprecision(2)
        << Ratio(
            static_cast<double>(fabric_backend.ApproxStorageBytes()),
            static_cast<double>(vector_backend.ApproxStorageBytes()))
        << "x\n";

    PrintPairRow("construction", construction);
    PrintPairRow("H parent scan", h_parent);
    PrintPairRow("V parent scan", v_parent);
    PrintPairRow("H reverse-child scan", h_reverse);
    PrintPairRow("V reverse-child scan", v_reverse);
    PrintPairRow("payload direct read", direct);
    PrintPairRow("payload atomic_ref read", atomic);
    PrintPairRow("H child + payload", graph_payload);
    PrintPairRow("H parent replace", h_replace);
    PrintPairRow("V parent replace", v_replace);

    const GraphProof vector_after = ProveRuntimeCombinedDAG(vector_backend, config);
    const GraphProof fabric_after = ProveRuntimeCombinedDAG(fabric_backend, config);
    ok = ok && vector_after.Passed() && fabric_after.Passed();
    std::cout << "    integrity=" << (ok ? "PASS" : "FAIL") << '\n';
    return ok;
}

inline Result Run(const std::array<BenchmarkCase, 4u>& cases)
{
    Banner("TEST 1 - SCALED FAIR BIDIRECTIONAL DAG / 1 KiB PAYLOAD COMPARISON");
    std::cout
        << "Compact vector DAG vs minimal single-region SuperNova Fabric.\n"
        << "Each case uses the same N, K, fully populated H/V bounded topology, and\n"
        << "exactly 128 x uint64_t (1024 B) payload per node. Four measured samples\n"
        << "are taken per row with backend order alternated.\n";

    bool ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        ok = RunScenario(cases[i], i + 1u) && ok;

    std::cout << "\nTEST 1 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test01_Baseline

// -----------------------------------------------------------------------------
// Test 2: row-local-lock vs optimistic structural mutation.
// -----------------------------------------------------------------------------

namespace Test02_Contention
{
using namespace BenchmarkCore;

inline bool RunCase(
    const BenchmarkCase& config,
    MutationLocality locality,
    std::size_t max_writers,
    std::size_t case_index)
{
    const auto maybe_scenario = MakeMutationScenario(config, locality, max_writers);
    if (!maybe_scenario.has_value()) return false;
    const MutationScenario scenario = maybe_scenario.value();
    const MutationSchedule schedule = BuildMutationSchedule(scenario);

    std::cout
        << "\n  CASE " << case_index << "/4"
        << "  N=" << config.NodeCount
        << "  K=" << static_cast<unsigned>(config.ParentCapacity)
        << "  parent-pool=" << scenario.ParentCount << '\n';

    bool all_ok = true;
    for (std::size_t writer_count = 1u; writer_count <= max_writers; ++writer_count)
    {
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> row_ns{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> fabric_ns{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> retry_rate{};
        bool row_ok = true;

        for (std::size_t run = 0u; run < ConcurrencyConfig::MEASURED_RUNS; ++run)
        {
            RowLockedVectorDAG<std::mutex, false> row_backend{};
            RuntimeAPCFabricBackend fabric_backend{};
            if (
                !BuildMutationBackend(row_backend, scenario, writer_count) ||
                !BuildMutationBackend(fabric_backend, scenario, writer_count)
            )
            {
                return false;
            }

            MutationSweepResult row_result{};
            MutationSweepResult fabric_result{};
            if ((run & 1u) == 0u)
            {
                row_result = RunMutationWorkers(
                    row_backend, scenario, schedule, writer_count);
                fabric_result = RunMutationWorkers(
                    fabric_backend, scenario, schedule, writer_count);
            }
            else
            {
                fabric_result = RunMutationWorkers(
                    fabric_backend, scenario, schedule, writer_count);
                row_result = RunMutationWorkers(
                    row_backend, scenario, schedule, writer_count);
            }

            const bool integrity =
                row_result.Ok && fabric_result.Ok &&
                VerifyMutationScenario(
                    row_backend, scenario, schedule, writer_count) &&
                VerifyMutationScenario(
                    fabric_backend, scenario, schedule, writer_count);
            row_ok = row_ok && integrity;
            row_ns[run] = row_result.NsPerSuccess;
            fabric_ns[run] = fabric_result.NsPerSuccess;
            retry_rate[run] = fabric_result.Success == 0u ? 0.0 :
                static_cast<double>(fabric_result.Retries) /
                static_cast<double>(fabric_result.Success);
        }

        const double row_median = Median(row_ns);
        const double fabric_median = Median(fabric_ns);
        const double retries = Median(retry_rate);
        const double row_mops = row_median > 0.0
            ? ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS / row_median
            : 0.0;
        const double fabric_mops = fabric_median > 0.0
            ? ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS / fabric_median
            : 0.0;

        all_ok = all_ok && row_ok;
        std::cout
            << "    threads=" << std::setw(2) << writer_count
            << "  rowlock=" << std::setw(9) << std::fixed << std::setprecision(2)
            << row_median << " ns/op (" << std::setw(7) << row_mops << " M/s)"
            << "  Fabric=" << std::setw(9) << fabric_median
            << " ns/op (" << std::setw(7) << fabric_mops << " M/s)"
            << "  Fabric/row=" << std::setw(6)
            << Ratio(fabric_median, row_median) << "x"
            << "  retry/success=" << std::setw(8) << std::setprecision(4)
            << retries
            << "  " << (row_ok ? "PASS" : "FAIL") << '\n';
    }
    return all_ok;
}

inline Result Run(
    const std::array<BenchmarkCase, 4u>& cases,
    std::size_t max_writers)
{
    Banner("TEST 2A - HOTSPOT STRUCTURAL MUTATION");
    std::cout
        << "Each writer owns one child; all writers contend on the same two H and two V\n"
        << "parents. The baseline locks only participating rows. Every writer count from\n"
        << "1 through usable_threads-2 is printed for each of the four (N,K) cases.\n";

    bool hotspot_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        hotspot_ok = RunCase(
            cases[i], MutationLocality::HOTSPOT, max_writers, i + 1u) && hotspot_ok;
    std::cout << "\nTEST 2A OVERALL: " << (hotspot_ok ? "PASS" : "FAIL") << '\n';

    Banner("TEST 2B - DISTRIBUTED STRUCTURAL MUTATION");
    std::cout
        << "Each writer owns one child and follows the same deterministic random schedule\n"
        << "for both backends. The parent pool is min(100, legal predecessor nodes).\n"
        << "Test 2B now uses the same 1..(usable_threads-2) sweep as Test 2A.\n";

    bool distributed_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        distributed_ok = RunCase(
            cases[i], MutationLocality::DISTRIBUTED, max_writers, i + 1u) && distributed_ok;
    std::cout << "\nTEST 2B OVERALL: " << (distributed_ok ? "PASS" : "FAIL") << '\n';

    const bool ok = hotspot_ok && distributed_ok;
    std::cout << "\nTEST 2 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test02_Contention

// -----------------------------------------------------------------------------
// Test 3: stable readers with exactly two continuously active writers.
// -----------------------------------------------------------------------------

namespace Test03_ReaderWriter
{
using namespace BenchmarkCore;

inline bool RunCase(
    const BenchmarkCase& config,
    bool distributed,
    std::size_t max_readers,
    std::size_t case_index)
{
    const auto maybe_scenario = MakeReaderScenario(config, distributed);
    if (!maybe_scenario.has_value()) return false;
    const ReaderScenario scenario = maybe_scenario.value();
    const ParentSchedule schedule = BuildParentSchedule(scenario);

    std::cout
        << "\n  CASE " << case_index << "/4"
        << "  N=" << config.NodeCount
        << "  K=" << static_cast<unsigned>(config.ParentCapacity)
        << "  parent-pool=" << scenario.ParentCount << '\n';

    bool all_ok = true;
    for (std::size_t reader_count = 1u; reader_count <= max_readers; ++reader_count)
    {
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> row_ns{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> fabric_ns{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> retry_rate{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> row_starvations{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> fabric_starvations{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> row_writer_mops{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> fabric_writer_mops{};
        bool row_ok = true;
        bool row_correctness_ok = true;
        bool fabric_correctness_ok = true;
        bool row_progress_ok = true;
        bool fabric_progress_ok = true;

        for (std::size_t run = 0u; run < ConcurrencyConfig::MEASURED_RUNS; ++run)
        {
            RowLockedVectorDAG<std::shared_mutex, true> row_backend{};
            RuntimeAPCFabricBackend fabric_backend{};
            if (
                !BuildReaderBackend(row_backend, scenario) ||
                !BuildReaderBackend(fabric_backend, scenario)
            )
            {
                return false;
            }

            ReaderSweepResult row_result{};
            ReaderSweepResult fabric_result{};
            if ((run & 1u) == 0u)
            {
                row_result = RunReadersWithWriters(
                    row_backend, scenario, schedule, reader_count);
                fabric_result = RunReadersWithWriters(
                    fabric_backend, scenario, schedule, reader_count);
            }
            else
            {
                fabric_result = RunReadersWithWriters(
                    fabric_backend, scenario, schedule, reader_count);
                row_result = RunReadersWithWriters(
                    row_backend, scenario, schedule, reader_count);
            }

            row_ok = row_ok && row_result.Ok && fabric_result.Ok;
            row_correctness_ok =
                row_correctness_ok && row_result.CorrectnessOk;
            fabric_correctness_ok =
                fabric_correctness_ok && fabric_result.CorrectnessOk;
            row_progress_ok =
                row_progress_ok && row_result.ProgressOk;
            fabric_progress_ok =
                fabric_progress_ok && fabric_result.ProgressOk;

            row_ns[run] = row_result.NsPerStableRead;
            fabric_ns[run] = fabric_result.NsPerStableRead;
            retry_rate[run] = fabric_result.StableReads == 0u ? 0.0 :
                static_cast<double>(fabric_result.ReaderRetries) /
                static_cast<double>(fabric_result.StableReads);
            row_starvations[run] =
                static_cast<double>(row_result.ReaderStarvations);
            fabric_starvations[run] =
                static_cast<double>(fabric_result.ReaderStarvations);

            row_writer_mops[run] = row_result.ElapsedNs > 0.0
                ? static_cast<double>(row_result.WriterSuccess) *
                    ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS /
                    row_result.ElapsedNs
                : 0.0;
            fabric_writer_mops[run] = fabric_result.ElapsedNs > 0.0
                ? static_cast<double>(fabric_result.WriterSuccess) *
                    ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS /
                    fabric_result.ElapsedNs
                : 0.0;
        }

        const double row_median = Median(row_ns);
        const double fabric_median = Median(fabric_ns);
        const double retries = Median(retry_rate);
        const double row_read_mops = row_median > 0.0
            ? ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS / row_median
            : 0.0;
        const double fabric_read_mops = fabric_median > 0.0
            ? ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS / fabric_median
            : 0.0;

        all_ok = all_ok && row_ok;
        std::cout
            << "    readers=" << std::setw(2) << reader_count
            << "  row-rw=" << std::setw(9) << std::fixed << std::setprecision(2)
            << row_median << " ns (" << std::setw(7) << row_read_mops << " M/s)"
            << "  Fabric=" << std::setw(9) << fabric_median
            << " ns (" << std::setw(7) << fabric_read_mops << " M/s)"
            << "  Fabric/row=" << std::setw(6)
            << Ratio(fabric_median, row_median) << "x"
            << "  retry/read=" << std::setw(8) << std::setprecision(4) << retries
            << "  starve row/Fabric=" << std::setprecision(0)
            << Median(row_starvations) << "/" << Median(fabric_starvations)
            << "  writers M/s row/Fabric=" << std::setprecision(2)
            << Median(row_writer_mops) << "/" << Median(fabric_writer_mops)
            << "  correctness="
            << (row_correctness_ok && fabric_correctness_ok ? "PASS" : "FAIL")
            << "  progress="
            << (row_progress_ok && fabric_progress_ok ? "PASS" : "FAIL")
            << "  " << (row_ok ? "PASS" : "FAIL") << '\n';
    }
    return all_ok;
}

inline Result Run(
    const std::array<BenchmarkCase, 4u>& cases,
    std::size_t max_readers)
{
    Banner("TEST 3A - HOTSPOT STABLE READS WITH TWO ACTIVE WRITERS");
    std::cout
        << "Exactly two writers remain active: H toggles 0<->1 and V toggles 2<->3\n"
        << "on the same child. Readers sweep 1..(usable_threads-2). The vector baseline\n"
        << "uses row-local shared_mutex shared reads and exclusive writer ownership.\n";

    bool hotspot_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        hotspot_ok = RunCase(cases[i], false, max_readers, i + 1u) && hotspot_ok;
    std::cout << "\nTEST 3A OVERALL: " << (hotspot_ok ? "PASS" : "FAIL") << '\n';

    Banner("TEST 3B - DISTRIBUTED STABLE READS WITH TWO ACTIVE WRITERS");
    std::cout
        << "Two writers own separate child relations and mutate across up to 100 legal\n"
        << "parents. Readers sweep every count from 1 through usable_threads-2; APC RETRY\n"
        << "outcomes are retried and never counted as successful stable reads.\n";

    bool distributed_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        distributed_ok = RunCase(cases[i], true, max_readers, i + 1u) && distributed_ok;
    std::cout << "\nTEST 3B OVERALL: " << (distributed_ok ? "PASS" : "FAIL") << '\n';

    const bool ok = hotspot_ok && distributed_ok;
    std::cout << "\nTEST 3 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
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


inline bool ValidateRunArguments(
    std::size_t lower_node_count,
    std::size_t higher_node_count,
    std::size_t lower_parent_capacity,
    std::size_t higher_parent_capacity,
    std::size_t usable_thread_count)
{
    const std::size_t system_threads =
        static_cast<std::size_t>(std::thread::hardware_concurrency());

    const auto fail = [](const char* reason)
    {
        std::cout << "\nCONFIGURATION ERROR: " << reason << '\n';
        return false;
    };

    if (system_threads < 3u)
        return fail("hardware_concurrency must report at least 3 threads");
    if (usable_thread_count < 3u)
        return fail("usable_thread_count must be at least 3");
    if (usable_thread_count > system_threads)
        return fail("usable_thread_count cannot exceed hardware_concurrency");
    if (lower_node_count == 0u || higher_node_count < lower_node_count)
        return fail("node bounds must satisfy 0 < lower <= higher");
    if (
        lower_parent_capacity == 0u ||
        higher_parent_capacity < lower_parent_capacity ||
        higher_parent_capacity > ADS::COMPILED_MAX_DIRECT_PARENTS_PER_AXIS
    )
        return fail("parent bounds must satisfy 0 < lower <= higher <= 64");
    if (higher_node_count > UINT32_MAX)
        return fail("node count exceeds the Fabric uint32 slot domain");
    if (
        higher_node_count >
        UINT32_MAX / higher_parent_capacity
    )
        return fail("N*K exceeds the uint32 relation-locator domain");

    const std::size_t worker_limit = usable_thread_count - 2u;
    const std::size_t minimum_nodes = std::max(
        higher_parent_capacity + 2u,
        worker_limit + 4u);
    if (lower_node_count < minimum_nodes)
        return fail("lower node bound is too small for K and the requested thread sweep");

    return true;
}

inline int Run(
    std::size_t lower_node_count = 100u,
    std::size_t higher_node_count = 10'000u,
    std::size_t lower_parent_capacity = 4u,
    std::size_t higher_parent_capacity = 32u,
    std::size_t usable_thread_count = 18u)
{
    PrintBenchmarkEnvironment();
    if (!ValidateRunArguments(
        lower_node_count,
        higher_node_count,
        lower_parent_capacity,
        higher_parent_capacity,
        usable_thread_count))
    {
        return 1;
    }

    const std::array<BenchmarkCase, 4u> cases = MakeBenchmarkCases(
        lower_node_count,
        higher_node_count,
        static_cast<std::uint8_t>(lower_parent_capacity),
        static_cast<std::uint8_t>(higher_parent_capacity));
    const std::size_t concurrent_threads = usable_thread_count - 2u;

    std::cout
        << "\nSUITE CONFIGURATION\n"
        << "  node bounds             : " << lower_node_count
        << " .. " << higher_node_count << '\n'
        << "  parent-capacity bounds  : " << lower_parent_capacity
        << " .. " << higher_parent_capacity << '\n'
        << "  usable threads          : " << usable_thread_count << '\n'
        << "  Test 2 writer sweep     : 1.." << concurrent_threads << '\n'
        << "  Test 3 reader sweep     : 1.." << concurrent_threads
        << " + 2 fixed writers\n"
        << "  measured samples/point  : "
        << ConcurrencyConfig::MEASURED_RUNS << '\n';

    const std::array<std::pair<const char*, Result>, 8u> results{{
        {"Test 1 - scaled fair bidirectional DAG benchmark", Test01_Baseline::Run(cases)},
        {"Test 2 - hotspot + distributed mutation", Test02_Contention::Run(cases, concurrent_threads)},
        {"Test 3 - hotspot + distributed readers", Test03_ReaderWriter::Run(cases, concurrent_threads)},
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
            << "  " << std::left << std::setw(54) << name
            << ResultName(result) << '\n';
        if (result == Result::FAIL) ++failures;
    }
    std::cout
        << "\n  failures: " << failures
        << "\n================================================================================\n";
    return failures == 0u ? 0 : 1;
}

inline int RunAll(
    std::size_t lower_node_count = 100u,
    std::size_t higher_node_count = 10'000u,
    std::size_t lower_parent_capacity = 4u,
    std::size_t higher_parent_capacity = 32u,
    std::size_t usable_thread_count = 18u)
{
    return Run(
        lower_node_count,
        higher_node_count,
        lower_parent_capacity,
        higher_parent_capacity,
        usable_thread_count);
}

} // namespace APCDAGTests





