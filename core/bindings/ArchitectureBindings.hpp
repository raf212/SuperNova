// #pragma once

// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

// #include <atomic>
// #include <cstdint>
// #include <memory>
// #include <optional>
// #include <stdexcept>
// #include <string>
// #include <utility>
// #include <vector>

// #include "../headers/NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

// namespace py = pybind11;

// namespace atomiccim::python
// {
//     using namespace BidirectionalInMemGraph;

//     using NativeAPC = AdaptivePackedCellContainer;
//     using NativeFabric = VagueTemoraryPremativeFabric;
//     using IAB = InstallAxisToBuffer;
//     using Axis = IAB::BidirectionalAxis;
//     using Inheritance = IAB::DescOfInharitance;
//     using NavigationStatus = FabricToAPCLinker::SeqLockedOperation;
//     using Layout = LayoutBoundsOrchestrator::LayoutSpanAndPercentageCarrier;
//     using DataTypes = SchemaDefinition::InitialRegionalDtypeConf;
//     using Protocols = SchemaDefinition::InitialRegionalProtocol;
//     using RegionProtocol = SchemaDefinition::SchemaProtocols;
//     using RegionDataType = SchemaDefinition::DataTypeOfMacroColumn;

//     class PythonAPC;
//     class PythonFabric;
//     struct RetainedHandleNode;

//     enum class BindingPhase : uint8_t
//     {
//         DETACHED = 0u,
//         CREATING = 1u,
//         BOUND = 2u,
//         RETIRED = 3u
//     };

//     struct FabricEpoch final
//     {
//         NativeFabric Native{};
//         std::atomic<bool> Accepting{false};
//         uint32_t SlotCount{0u};
//         std::unique_ptr<std::atomic<PythonAPC*>[]> CurrentHandles{};
//         std::atomic<RetainedHandleNode*> RetainedHead{nullptr};

//         explicit FabricEpoch(uint32_t slot_count)
//             : SlotCount(slot_count),
//               CurrentHandles(
//                   slot_count == 0u
//                       ? nullptr
//                         : std::make_unique<std::atomic<PythonAPC*>[]>(slot_count)
//               )
//         {
//             for (uint32_t slot = 0u; slot < SlotCount; ++slot)
//             {
//                 CurrentHandles[slot].store(nullptr, std::memory_order_relaxed);
//             }
//         }

//         FabricEpoch(const FabricEpoch&) = delete;
//         FabricEpoch& operator=(const FabricEpoch&) = delete;
//         ~FabricEpoch() noexcept;

//         void Retain(RetainedHandleNode* node) noexcept;

//         [[nodiscard]] std::shared_ptr<PythonAPC> Resolve(
//             NativeAPC* native
//         ) noexcept;
//     };

//     struct APCBinding final
//     {
//         std::weak_ptr<FabricEpoch> Epoch{};
//         uint32_t Slot{ADS::APC_INDEX_BOUND_SENTINAL};
//         BindingPhase Phase{BindingPhase::DETACHED};
//         APCBinding* Next{nullptr};
//     };

//     struct APCSnapshot final
//     {
//         std::shared_ptr<FabricEpoch> Epoch{};
//         std::shared_ptr<NativeAPC> Native{};
//         const APCBinding* Binding{nullptr};

//         [[nodiscard]] explicit operator bool() const noexcept
//         {
//             return
//                 Epoch &&
//                 Native &&
//                 Binding != nullptr &&
//                 Binding->Phase == BindingPhase::BOUND;
//         }
//     };

//     struct NavigationResult final
//     {
//         NavigationStatus Status{NavigationStatus::NONE};
//         std::shared_ptr<PythonAPC> APC{};

//         [[nodiscard]] bool Found() const noexcept
//         {
//             return Status == NavigationStatus::FOUND && static_cast<bool>(APC);
//         }
//     };

//     template<class Operation>
//     bool MutateTwoAPCs(PythonAPC& first, PythonAPC& second, Operation&& operation);

//     template<class Operation>
//     bool MutateOneAPC(PythonAPC& apc, Operation&& operation);

//     class PythonAPC final : public std::enable_shared_from_this<PythonAPC>
//     {
//         friend class PythonFabric;
//         friend struct FabricEpoch;

//         template<class Operation>
//         friend bool MutateTwoAPCs(PythonAPC&, PythonAPC&, Operation&&);

//         template<class Operation>
//         friend bool MutateOneAPC(PythonAPC&, Operation&&);

//     private:
//         std::shared_ptr<NativeAPC> Native_{std::make_shared<NativeAPC>()};
//         std::atomic<APCBinding*> Binding_{nullptr};
//         std::atomic<APCBinding*> BindingRecords_{nullptr};

//         void RetainBindingRecord_(APCBinding* record) noexcept
//         {
//             APCBinding* observed =
//                 BindingRecords_.load(std::memory_order_acquire);
//             do
//             {
//                 record->Next = observed;
//             }
//             while (!BindingRecords_.compare_exchange_weak(
//                 observed,
//                 record,
//                 std::memory_order_release,
//                 std::memory_order_acquire
//             ));
//         }

//         [[nodiscard]] APCSnapshot Snapshot_() const noexcept
//         {
//             APCBinding* binding =
//                 Binding_.load(std::memory_order_acquire);

//             if (!binding || binding->Phase != BindingPhase::BOUND)
//             {
//                 return {};
//             }

//             std::shared_ptr<FabricEpoch> epoch = binding->Epoch.lock();
//             if (!epoch)
//             {
//                 return {};
//             }

//             return APCSnapshot{
//                 std::move(epoch),
//                 Native_,
//                 binding
//             };
//         }

//         [[nodiscard]] static bool SnapshotUsable_(
//             const APCSnapshot& snapshot
//         ) noexcept
//         {
//             return
//                 static_cast<bool>(snapshot) &&
//                 snapshot.Epoch->Accepting.load(std::memory_order_acquire);
//         }

//     public:
//         PythonAPC()
//         {
//             APCBinding* detached = new APCBinding{};
//             RetainBindingRecord_(detached);
//             Binding_.store(detached, std::memory_order_relaxed);
//         }

//         ~PythonAPC() noexcept
//         {
//             APCBinding* record =
//                 BindingRecords_.exchange(nullptr, std::memory_order_acq_rel);
//             while (record)
//             {
//                 APCBinding* next = record->Next;
//                 delete record;
//                 record = next;
//             }
//         }

//         PythonAPC(const PythonAPC&) = delete;
//         PythonAPC& operator=(const PythonAPC&) = delete;

//         [[nodiscard]] uint32_t GetThisSlotIdx() const noexcept
//         {
//             const APCBinding* binding =
//                 Binding_.load(std::memory_order_acquire);

//             if (
//                 !binding ||
//                 (
//                     binding->Phase != BindingPhase::BOUND &&
//                     binding->Phase != BindingPhase::RETIRED
//                 )
//             )
//             {
//                 return ADS::APC_INDEX_BOUND_SENTINAL;
//             }
//             return binding->Slot;
//         }

//         [[nodiscard]] bool IsActiveAPC() const noexcept;

//         bool AttachSiblingOrChild(
//             PythonAPC& sibling,
//             Axis axis,
//             Inheritance inheritance,
//             uint32_t max_tries = DEFAULT_MAX_TRIES
//         );

//         bool AttachMeToAnother(
//             PythonAPC& sibling,
//             Axis axis,
//             Inheritance inheritance,
//             uint32_t max_tries = DEFAULT_MAX_TRIES
//         );

//         bool DetachMyChild(
//             PythonAPC& child,
//             Axis axis,
//             uint32_t max_tries = DEFAULT_MAX_TRIES
//         );

//         bool DetachMeFromAnotherEdge(
//             Axis axis,
//             uint32_t max_tries = DEFAULT_MAX_TRIES
//         );

//         bool DetachAndReAttachMeToThisParent(
//             PythonAPC& root_parent,
//             Axis axis,
//             uint32_t max_tries = DEFAULT_MAX_TRIES
//         );

//         bool DetachAndReattachMeAsEquivelentSibbling(
//             PythonAPC& sibling,
//             Axis axis,
//             uint32_t max_tries = DEFAULT_MAX_TRIES
//         );

//         NavigationResult FindPrevious(
//             Axis axis,
//             uint32_t max_tries = NativeAPC::REALTION_FIND_TRIES
//         );

//         NavigationResult FindMyNext(
//             Axis axis,
//             Inheritance inheritance,
//             uint32_t max_tries = NativeAPC::REALTION_FIND_TRIES
//         );

//         bool Retire(uint32_t max_tries = DEFAULT_MAX_TRIES);

//         template<class DType>
//         std::optional<RegionView<DType>> BuildNativeView(
//             MacroColumnOfAPC column,
//             std::shared_ptr<FabricEpoch>& epoch
//         );

//         template<class DType>
//         bool ZeroARegion(MacroColumnOfAPC column);
//     };

//     struct RetainedHandleNode final
//     {
//         std::shared_ptr<PythonAPC> Handle{};
//         RetainedHandleNode* Next{nullptr};

//         explicit RetainedHandleNode(std::shared_ptr<PythonAPC> handle) noexcept
//             : Handle(std::move(handle))
//         {}
//     };

//     inline FabricEpoch::~FabricEpoch() noexcept
//     {
//         Accepting.store(false, std::memory_order_release);

//         // Native memory is released only after every operation/view that held
//         // this epoch has finished. Retained handles keep their native wrappers
//         // alive until after the raw-pointer table and slab have shut down.
//         Native.ShutDownFabric();

//         for (uint32_t slot = 0u; slot < SlotCount; ++slot)
//         {
//             CurrentHandles[slot].store(nullptr, std::memory_order_relaxed);
//         }

//         RetainedHandleNode* node =
//             RetainedHead.exchange(nullptr, std::memory_order_acq_rel);

//         // Break the persistent list iteratively so a long-running Fabric does
//         // not recurse through thousands of generations during shutdown.
//         while (node)
//         {
//             RetainedHandleNode* next = node->Next;
//             delete node;
//             node = next;
//         }
//     }

//     inline void FabricEpoch::Retain(RetainedHandleNode* node) noexcept
//     {
//         RetainedHandleNode* observed =
//             RetainedHead.load(std::memory_order_acquire);

//         do
//         {
//             node->Next = observed;
//         }
//         while (!RetainedHead.compare_exchange_weak(
//             observed,
//             node,
//             std::memory_order_release,
//             std::memory_order_acquire
//         ));
//     }

//     inline std::shared_ptr<PythonAPC> FabricEpoch::Resolve(
//         NativeAPC* native
//     ) noexcept
//     {
//         if (
//             !native ||
//             !Accepting.load(std::memory_order_acquire)
//         )
//         {
//             return {};
//         }

//         // Every native wrapper is retained until this epoch is destroyed, so
//         // dereferencing the navigation pointer here is safe even when its slot
//         // was concurrently retired and reused.
//         const uint32_t slot = native->GetThisSlotIdx();
//         if (slot >= SlotCount)
//         {
//             return {};
//         }

//         PythonAPC* handle =
//             CurrentHandles[slot].load(std::memory_order_acquire);

//         if (!handle || handle->Native_.get() != native)
//         {
//             return {};
//         }

//         const APCSnapshot snapshot = handle->Snapshot_();
//         if (!snapshot || snapshot.Epoch.get() != this)
//         {
//             return {};
//         }
//         return handle->weak_from_this().lock();
//     }

//     class PythonFabric final
//     {
//         friend class PythonAPC;

//     private:
//         std::atomic<std::shared_ptr<FabricEpoch>> Current_{};

//     public:
//         PythonFabric() = default;
//         PythonFabric(const PythonFabric&) = delete;
//         PythonFabric& operator=(const PythonFabric&) = delete;

//         ~PythonFabric() noexcept
//         {
//             ShutDownFabric();
//         }

//         bool InitializeFabric(
//             uint32_t slot_count,
//             uint32_t slot_cell_count = MINIMUM_APC_CELL_COUNT
//         )
//         {
//             if (
//                 slot_count == 0u ||
//                 Current_.load(std::memory_order_acquire)
//             )
//             {
//                 return false;
//             }

//             auto candidate = std::make_shared<FabricEpoch>(slot_count);
//             if (!candidate->Native.InitializeFabric(
//                     slot_count,
//                     slot_cell_count
//                 ))
//             {
//                 return false;
//             }

//             candidate->Accepting.store(true, std::memory_order_release);

//             std::shared_ptr<FabricEpoch> expected{};
//             if (!Current_.compare_exchange_strong(
//                     expected,
//                     candidate,
//                     std::memory_order_release,
//                     std::memory_order_acquire
//                 ))
//             {
//                 candidate->Accepting.store(false, std::memory_order_release);
//                 return false;
//             }
//             return true;
//         }

//         void ShutDownFabric() noexcept
//         {
//             std::shared_ptr<FabricEpoch> epoch =
//                 Current_.exchange({}, std::memory_order_acq_rel);

//             if (epoch)
//             {
//                 // This is logical shutdown. Physical slab destruction is
//                 // deferred until already-started operations/views drop epoch.
//                 epoch->Accepting.store(false, std::memory_order_release);
//             }
//         }

//         [[nodiscard]] bool IsFabricActive() noexcept
//         {
//             const std::shared_ptr<FabricEpoch> epoch =
//                 Current_.load(std::memory_order_acquire);

//             return
//                 epoch &&
//                 epoch->Accepting.load(std::memory_order_acquire) &&
//                 epoch->Native.IsFabricActive();
//         }

//         bool CreateAPC(
//             const std::shared_ptr<PythonAPC>& desired_apc,
//             bool wants_horizontal_root,
//             bool wants_vertical_root,
//             const Layout& layout,
//             const DataTypes& dtype,
//             const Protocols& protocol,
//             uint8_t version,
//             uint32_t internal_max_tries = DEFAULT_MAX_TRIES
//         )
//         {
//             if (!desired_apc)
//             {
//                 return false;
//             }

//             std::shared_ptr<FabricEpoch> epoch =
//                 Current_.load(std::memory_order_acquire);

//             if (
//                 !epoch ||
//                 !epoch->Accepting.load(std::memory_order_acquire) ||
//                 !epoch->Native.IsFabricActive()
//             )
//             {
//                 return false;
//             }

//             // Allocate every publication record before claiming or mutating
//             // the native APC, so allocation failure cannot strand CREATING.
//             auto claiming_owner = std::make_unique<APCBinding>();
//             claiming_owner->Epoch = epoch;
//             claiming_owner->Phase = BindingPhase::CREATING;

//             auto detached_owner = std::make_unique<APCBinding>();

//             auto bound_owner = std::make_unique<APCBinding>();
//             bound_owner->Epoch = epoch;
//             bound_owner->Phase = BindingPhase::BOUND;

//             auto retired_owner = std::make_unique<APCBinding>();
//             retired_owner->Epoch = epoch;
//             retired_owner->Phase = BindingPhase::RETIRED;

//             auto retained_node =
//                 std::make_unique<RetainedHandleNode>(desired_apc);

//             APCBinding* claiming = claiming_owner.get();
//             APCBinding* detached = detached_owner.get();
//             APCBinding* bound = bound_owner.get();
//             APCBinding* retired = retired_owner.get();

//             desired_apc->RetainBindingRecord_(claiming_owner.release());
//             desired_apc->RetainBindingRecord_(detached_owner.release());
//             desired_apc->RetainBindingRecord_(bound_owner.release());
//             desired_apc->RetainBindingRecord_(retired_owner.release());

//             APCBinding* observed =
//                 desired_apc->Binding_.load(std::memory_order_acquire);

//             if (
//                 !observed ||
//                 observed->Phase != BindingPhase::DETACHED ||
//                 !desired_apc->Binding_.compare_exchange_strong(
//                     observed,
//                     claiming,
//                     std::memory_order_acq_rel,
//                     std::memory_order_acquire
//                 )
//             )
//             {
//                 return false;
//             }

//             if (
//                 !epoch->Accepting.load(std::memory_order_acquire) ||
//                 desired_apc->Native_->IsActiveAPC()
//             )
//             {
//                 desired_apc->Binding_.store(detached, std::memory_order_release);
//                 return false;
//             }

//             if (!epoch->Native.CreateAPC(
//                     *desired_apc->Native_,
//                     wants_horizontal_root,
//                     wants_vertical_root,
//                     layout,
//                     dtype,
//                     protocol,
//                     version,
//                     internal_max_tries
//                 ))
//             {
//                 desired_apc->Binding_.store(detached, std::memory_order_release);
//                 return false;
//             }

//             const uint32_t slot = desired_apc->Native_->GetThisSlotIdx();
//             bound->Slot = slot;
//             retired->Slot = slot;

//             // The native table returns raw NativeAPC pointers. Retaining every
//             // wrapper for the epoch is the lock-free reclamation boundary that
//             // makes a pointer returned immediately before retirement safe to
//             // resolve without a mutable registry or a use-after-free.
//             epoch->Retain(retained_node.release());

//             if (slot >= epoch->SlotCount)
//             {
//                 (void)desired_apc->Native_->Retire(internal_max_tries);
//                 desired_apc->Binding_.store(retired, std::memory_order_release);
//                 return false;
//             }

//             epoch->CurrentHandles[slot].exchange(
//                 desired_apc.get(),
//                 std::memory_order_acq_rel
//             );
//             desired_apc->Binding_.store(bound, std::memory_order_release);

//             if (!epoch->Accepting.load(std::memory_order_acquire))
//             {
//                 (void)desired_apc->Native_->Retire(internal_max_tries);
//                 desired_apc->Binding_.store(retired, std::memory_order_release);

//                 PythonAPC* current =
//                     epoch->CurrentHandles[slot].load(std::memory_order_acquire);
//                 while (
//                     current &&
//                     current == desired_apc.get() &&
//                     !epoch->CurrentHandles[slot].compare_exchange_weak(
//                         current,
//                         nullptr,
//                         std::memory_order_acq_rel,
//                         std::memory_order_acquire
//                     )
//                 )
//                 {}
//                 return false;
//             }

//             return true;
//         }

//         std::shared_ptr<PythonAPC> CreateAPCAndReturn(
//             bool wants_horizontal_root,
//             bool wants_vertical_root,
//             const Layout& layout,
//             const DataTypes& dtype,
//             const Protocols& protocol,
//             uint8_t version,
//             uint32_t internal_max_tries = DEFAULT_MAX_TRIES
//         )
//         {
//             auto apc = std::make_shared<PythonAPC>();
//             if (!CreateAPC(
//                     apc,
//                     wants_horizontal_root,
//                     wants_vertical_root,
//                     layout,
//                     dtype,
//                     protocol,
//                     version,
//                     internal_max_tries
//                 ))
//             {
//                 throw std::runtime_error("CreateAPC failed");
//             }
//             return apc;
//         }
//     };

//     inline bool PythonAPC::IsActiveAPC() const noexcept
//     {
//         const APCSnapshot snapshot = Snapshot_();
//         return
//             SnapshotUsable_(snapshot) &&
//             snapshot.Native->IsActiveAPC();
//     }

//     template<class Operation>
//     bool MutateTwoAPCs(
//         PythonAPC& first,
//         PythonAPC& second,
//         Operation&& operation
//     )
//     {
//         const APCSnapshot first_snapshot = first.Snapshot_();
//         const APCSnapshot second_snapshot = second.Snapshot_();

//         if (
//             !PythonAPC::SnapshotUsable_(first_snapshot) ||
//             !PythonAPC::SnapshotUsable_(second_snapshot) ||
//             first_snapshot.Epoch.get() != second_snapshot.Epoch.get()
//         )
//         {
//             return false;
//         }

//         // The strong epoch snapshots keep the slab alive. Native APC methods
//         // acquire their own generation/access scopes exactly once.
//         return std::forward<Operation>(operation)(
//             *first_snapshot.Native,
//             *second_snapshot.Native
//         );
//     }

//     template<class Operation>
//     bool MutateOneAPC(PythonAPC& apc, Operation&& operation)
//     {
//         const APCSnapshot snapshot = apc.Snapshot_();
//         if (!PythonAPC::SnapshotUsable_(snapshot))
//         {
//             return false;
//         }
//         return std::forward<Operation>(operation)(*snapshot.Native);
//     }

//     inline bool PythonAPC::AttachSiblingOrChild(
//         PythonAPC& sibling,
//         Axis axis,
//         Inheritance inheritance,
//         uint32_t max_tries
//     )
//     {
//         return MutateTwoAPCs(*this, sibling, [&](NativeAPC& self, NativeAPC& other)
//         {
//             return self.AttachSiblingOrChild(other, axis, inheritance, max_tries);
//         });
//     }

//     inline bool PythonAPC::AttachMeToAnother(
//         PythonAPC& sibling,
//         Axis axis,
//         Inheritance inheritance,
//         uint32_t max_tries
//     )
//     {
//         return MutateTwoAPCs(*this, sibling, [&](NativeAPC& self, NativeAPC& other)
//         {
//             return self.AttachMeToAnother(other, axis, inheritance, max_tries);
//         });
//     }

//     inline bool PythonAPC::DetachMyChild(
//         PythonAPC& child,
//         Axis axis,
//         uint32_t max_tries
//     )
//     {
//         return MutateTwoAPCs(*this, child, [&](NativeAPC& self, NativeAPC& other)
//         {
//             return self.DetachMyChild(other, axis, max_tries);
//         });
//     }

//     inline bool PythonAPC::DetachMeFromAnotherEdge(
//         Axis axis,
//         uint32_t max_tries
//     )
//     {
//         return MutateOneAPC(*this, [&](NativeAPC& self)
//         {
//             return self.DetachMeFromAnotherEdge(axis, max_tries);
//         });
//     }

//     inline bool PythonAPC::DetachAndReAttachMeToThisParent(
//         PythonAPC& root_parent,
//         Axis axis,
//         uint32_t max_tries
//     )
//     {
//         return MutateTwoAPCs(*this, root_parent, [&](NativeAPC& self, NativeAPC& parent)
//         {
//             return self.DetachAndReAttachMeToThisParent(parent, axis, max_tries);
//         });
//     }

//     inline bool PythonAPC::DetachAndReattachMeAsEquivelentSibbling(
//         PythonAPC& sibling,
//         Axis axis,
//         uint32_t max_tries
//     )
//     {
//         return MutateTwoAPCs(*this, sibling, [&](NativeAPC& self, NativeAPC& other)
//         {
//             return self.DetachAndReattachMeAsEquivelentSibbling(
//                 other,
//                 axis,
//                 max_tries
//             );
//         });
//     }

//     inline NavigationResult PythonAPC::FindPrevious(
//         Axis axis,
//         uint32_t max_tries
//     )
//     {
//         const APCSnapshot snapshot = Snapshot_();
//         if (!SnapshotUsable_(snapshot))
//         {
//             return {};
//         }

//         const auto result = snapshot.Native->FindPrevious(axis, max_tries);
//         if (result.MutationOP_ != NavigationStatus::FOUND)
//         {
//             return NavigationResult{result.MutationOP_, {}};
//         }

//         std::shared_ptr<PythonAPC> resolved =
//             snapshot.Epoch->Resolve(result.APC_);

//         return resolved
//             ? NavigationResult{NavigationStatus::FOUND, std::move(resolved)}
//             : NavigationResult{NavigationStatus::RETRY, {}};
//     }

//     inline NavigationResult PythonAPC::FindMyNext(
//         Axis axis,
//         Inheritance inheritance,
//         uint32_t max_tries
//     )
//     {
//         const APCSnapshot snapshot = Snapshot_();
//         if (!SnapshotUsable_(snapshot))
//         {
//             return {};
//         }

//         const auto result = snapshot.Native->FindMyNext(
//             axis,
//             inheritance,
//             max_tries
//         );
//         if (result.MutationOP_ != NavigationStatus::FOUND)
//         {
//             return NavigationResult{result.MutationOP_, {}};
//         }

//         std::shared_ptr<PythonAPC> resolved =
//             snapshot.Epoch->Resolve(result.APC_);

//         return resolved
//             ? NavigationResult{NavigationStatus::FOUND, std::move(resolved)}
//             : NavigationResult{NavigationStatus::RETRY, {}};
//     }

//     inline bool PythonAPC::Retire(uint32_t max_tries)
//     {
//         const APCSnapshot snapshot = Snapshot_();
//         if (!SnapshotUsable_(snapshot))
//         {
//             return false;
//         }

//         // Preallocate publication state before the native operation commits.
//         auto retired_owner = std::make_unique<APCBinding>();
//         retired_owner->Epoch = snapshot.Epoch;
//         retired_owner->Slot = snapshot.Binding->Slot;
//         retired_owner->Phase = BindingPhase::RETIRED;
//         APCBinding* retired = retired_owner.get();
//         RetainBindingRecord_(retired_owner.release());

//         if (!snapshot.Native->Retire(max_tries))
//         {
//             return false;
//         }

//         Binding_.store(retired, std::memory_order_release);

//         const uint32_t slot = snapshot.Binding->Slot;
//         if (slot < snapshot.Epoch->SlotCount)
//         {
//             PythonAPC* current =
//                 snapshot.Epoch->CurrentHandles[slot].load(
//                     std::memory_order_acquire
//                 );

//             while (
//                 current &&
//                 current == this &&
//                 !snapshot.Epoch->CurrentHandles[slot].compare_exchange_weak(
//                     current,
//                     nullptr,
//                     std::memory_order_acq_rel,
//                     std::memory_order_acquire
//                 )
//             )
//             {}
//         }
//         return true;
//     }

//     template<class DType>
//     std::optional<RegionView<DType>> PythonAPC::BuildNativeView(
//         MacroColumnOfAPC column,
//         std::shared_ptr<FabricEpoch>& epoch
//     )
//     {
//         epoch.reset();
//         const APCSnapshot snapshot = Snapshot_();
//         if (!SnapshotUsable_(snapshot))
//         {
//             return std::nullopt;
//         }

//         std::optional<RegionView<DType>> view =
//             snapshot.Native->template BuildAViewOverRegion<DType>(column);

//         if (view)
//         {
//             epoch = snapshot.Epoch;
//         }
//         return view;
//     }

//     template<class DType>
//     bool PythonAPC::ZeroARegion(MacroColumnOfAPC column)
//     {
//         return MutateOneAPC(*this, [&](NativeAPC& self)
//         {
//             return self.template ZeroARegion<DType>(column);
//         });
//     }

//     template<class DType>
//     class PythonRegionView final
//     {
//     private:
//         // Member order is intentional: NativeView_ releases its APC use scope
//         // before Epoch_ can destroy the slab.
//         std::shared_ptr<FabricEpoch> Epoch_{};
//         RegionView<DType> NativeView_{};

//         void Validate_() const
//         {
//             if (
//                 !Epoch_ ||
//                 !Epoch_->Accepting.load(std::memory_order_acquire) ||
//                 !NativeView_.IsValid()
//             )
//             {
//                 throw std::runtime_error(
//                     "The region view was invalidated by Fabric shutdown"
//                 );
//             }
//         }

//         void CheckIndex_(size_t index) const
//         {
//             if (index >= NativeView_.Size())
//             {
//                 throw py::index_error("region index is out of range");
//             }
//         }

//     public:
//         PythonRegionView(
//             std::shared_ptr<FabricEpoch> epoch,
//             RegionView<DType> native_view
//         ) noexcept
//             : Epoch_(std::move(epoch)),
//               NativeView_(std::move(native_view))
//         {}

//         [[nodiscard]] bool IsValid() const noexcept
//         {
//             return
//                 Epoch_ &&
//                 Epoch_->Accepting.load(std::memory_order_acquire) &&
//                 NativeView_.IsValid();
//         }

//         [[nodiscard]] size_t Size() const noexcept
//         {
//             return NativeView_.Size();
//         }

//         [[nodiscard]] RegionProtocol GetProtocol() const noexcept
//         {
//             return NativeView_.GetProtocol();
//         }

//         DType Load(size_t index)
//         {
//             Validate_();
//             CheckIndex_(index);

//             if (NativeView_.GetProtocol() == RegionProtocol::PRIVATE_REGION)
//             {
//                 auto span = NativeView_.RawMutableSpan();
//                 if (!span)
//                 {
//                     throw std::runtime_error(
//                         "PRIVATE_REGION did not provide its mutable span"
//                     );
//                 }
//                 return span.value()[index];
//             }

//             if (NativeView_.GetProtocol() == RegionProtocol::ATOMIC_WORD_ARRAY)
//             {
//                 return NativeView_.AtomicLoad(index, std::memory_order_acquire);
//             }

//             throw std::runtime_error(
//                 "This native protocol has no public readable RegionView operation"
//             );
//         }

//         bool Store(size_t index, DType value)
//         {
//             Validate_();
//             CheckIndex_(index);

//             if (NativeView_.GetProtocol() == RegionProtocol::PRIVATE_REGION)
//             {
//                 auto span = NativeView_.RawMutableSpan();
//                 if (!span)
//                 {
//                     return false;
//                 }
//                 span.value()[index] = value;
//                 return true;
//             }

//             if (NativeView_.GetProtocol() == RegionProtocol::ATOMIC_WORD_ARRAY)
//             {
//                 return NativeView_.AtomicStore(
//                     index,
//                     value,
//                     std::memory_order_release
//                 );
//             }
//             return false;
//         }

//         DType AtomicLoad(size_t index)
//         {
//             Validate_();
//             CheckIndex_(index);
//             if (NativeView_.GetProtocol() != RegionProtocol::ATOMIC_WORD_ARRAY)
//             {
//                 throw std::runtime_error("AtomicLoad requires ATOMIC_WORD_ARRAY");
//             }
//             return NativeView_.AtomicLoad(index, std::memory_order_acquire);
//         }

//         bool AtomicStore(size_t index, DType value)
//         {
//             Validate_();
//             CheckIndex_(index);
//             return NativeView_.AtomicStore(
//                 index,
//                 value,
//                 std::memory_order_release
//             );
//         }

//         std::pair<bool, DType> AtomicCompareExchangeStrong(
//             size_t index,
//             DType expected,
//             DType desired
//         )
//         {
//             Validate_();
//             CheckIndex_(index);
//             const bool exchanged = NativeView_.AtomicCompareExchangeStrong(
//                 index,
//                 expected,
//                 desired,
//                 std::memory_order_acq_rel,
//                 std::memory_order_acquire
//             );
//             return {exchanged, expected};
//         }

//         bool FillZero()
//         {
//             Validate_();
//             if (NativeView_.GetProtocol() == RegionProtocol::PRIVATE_REGION)
//             {
//                 auto span = NativeView_.RawMutableSpan();
//                 if (!span)
//                 {
//                     return false;
//                 }
//                 for (DType& value : span.value())
//                 {
//                     value = DType{};
//                 }
//                 return true;
//             }

//             if (NativeView_.GetProtocol() == RegionProtocol::ATOMIC_WORD_ARRAY)
//             {
//                 for (size_t i = 0u; i < NativeView_.Size(); ++i)
//                 {
//                     if (!NativeView_.AtomicStore(
//                             i,
//                             DType{},
//                             std::memory_order_relaxed
//                         ))
//                     {
//                         return false;
//                     }
//                 }
//                 return true;
//             }
//             return false;
//         }

//         py::array_t<DType> ToNumpyCopy()
//         {
//             Validate_();
//             py::array_t<DType> output(NativeView_.Size());
//             auto output_view = output.template mutable_unchecked<1>();

//             if (NativeView_.GetProtocol() == RegionProtocol::PRIVATE_REGION)
//             {
//                 auto span = NativeView_.RawMutableSpan();
//                 if (!span)
//                 {
//                     throw std::runtime_error(
//                         "PRIVATE_REGION did not provide its mutable span"
//                     );
//                 }
//                 for (py::ssize_t i = 0; i < output_view.shape(0); ++i)
//                 {
//                     output_view(i) = span.value()[static_cast<size_t>(i)];
//                 }
//                 return output;
//             }

//             if (NativeView_.GetProtocol() == RegionProtocol::ATOMIC_WORD_ARRAY)
//             {
//                 for (py::ssize_t i = 0; i < output_view.shape(0); ++i)
//                 {
//                     output_view(i) = NativeView_.AtomicLoad(
//                         static_cast<size_t>(i),
//                         std::memory_order_acquire
//                     );
//                 }
//                 return output;
//             }

//             throw std::runtime_error(
//                 "IMMUTABLE_SNAPSHOT has no public native read accessor yet"
//             );
//         }
//     };

//     template<class DType>
//     std::shared_ptr<PythonRegionView<DType>> MakeRegionView(
//         PythonAPC& apc,
//         MacroColumnOfAPC column
//     )
//     {
//         std::shared_ptr<FabricEpoch> epoch{};
//         std::optional<RegionView<DType>> native_view =
//             apc.template BuildNativeView<DType>(column, epoch);

//         if (!native_view || !epoch)
//         {
//             return {};
//         }

//         return std::make_shared<PythonRegionView<DType>>(
//             std::move(epoch),
//             std::move(native_view.value())
//         );
//     }

//     inline py::object BuildRegionView(
//         PythonAPC& apc,
//         MacroColumnOfAPC column,
//         RegionDataType dtype
//     )
//     {
//         switch (dtype)
//         {
//         case RegionDataType::UINT8_T:
//             return py::cast(MakeRegionView<uint8_t>(apc, column));
//         case RegionDataType::UINT16_T:
//             return py::cast(MakeRegionView<uint16_t>(apc, column));
//         case RegionDataType::UINT32_T:
//             return py::cast(MakeRegionView<uint32_t>(apc, column));
//         case RegionDataType::UINT64_T:
//             return py::cast(MakeRegionView<uint64_t>(apc, column));
//         case RegionDataType::INT8_T:
//             return py::cast(MakeRegionView<int8_t>(apc, column));
//         case RegionDataType::INT16_T:
//             return py::cast(MakeRegionView<int16_t>(apc, column));
//         case RegionDataType::INT32_T:
//             return py::cast(MakeRegionView<int32_t>(apc, column));
//         case RegionDataType::INT64_T:
//             return py::cast(MakeRegionView<int64_t>(apc, column));
//         case RegionDataType::FLOAT32_T:
//             return py::cast(MakeRegionView<float>(apc, column));
//         case RegionDataType::FLOAT64_T:
//             return py::cast(MakeRegionView<double>(apc, column));
//         case RegionDataType::CHAR:
//             return py::cast(MakeRegionView<char>(apc, column));
//         case RegionDataType::FLOAT16_T:
//             throw py::value_error(
//                 "FLOAT16_T is present in the schema enum, but "
//                 "CppTypeToRegionDType does not currently define a C++ half type"
//             );
//         }
//         throw py::value_error("Unknown RegionDataType");
//     }

//     inline bool ZeroRegion(
//         PythonAPC& apc,
//         MacroColumnOfAPC column,
//         RegionDataType dtype
//     )
//     {
//         switch (dtype)
//         {
//         case RegionDataType::UINT8_T:
//             return apc.ZeroARegion<uint8_t>(column);
//         case RegionDataType::UINT16_T:
//             return apc.ZeroARegion<uint16_t>(column);
//         case RegionDataType::UINT32_T:
//             return apc.ZeroARegion<uint32_t>(column);
//         case RegionDataType::UINT64_T:
//             return apc.ZeroARegion<uint64_t>(column);
//         case RegionDataType::INT8_T:
//             return apc.ZeroARegion<int8_t>(column);
//         case RegionDataType::INT16_T:
//             return apc.ZeroARegion<int16_t>(column);
//         case RegionDataType::INT32_T:
//             return apc.ZeroARegion<int32_t>(column);
//         case RegionDataType::INT64_T:
//             return apc.ZeroARegion<int64_t>(column);
//         case RegionDataType::FLOAT32_T:
//             return apc.ZeroARegion<float>(column);
//         case RegionDataType::FLOAT64_T:
//             return apc.ZeroARegion<double>(column);
//         case RegionDataType::CHAR:
//             return apc.ZeroARegion<char>(column);
//         case RegionDataType::FLOAT16_T:
//             throw py::value_error("FLOAT16_T has no corresponding C++ half type");
//         }
//         throw py::value_error("Unknown RegionDataType");
//     }

//     template<class DType>
//     void BindRegionView(py::module_& module, const char* python_name)
//     {
//         using View = PythonRegionView<DType>;

//         py::class_<View, std::shared_ptr<View>>(module, python_name)
//             .def_property_readonly("is_valid", &View::IsValid)
//             .def_property_readonly("size", &View::Size)
//             .def_property_readonly("protocol", &View::GetProtocol)
//             .def("IsValid", &View::IsValid)
//             .def("Size", &View::Size)
//             .def("GetProtocol", &View::GetProtocol)
//             .def("load", &View::Load, py::arg("index"))
//             .def("store", &View::Store, py::arg("index"), py::arg("value"))
//             .def("AtomicLoad", &View::AtomicLoad, py::arg("index"))
//             .def(
//                 "AtomicStore",
//                 &View::AtomicStore,
//                 py::arg("index"),
//                 py::arg("value")
//             )
//             .def(
//                 "AtomicCompareExchangeStrong",
//                 &View::AtomicCompareExchangeStrong,
//                 py::arg("index"),
//                 py::arg("expected"),
//                 py::arg("desired"),
//                 "Return (exchanged, observed). observed is the updated C++ expected value."
//             )
//             .def("fill_zero", &View::FillZero)
//             .def(
//                 "to_numpy",
//                 &View::ToNumpyCopy,
//                 "Return a safe copy. Direct NumPy exposure is intentionally "
//                 "not provided for atomic storage."
//             )
//             .def("__len__", &View::Size)
//             .def("__getitem__", &View::Load, py::arg("index"))
//             .def("__setitem__", [](View& self, size_t index, DType value)
//             {
//                 if (!self.Store(index, value))
//                 {
//                     throw std::runtime_error(
//                         "The region protocol rejected the store"
//                     );
//                 }
//             });
//     }

//     inline void BindArchitecture(py::module_& module)
//     {
//         module.doc() =
//             "Lock-free pybind11 adapter for SuperNova APC/Fabric architecture";

//         py::enum_<Axis>(module, "BidirectionalAxis")
//             .value("HORIZONTAL", Axis::HORIZONTAL)
//             .value("VERTICAL", Axis::VERTICAL)
//             .export_values();

//         py::enum_<Inheritance>(module, "DescOfInharitance")
//             .value("FIRST_CHILD", Inheritance::FIRST_CHILD)
//             .value("LINKED_CHILD", Inheritance::LINKED_CHILD)
//             .export_values();

//         py::enum_<NavigationStatus>(module, "SeqLockedOperation")
//             .value("FOUND", NavigationStatus::FOUND)
//             .value("NONE", NavigationStatus::NONE)
//             .value("RETRY", NavigationStatus::RETRY)
//             .export_values();

//         py::enum_<MacroColumnOfAPC>(module, "MacroColumnOfAPC")
//             .value(
//                 "FEEDFORWARD_MESSAGE",
//                 MacroColumnOfAPC::FEEDFORWARD_MESSAGE
//             )
//             .value(
//                 "FEEDBACKWARD_MESSAGE",
//                 MacroColumnOfAPC::FEEDBACKWARD_MESSAGE
//             )
//             .value("LATERAL_MESAGE", MacroColumnOfAPC::LATERAL_MESAGE)
//             .value("STATE_SLOT", MacroColumnOfAPC::STATE_SLOT)
//             .value("ERROR_SLOT", MacroColumnOfAPC::ERROR_SLOT)
//             .value("WEIGHTLESS_LOOKUP", MacroColumnOfAPC::WEIGHTLESS_LOOKUP)
//             .value("WEIGHT_SLOT", MacroColumnOfAPC::WEIGHT_SLOT)
//             .value("AUX_SLOT", MacroColumnOfAPC::AUX_SLOT)
//             .value("HETEROGENOUS_PTR", MacroColumnOfAPC::HETEROGENOUS_PTR)
//             .value("FREE_SLOT", MacroColumnOfAPC::FREE_SLOT)
//             .export_values();

//         py::enum_<RegionProtocol>(module, "SchemaProtocols")
//             .value("PRIVATE_REGION", RegionProtocol::PRIVATE_REGION)
//             .value("IMMUTABLE_SNAPSHOT", RegionProtocol::IMMUTABLE_SNAPSHOT)
//             .value("ATOMIC_WORD_ARRAY", RegionProtocol::ATOMIC_WORD_ARRAY)
//             .value(
//                 "MPMC_FIXED_RECORD_QUEUE",
//                 RegionProtocol::MPMC_FIXED_RECORD_QUEUE
//             )
//             .value("DOUBLE_BUFFERED", RegionProtocol::DOUBLE_BUFFERED)
//             .export_values();

//         py::enum_<RegionDataType>(module, "DataTypeOfMacroColumn")
//             .value("UINT8_T", RegionDataType::UINT8_T)
//             .value("UINT16_T", RegionDataType::UINT16_T)
//             .value("UINT32_T", RegionDataType::UINT32_T)
//             .value("UINT64_T", RegionDataType::UINT64_T)
//             .value("INT8_T", RegionDataType::INT8_T)
//             .value("INT16_T", RegionDataType::INT16_T)
//             .value("INT32_T", RegionDataType::INT32_T)
//             .value("INT64_T", RegionDataType::INT64_T)
//             .value("FLOAT16_T", RegionDataType::FLOAT16_T)
//             .value("FLOAT32_T", RegionDataType::FLOAT32_T)
//             .value("FLOAT64_T", RegionDataType::FLOAT64_T)
//             .value("CHAR", RegionDataType::CHAR)
//             .export_values();

//         py::class_<Layout>(module, "LayoutSpanAndPercentageCarrier")
//             .def(py::init<>())
//             .def_readwrite("FeedForward", &Layout::FeedForward)
//             .def_readwrite("FeedBackward", &Layout::FeedBackward)
//             .def_readwrite("Lateral", &Layout::Lateral)
//             .def_readwrite("StateSlot", &Layout::StateSlot)
//             .def_readwrite("ErrorSlot", &Layout::ErrorSlot)
//             .def_readwrite("Weightless", &Layout::Weightless)
//             .def_readwrite("WeightSlot", &Layout::WeightSlot)
//             .def_readwrite("AUXSlot", &Layout::AUXSlot)
//             .def_readwrite("HeterogenousPtr", &Layout::HeterogenousPtr)
//             .def_readwrite("FreeSlot", &Layout::FreeSlot);

//         py::class_<DataTypes>(module, "InitialRegionalDtypeConf")
//             .def(py::init<>())
//             .def_readwrite(
//                 "FEEDFORWARD_MESSAGE",
//                 &DataTypes::FEEDFORWARD_MESSAGE
//             )
//             .def_readwrite(
//                 "FEEDBACKWARD_MESSAGE",
//                 &DataTypes::FEEDBACKWARD_MESSAGE
//             )
//             .def_readwrite("LATERAL_MESAGE", &DataTypes::LATERAL_MESAGE)
//             .def_readwrite("STATE_SLOT", &DataTypes::STATE_SLOT)
//             .def_readwrite("ERROR_SLOT", &DataTypes::ERROR_SLOT)
//             .def_readwrite(
//                 "WEIGHTLESS_LOOKUP",
//                 &DataTypes::WEIGHTLESS_LOOKUP
//             )
//             .def_readwrite("WEIGHT_SLOT", &DataTypes::WEIGHT_SLOT)
//             .def_readwrite("AUX_SLOT", &DataTypes::AUX_SLOT)
//             .def_readwrite("HETEROGENOUS_PTR", &DataTypes::HETEROGENOUS_PTR)
//             .def_readwrite("FREE_SLOT", &DataTypes::FREE_SLOT);

//         py::class_<Protocols>(module, "InitialRegionalProtocol")
//             .def(py::init<>())
//             .def_readwrite(
//                 "FEEDFORWARD_MESSAGE",
//                 &Protocols::FEEDFORWARD_MESSAGE
//             )
//             .def_readwrite(
//                 "FEEDBACKWARD_MESSAGE",
//                 &Protocols::FEEDBACKWARD_MESSAGE
//             )
//             .def_readwrite("LATERAL_MESAGE", &Protocols::LATERAL_MESAGE)
//             .def_readwrite("STATE_SLOT", &Protocols::STATE_SLOT)
//             .def_readwrite("ERROR_SLOT", &Protocols::ERROR_SLOT)
//             .def_readwrite(
//                 "WEIGHTLESS_LOOKUP",
//                 &Protocols::WEIGHTLESS_LOOKUP
//             )
//             .def_readwrite("WEIGHT_SLOT", &Protocols::WEIGHT_SLOT)
//             .def_readwrite("AUX_SLOT", &Protocols::AUX_SLOT)
//             .def_readwrite("HETEROGENOUS_PTR", &Protocols::HETEROGENOUS_PTR)
//             .def_readwrite("FREE_SLOT", &Protocols::FREE_SLOT);

//         py::class_<PythonAPC, std::shared_ptr<PythonAPC>> apc_class(
//             module,
//             "AdaptivePackedCellContainer"
//         );

//         py::class_<NavigationResult>(module, "RelationOparation")
//             .def_readonly("MutationOP_", &NavigationResult::Status)
//             .def_readonly("APC_", &NavigationResult::APC)
//             .def_property_readonly("status", [](const NavigationResult& self)
//             {
//                 return self.Status;
//             })
//             .def_property_readonly("apc", [](const NavigationResult& self)
//             {
//                 return self.APC;
//             })
//             .def_property_readonly("found", &NavigationResult::Found)
//             .def("as_tuple", [](const NavigationResult& self)
//             {
//                 return py::make_tuple(self.Status, self.APC);
//             })
//             .def("__bool__", &NavigationResult::Found);

//         BindRegionView<uint8_t>(module, "RegionViewUInt8");
//         BindRegionView<uint16_t>(module, "RegionViewUInt16");
//         BindRegionView<uint32_t>(module, "RegionViewUInt32");
//         BindRegionView<uint64_t>(module, "RegionViewUInt64");
//         BindRegionView<int8_t>(module, "RegionViewInt8");
//         BindRegionView<int16_t>(module, "RegionViewInt16");
//         BindRegionView<int32_t>(module, "RegionViewInt32");
//         BindRegionView<int64_t>(module, "RegionViewInt64");
//         BindRegionView<float>(module, "RegionViewFloat32");
//         BindRegionView<double>(module, "RegionViewFloat64");
//         BindRegionView<char>(module, "RegionViewChar");

//         apc_class
//             .def(py::init<>())
//             .def("GetThisSlotIdx", &PythonAPC::GetThisSlotIdx)
//             .def("IsActiveAPC", &PythonAPC::IsActiveAPC)
//             .def(
//                 "AttachSiblingOrChild",
//                 &PythonAPC::AttachSiblingOrChild,
//                 py::arg("sibling"),
//                 py::arg("axis"),
//                 py::arg("inheritance"),
//                 py::arg("max_tries") = DEFAULT_MAX_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "AttachMeToAnother",
//                 &PythonAPC::AttachMeToAnother,
//                 py::arg("sibling"),
//                 py::arg("axis"),
//                 py::arg("inheritance"),
//                 py::arg("max_tries") = DEFAULT_MAX_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "DetachMyChild",
//                 &PythonAPC::DetachMyChild,
//                 py::arg("child"),
//                 py::arg("axis"),
//                 py::arg("max_tries") = DEFAULT_MAX_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "DetachMeFromAnotherEdge",
//                 &PythonAPC::DetachMeFromAnotherEdge,
//                 py::arg("axis"),
//                 py::arg("max_tries") = DEFAULT_MAX_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "DetachAndReAttachMeToThisParent",
//                 &PythonAPC::DetachAndReAttachMeToThisParent,
//                 py::arg("root_parent"),
//                 py::arg("axis"),
//                 py::arg("max_tries") = DEFAULT_MAX_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "DetachAndReattachMeAsEquivelentSibbling",
//                 &PythonAPC::DetachAndReattachMeAsEquivelentSibbling,
//                 py::arg("sibling"),
//                 py::arg("axis"),
//                 py::arg("max_tries") = DEFAULT_MAX_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "FindPrevious",
//                 &PythonAPC::FindPrevious,
//                 py::arg("axis"),
//                 py::arg("max_tries") = NativeAPC::REALTION_FIND_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "FindMyNext",
//                 &PythonAPC::FindMyNext,
//                 py::arg("axis"),
//                 py::arg("inheritance"),
//                 py::arg("max_tries") = NativeAPC::REALTION_FIND_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "Retire",
//                 &PythonAPC::Retire,
//                 py::arg("max_tries") = DEFAULT_MAX_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "BuildAViewOverRegion",
//                 &BuildRegionView,
//                 py::arg("macro_column"),
//                 py::arg("dtype"),
//                 "Build the schema-exact typed view; return None when schema "
//                 "and dtype do not match."
//             )
//             .def(
//                 "ZeroARegion",
//                 &ZeroRegion,
//                 py::arg("macro_column"),
//                 py::arg("dtype")
//             );

//         py::class_<PythonFabric, std::shared_ptr<PythonFabric>> fabric_class(
//             module,
//             "VagueTemoraryPremativeFabric"
//         );

//         fabric_class
//             .def(py::init<>())
//             .def(
//                 "InitializeFabric",
//                 &PythonFabric::InitializeFabric,
//                 py::arg("slot_count"),
//                 py::arg("slot_cell_count") = MINIMUM_APC_CELL_COUNT,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "ShutDownFabric",
//                 &PythonFabric::ShutDownFabric,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "ShutDownFabric",
//                 &PythonFabric::ShutDownFabric,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def("IsFabricActive", &PythonFabric::IsFabricActive)
//             .def(
//                 "CreateAPC",
//                 &PythonFabric::CreateAPC,
//                 py::arg("desired_apc"),
//                 py::arg("wants_horizontal_root") = false,
//                 py::arg("wants_vertical_root") = false,
//                 py::arg("layout") = Layout{},
//                 py::arg("dtype") = DataTypes{},
//                 py::arg("protocol") = Protocols{},
//                 py::arg("version") = ADS::BRANCH_VERSION,
//                 py::arg("internal_max_tries") = DEFAULT_MAX_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             )
//             .def(
//                 "create_apc",
//                 &PythonFabric::CreateAPCAndReturn,
//                 py::arg("wants_horizontal_root") = false,
//                 py::arg("wants_vertical_root") = false,
//                 py::arg("layout") = Layout{},
//                 py::arg("dtype") = DataTypes{},
//                 py::arg("protocol") = Protocols{},
//                 py::arg("version") = ADS::BRANCH_VERSION,
//                 py::arg("internal_max_tries") = DEFAULT_MAX_TRIES,
//                 py::call_guard<py::gil_scoped_release>()
//             );

//         // Pythonic aliases while preserving the architecture's original names.
//         apc_class.attr("get_this_slot_idx") = apc_class.attr("GetThisSlotIdx");
//         apc_class.attr("is_valid") = apc_class.attr("IsActiveAPC");
//         apc_class.attr("attach_sibling_or_child") =
//             apc_class.attr("AttachSiblingOrChild");
//         apc_class.attr("attach_me_to_another") =
//             apc_class.attr("AttachMeToAnother");
//         apc_class.attr("detach_my_child") = apc_class.attr("DetachMyChild");
//         apc_class.attr("detach_me_from_another_edge") =
//             apc_class.attr("DetachMeFromAnotherEdge");
//         apc_class.attr("detach_and_reattach_to_parent") =
//             apc_class.attr("DetachAndReAttachMeToThisParent");
//         apc_class.attr("detach_and_reattach_as_sibling") =
//             apc_class.attr("DetachAndReattachMeAsEquivelentSibbling");
//         apc_class.attr("find_previous") = apc_class.attr("FindPrevious");
//         apc_class.attr("find_my_next") = apc_class.attr("FindMyNext");
//         apc_class.attr("retire") = apc_class.attr("Retire");
//         apc_class.attr("build_region_view") =
//             apc_class.attr("BuildAViewOverRegion");
//         apc_class.attr("zero_region") = apc_class.attr("ZeroARegion");

//         fabric_class.attr("initialize") =
//             fabric_class.attr("InitializeFabric");
//         fabric_class.attr("shutdown") = fabric_class.attr("ShutDownFabric");
//         fabric_class.attr("is_active") = fabric_class.attr("IsFabricActive");

//         module.attr("APC") = module.attr("AdaptivePackedCellContainer");
//         module.attr("Fabric") = module.attr("VagueTemoraryPremativeFabric");
//         module.attr("Axis") = module.attr("BidirectionalAxis");
//         module.attr("Inheritance") = module.attr("DescOfInharitance");
//         module.attr("NavigationStatus") = module.attr("SeqLockedOperation");
//         module.attr("RegionProtocol") = module.attr("SchemaProtocols");
//         module.attr("RegionDataType") = module.attr("DataTypeOfMacroColumn");
//         module.attr("DEFAULT_MAX_TRIES") = py::int_(DEFAULT_MAX_TRIES);
//         module.attr("BRANCH_VERSION") =
//             py::int_(ADS::BRANCH_VERSION);
//         module.attr("MINIMUM_APC_CELL_COUNT") =
//             py::int_(MINIMUM_APC_CELL_COUNT);
//     }
// }
