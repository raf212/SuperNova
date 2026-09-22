# SuperNova APC + Fabric + GHGF — Architecture and API Documentation

## Table of contents

- [1. Architecture in one page](#1-architecture-in-one-page)
- [2. File and dependency map](#2-file-and-dependency-map)
- [3. Inheritance / ownership boundaries](#3-inheritance--ownership-boundaries)
- [4. Persistent slab state vs runtime-only state](#4-persistent-slab-state-vs-runtime-only-state)
- [5. Shared constants and enums](#5-shared-constants-and-enums)
- [6. APC fundamental data structures](#6-apc-fundamental-data-structures)
- [7. APC header lifecycle](#7-apc-header-lifecycle)
- [8. Schema system](#8-schema-system)
- [9. Typed region views](#9-typed-region-views)
- [10. Fabric-to-APC binding](#10-fabric-to-apc-binding)
- [11. AdaptivePackedCellContainer public facade](#11-adaptivepackedcellcontainer-public-facade)
- [12. Fabric core and slab geometry](#12-fabric-core-and-slab-geometry)
- [13. Edge-table representation](#13-edge-table-representation)
- [14. Handle/generation and retirement control](#14-handlegeneration-and-retirement-control)
- [15. Fabric table constructors](#15-fabric-table-constructors)
- [16. Fabric initialization, save, attach, detach, shutdown](#16-fabric-initialization-save-attach-detach-shutdown)
- [17. Transactional DAG mutation](#17-transactional-dag-mutation)
- [18. APCFinilizer](#18-apcfinilizer)
- [19. End-to-end APC/Fabric flows](#19-end-to-end-apcfabric-flows)
- [20. GHGF storage model](#20-ghgf-storage-model)
- [21. GHGF classes and API](#21-ghgf-classes-and-api)
- [22. GHGF prediction/update/learning data flow](#22-ghgf-predictionupdatelearning-data-flow)
- [23. Online structural learning](#23-online-structural-learning)
- [24. Correct usage recipes](#24-correct-usage-recipes)
- [25. Misuse and anti-patterns](#25-misuse-and-anti-patterns)
- [26. Current limits / unfinished surfaces](#26-current-limits--unfinished-surfaces)
- [27. Function index by subsystem](#27-function-index-by-subsystem)
- [28. Glossary](#28-glossary)

---

## 1. Architecture in one page

SuperNova separates **stable numeric storage** from **mutable structural metadata** while keeping both inside one relocatable slab. The major boundary is:

```text

raw 64-bit slab

    │

    ├── Fabric metadata / record book / schemas

    ├── H edge table ───────────┐

    ├── V edge table ───────────┤ transactional structural metadata

    ├── generation table ───────┤

    ├── compiled DAG masks ─────┘

    └── APC segment pool

          │

          ├── 8-cell APC header

          └── typed, aligned regions

                 │

                 ▼

       AdaptivePackedCellContainer

                 │

                 ▼

              GHGFNode

                 │

                 ▼

       GHGFModelConstructor

```

The intended design rule is: **topology mutation should change relation/control metadata without relocating or reformatting the APC numeric regions.** A model-specific mutation may still update a scalar parameter (for GHGF, a coupling in `WEIGHT_SLOT`), but the region geometry remains stable.

### Core invariants

| Invariant | Meaning | Why it exists |
| --- | --- | --- |
| `parent_slot < child_slot` | Every legal H/V parent relation points from a lower slot to a higher slot. | The slot order itself is a topological order; accepted insertions cannot create a cycle through the public mutation path. |
| slot + generation identity | A node reference is not only a slot number. | Prevents an old reference from silently becoming a reference to a later node that reuses the same slot (ABA protection). |
| active-use count + closed bit | A live generation can be closed and drained before destructive lifecycle work. | Supports quiescence, retirement, save/detach and generation transition. |
| schema-defined region layout | APC data is interpreted through a fixed record that stores type/shape/protocol/offset. | Keeps data geometry explicit and relocatable. |
| edge-row sequence/status | Relation rows are reserved (odd sequence), edited, then published (even sequence). | Lets optimistic readers detect a concurrent mutation and return `RETRY` rather than observe a torn relation. |
| persistent data stores offsets/indices, not runtime C++ object pointers | The slab must remain meaningful after mapping at a different address. | Enables `SaveFabric` / `AttachFabric` relocation. |

### Boundary hierarchy — who should call what

```text

Application / model code

        │

        ▼

GHGFModelConstructor / GHGFStructralLearningModel

        │

        ▼

GHGFModel / GHGFNode

        │

        ▼

AdaptivePackedCellContainer             ← generic node-facing API

        │

        ▼

APCFinilizer                            ← generic Fabric/APC authority

        │

        ▼

ConstructDAGOnEachAxis / DAGMutationConf

        │

        ▼

SlabToFabricConverterAndCordinator      ← slab ownership/relocation

        │

        ▼

table constructors + lifecycle + edge/handle primitives

        │

        ▼

64-bit slab cells

```

**Rule of thumb:** model/application code should not write Fabric control cells, edge rows, generation cells, or schema rows directly. Those are implementation layers whose invariants are enforced by the classes above them.

---

## 2. File and dependency map

| File | Primary responsibility | Important types |
| --- | --- | --- |
| `AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp` | Public generic APC graph facade. | `AdaptivePackedCellContainer` |
| `APCOrchestrators/APCDataStructure.hpp` | APC header constants, packed helpers, runtime cache, use-scope RAII. | `APCDataStructure`, `APCUseScope`, `TwinU32ToU64`, `Twin28Plus8` |
| `APCOrchestrators/HeaderOrchestrator.hpp` | APC header lifecycle sequence/state representation. | `DescriptionOfAPC`, `HeaderOrchestrator` |
| `APCOrchestrators/SchemaOrchestratorForRegion.hpp` | Region schema ABI, dtype/protocol flags and validation. | `SchemaOrchestrator`, `SchemaValidator`, `SchemaDefinition` |
| `APCOrchestrators/SharedConf.hpp` | Shared enums, sentinels, limits. | `MacroColumnOfAPC`, `FabricSegments`, `StateOfAPC` |
| `APCOrchestrators/ViewOrchestrator.hpp` | Typed non-owning region views held by an APC use token. | `ResolveRegionBiteView`, `APCStorageGeometry`, `RegionView<T>` |
| `AdaptivePackedCellContainer/FabricToAPCLinker.hpp` | Runtime facade binding and generation-safe use acquisition. | `FabricToAPCLinker`, `RegionViewConstructor` |
| `FabricOrchestrators/CoreOfFabricCoordinator.hpp` | Global Fabric metadata, record-book geometry, raw allocator. | `CoreOfFabricCoordinator`, `RecordBookConf`, `RawPackedCellAllocator` |
| `FabricOrchestrators/EdgeTableConf.hpp` | Packed relation/header layout and DAG legality helpers. | `EdgeBuilder` |
| `FabricOrchestrators/HandleAndRetirement.hpp` | Generation/active-count control cell and relocation cache validation. | `HandleOfAPCStatic`, `APCRelocationDef` |
| `FabricTableConstructors/FabricConstructor.h` | Lowest slab read/write/atomic primitives; schema/handle table base classes. | `FabricConstructor`, `MatrixViewConstructor`, `APCHandleAndRetirement` |
| `FabricTableConstructors/CompleteFabric.h` | Record book, APC lifecycle, edge table, compiled DAG masks. | `RecordBookConstructor`, `APCLifeCycle`, `EdgeTableConstructor`, `CompiledDAGTableConstructor` |
| `SlabToFabricConverterAndCordinator.h` | Full slab layout, ownership, relocation, transaction structures. | `SlabToFabricConverterAndCordinator`, `DAGMutationConf`, `ConstructDAGOnEachAxis` |
| `VagueTemoraryPremativeFabric.hpp` | Final generic Fabric/APC facade constructor and resolver. | `APCFinilizer` |
| `Models/GHGF/GHGFLayer.hpp` | GHGF mathematical storage schema, roles, message/state indices, structural mutation structs. | `GHGFLayerModel`, `GHGFLearningConfig`, `StructuralLearningGHGF` |
| `Models/GHGF/GHGFNode.hpp` | One GHGF node layered over an APC. | `GHGFNode` |
| `Models/GHGF/GHGFModelOfAPC.hpp` | GHGF Fabric wrapper, model constructor, online structural-learning interface. | `GHGFModel`, `GHGFStructralLearningModel`, `GHGFModelConstructor` |

### Source implementation files

The corresponding `.cpp` sections implement the same boundaries: APC facade, linker/view construction, DAG transactions, GHGF model/node/structural learning, lifecycle/edge/matrix/record tables, slab coordination, and `APCFinilizer`. This documentation groups declarations with their implementations rather than repeating the same symbol twice.

---

## 3. Inheritance / ownership boundaries

### Fabric-side inheritance

```text

FabricConstructor

  └─ MatrixViewConstructor

      └─ APCHandleAndRetirement

          └─ RecordBookConstructor

              └─ APCLifeCycle

                  └─ EdgeTableConstructor

                      └─ CompiledDAGTableConstructor

                          └─ SlabToFabricConverterAndCordinator

                              └─ DAGMutationConf

                                  └─ ConstructDAGOnEachAxis

                                      └─ APCFinilizer

                                          └─(protected) GHGFModel

                                              └─ GHGFStructralLearningModel

                                                  └─ GHGFModelConstructor

```

### Node-side inheritance

```text

FabricToAPCLinker

  └─ RegionViewConstructor

      └─ AdaptivePackedCellContainer

          └─(protected) GHGFNode

```

The Fabric owns storage and graph authority. An APC facade is a **runtime view/binding** into that Fabric, not the storage owner. `GHGFNode` is a model-specific facade over one APC; `GHGFModelConstructor` owns/controls the model Fabric through protected inheritance.

---

## 4. Persistent slab state vs runtime-only state

| Category | Examples | Persistent / relocatable? | Rule |
| --- | --- | --- | --- |
| Fabric metadata | `FabricCache`, record-book entries | Yes | Contains sizes/offsets/indices, not process-local object pointers. |
| Topology | H/V edge rows, `ParentRelation`, compiled masks | Yes | Handles are slot+generation; sibling links are packed relation locators. |
| Node identity/lifecycle | APC header, handle control cells | Yes | Lifecycle and generation survive slab copy. |
| Schema | `RegionSchemaRecord` rows | Yes | Trivially copyable fixed records. |
| Node numeric data | APC region bytes | Yes | Model state/weights/messages live here. |
| APC facade cache | `CacheOfAPC::FabricOwnerPtr_`, `RawAPCBasePtr_`, `GenerationCellPtr_` | No | Rebuilt when an APC facade is bound; never treat these pointers as part of the saved format. |
| GHGF runtime pointer | `GHGFNode::GHGFFabric_` | No | Process-local link from node facade to model wrapper. |
| C++ spans/views | `std::span`, `RegionView<T>` | No | Ephemeral runtime views only. |

**Relocation rule:** any new field added to persistent slab structures must remain meaningful when the slab base address changes. Raw process pointers belong in runtime caches/facades, not persistent records.

---

## 5. Shared constants and enums

### `MacroColumnOfAPC` — logical data regions

| Value | Ordinal | Intended generic meaning |
| --- | --- | --- |
| `BOTTOM_UP_SLOT` | 0 | Bottom-up / feed-forward model data. |
| `TOP_DOWN_SLOT` | 1 | Top-down / feedback model data. |
| `LATERAL_MESAGE` | 2 | Reserved lateral-message region. |
| `STATE_SLOT` | 3 | Persistent/current node state. |
| `ERROR_SLOT` | 4 | Prediction/error state. |
| `WEIGHTLESS_LOOKUP` | 5 | Lookup/index-like region. |
| `WEIGHT_SLOT` | 6 | Model parameters / couplings. |
| `AUX_SLOT` | 7 | Auxiliary storage. |
| `EXTRA_SLOT` | 8 | Extra model-defined storage. |
| `FREE_SLOT` | 9 | Last generic region slot / unused model-defined slot. |

### `FabricSegments` — slab-global segments

| Segment | Purpose |
| --- | --- |
| `SLAB_RECORD_MAP` | Record book containing begin/end ranges for every Fabric segment. |
| `MATRIX_VIEW_TABLE` | Per-APC compact schema rows. |
| `VALUE_PARENT_EDGE_TABLE_H` | H-axis relation rows. |
| `VOLATILE_PARENT_EDGE_TABLE_V` | V-axis relation rows. |
| `APC_HANDLE_TABLE` | Generation + active-use + closed control word per APC. |
| `EDGE_TOPOLOGY_BITMAP` | Fast parent-occupancy masks for H and V. |
| `DEVICE_PLANNER_TABLE` | Reserved in current snapshot; configured record length is zero. |
| `WORK_QUEUE` | Reserved in current snapshot; configured record width is zero. |
| `SEGMENT_POOL` | Actual APC slots. |

### `StateOfAPC`

| State | Meaning |
| --- | --- |
| `FREE` | Unused slot state. |
| `RESERVED` | Lifecycle transition in progress; by invariant sequence is odd. |
| `LIVE` | Constructed node usable by clients; sequence even. |
| `RETIRED` | Logically retired slot awaiting/requiring reclaim before reuse. |
| `HAULTED` | Temporarily halted live APC state; spelling preserved from source. |

### Shared helpers

| Function | Arguments | Returns | What it does / who uses it | Do not use it for |
| --- | --- | --- | --- | --- |
| `IsLiveSateOfAPC(state)` | `optional<StateOfAPC>` | `bool` | Simple LIVE-state check used by lifecycle/resolver code. | Do not confuse header lifecycle LIVE with handle-table generation being open; both checks matter. |
| `MaskLowBitsForU64(n)` | bit count | `uint64_t` mask | Builds low-N-bit mask without undefined full-width shift. | Not a range check. |
| `MaskLowBitsForU32(n)` | bit count | `uint32_t` mask | 32-bit equivalent. | Not a relation locator validator. |

---

## 6. APC fundamental data structures

### `APCDataStructure`

Static description of APC-local ABI and helper functions. It does not own storage.

| Member/helper | Arguments / fields | Return | What it does | Misuse warning |
| --- | --- | --- | --- | --- |
| `HeaderIdentifierOfAPC` | `MAGIC_ID=0`, `APC_SLOT_IDX=1`, `GHGF_ROLE_CELL=2`, `APC_LIFE_CYCLE=6`, `EOF_APC_HEADER=7` | enum | Defines fixed positions in the 8-cell APC header. | Do not repurpose these cells without changing validation/format expectations. |
| `CountOfMacroColumn()` | none | `uint8_t` | Returns number of generic macro columns (10 in current source). | Do not hard-code 10 elsewhere when this helper is available. |
| `RegionBit(column)` | `MacroColumnOfAPC` | `uint16_t` | Maps one macro column to its active-mask bit. | Only valid for known columns. |
| `ValidRegionMask()` | none | `uint16_t` | Mask containing every valid region bit. | Not the active mask of a particular Fabric. |
| `CompactRegionIndex(active_mask,column)` | active bit mask + logical column | `optional<uint8_t>` | Returns compact schema-row ordinal by popcounting lower active bits. | Do not treat logical enum ordinal as compact row ordinal when some regions are disabled. |
| `IsValid32BitAPCUnit(index)` | `uint64_t` | `bool` | Ensures value is below `UINT32_MAX` sentinel. | Not sufficient to prove slot < `CountOfAPC_`. |
| `IsValidFabricUnit(index)` | `uint64_t` | `bool` | Ensures value is below `UINT64_MAX` Fabric sentinel. | Not a slab bounds check. |
| `InLimitOfUint8(version)` | `uint32_t` | `bool` | Valid nonzero value below `UINT8_MAX`. | Not used as a general integer validator. |
| `IsCapacityOfAPCValid(capacity)` | cell count | `bool` | Requires at least `MINIMUM_APC_CELL_COUNT` and 32-bit-valid cell count. | Does not validate a schema fits that capacity. |
| `IsPowerOfTwoValue(value)` | `uint64_t` | `bool` | Generic power-of-two test. | Zero is false. |
| `IsValidEven64(value)` | `uint64_t` | `bool` | Parity check used by lifecycle sequence invariant. | Even alone does not mean a lifecycle state is valid. |

### `APCDataStructure::CacheOfAPC`

| Field | Meaning |
| --- | --- |
| `FabricOwnerPtr_` | Runtime pointer to owning `APCFinilizer`. |
| `RawAPCBasePtr_` | Runtime pointer to this slot start. |
| `APCSlotIdx_` | Persistent identity component copied into runtime facade. |
| `GenerationCellPtr_` | Runtime pointer to handle-table control cell. |
| `CurrentGeneration_` | Generation expected by this facade. |

**Do not persist/copy this cache as a storage format.** The pointer fields are valid only in the current process/mapping.

### `APCUseScope`

Move-only RAII token proving that a specific APC generation is currently in active use. Acquisition happens through `FabricToAPCLinker::AcquireAPCUse_`; destruction/release atomically decrements the active-use count.

| Function | Arguments | Return | Who uses it | What it does / misuse |
| --- | --- | --- | --- | --- |
| default constructor | none | empty scope | Internal callers needing an output token. | Does not protect anything until moved a valid control-cell binding. |
| move constructor / move assignment | another scope | moved scope | Views/resolvers. | Transfers ownership; source becomes empty. Never copy. |
| `operator bool()` | none | `bool` | Any caller checking whether acquisition succeeded. | True means a use token is held, not that arbitrary pointers remain valid after token release. |
| `Release()` | none | `void` | Destructor or explicit early release. | Atomic `fetch_sub(1, release)`. Do not call twice expecting two decrements; it nulls its cell after first release. |

**Quiescence implication:** keeping a `RegionView` alive keeps its `APCUseScope` alive; save/detach/retire/shutdown may wait or fail until active uses drain. Do not hold views indefinitely across lifecycle operations.

### Packing helpers

| Function | Arguments | Return | Use |
| --- | --- | --- | --- |
| `TwinU32ToU64::PackDoubleUnsigned32In64` | low u32, high u32 | u64 | Parent handle and sibling-locator packing. |
| `ExtractLow32Of64` | u64 | u32 | Extract slot / previous locator depending on context. |
| `ExtractHigh32Of64` | u64 | u32 | Extract generation / next locator depending on context. |
| `Twin28Plus8::IsCarrierValid` | carrier | bool | Validates two 28-bit components. |
| `Twin28Plus8::PackValues` | carrier | optional<u64> | Packs two 28-bit + one 8-bit value. |
| `Twin28Plus8::UnpackUnitToCarrier` | u64 | carrier | Inverse unpack helper. |

---

## 7. APC header lifecycle

There are **two different lifecycle/control systems**. Do not merge them conceptually:

1. APC header lifecycle: `FREE / RESERVED / LIVE / RETIRED / HAULTED` plus a sequence counter.
2. Handle-table generation cell: generation + active-use count + closed bit.

A node should normally be considered safely usable only when the lifecycle and expected generation are both valid/open.

### `DescriptionOfAPC::SeqLockAndStateStruct`

Fields: `SeqLock`, `StateOfTheAPC`, `IsValid`. The source enforces: RESERVED ⇢ odd sequence; non-RESERVED ⇢ even sequence.

| Function | Args | Return | What it does | Do not use incorrectly |
| --- | --- | --- | --- | --- |
| `ComposeSeqLockAndState(files)` | mutable `SeqLockAndStateStruct&` | packed u64 or sentinel | Validates sequence/state pairing then packs sequence and state into one 64-bit word. | Do not bypass `ValidateStateAgainstSeqLock` by hand-packing arbitrary state. |
| `GetSeqLockAndLifeCycle(raw,values)` | raw u64 + output struct | bool | Unpacks low sequence/high state and validates parity/state. | False means the packed lifecycle cannot be trusted. |
| `ValidateStateAgainstSeqLock(files)` | struct ref | bool | Checks sentinel/range and parity rule. | Does not check whether a transition from previous state is legal. |
| `IsTransitionStateLeagal(current,desired)` | two states | bool | Allows only the explicitly enumerated lifecycle transitions. | Do not add direct FREE→LIVE or LIVE→RETIRED transitions outside this state machine. |

Legal transitions in this snapshot:

```text

FREE ──► RESERVED ──► LIVE

 ▲          │          │

 └──────────┘          ├──► RESERVED ──► RETIRED

                       └──► HAULTED ──► LIVE

RETIRED ──► RESERVED

```

### `HeaderOrchestrator::InitializeDefaultHeaderBuffer`

| Argument | Meaning |
| --- | --- |
| `APCMetaBuffer& header` | 8-cell temporary header to fill. |
| `uint32_t apc_slot_idx` | Slot identity written to header cell 1. |
| `uint32_t capacity_of_apc` | Validated against APC minimum/32-bit limit; not stored directly by this function. |

**Returns:** `true` after zeroing and writing magic, slot, and EOF marker; `false` for invalid capacity/slot. **Caller:** APC creation path through `InitiateAPCMetaHeader`. **Do not call:** on an arbitrary live slot to “reset” it; lifecycle/relations/schema would be bypassed.

---

## 8. Schema system

A schema describes **how bytes in an APC region are interpreted**, not where an APC lives in the Fabric. The schema row is fixed-size (40 bytes), trivially copyable/destructible, and stored in `MATRIX_VIEW_TABLE`.

### Protocols

| Protocol | Contract | Supported by generic `BuildAViewOverRegion`? |
| --- | --- | --- |
| `PRIVATE_REGION` | Single record; direct mutable access only while APC use is held. | Yes. |
| `IMMUTABLE_SNAPSHOT` | Single record intended as immutable read snapshot. | Yes for typed view; `RawMutableSpan()` refuses mutation. |
| `ATOMIC_WORD_ARRAY` | Single record; element access via `atomic_ref`. | Yes. |
| `MPMC_FIXED_RECORD_QUEUE` | Power-of-two record ring with per-slot sequence metadata. | Storage/schema initialization exists; generic `RegionView` does **not** expose queue operations in this snapshot. |
| `DOUBLE_BUFFERED` | Exactly two records; one published/read bank and one write bank. | Storage/schema exists; generic `RegionView` constructor does **not** expose it. |

### Dtypes

The schema enum includes `UINT8/16/32/64`, `INT8/16/32/64`, `FLOAT16/32/64`, and `CHAR`. `CppTypeToRegionDType<T>()` maps native integer types, `float`, `double`, and `char`; there is no native C++ half type mapping in this snapshot even though `FLOAT16_T` exists as a descriptor.

### `RegionSchemaRecord`

| Field | Meaning |
| --- | --- |
| `CellOffset` | Offset, in 64-bit cells, from APC slot start to the region record storage. |
| `CellCount` | Total allocated cells for all protocol records. |
| `MatrixHeight` / `MatrixWidth` | Logical typed matrix shape per record. |
| `EnqueuePosition` / `DequeuePosition` | Protocol state for queue/double buffer, otherwise sentinel. |
| `Region` | Logical `MacroColumnOfAPC`. |
| `Protocol` | Access/storage protocol. |
| `Dtype` | Stored element type. |
| `Flags` | Alignment/layout/protocol flags. |
| `SeqLockCounter` | Schema/protocol sequence field. |

### Schema helper API

| Function | Args | Return | What it does / caller | Misuse warning |
| --- | --- | --- | --- | --- |
| `RegionSchemaCellCount()` | none | `size_t` | Returns 40-byte record size in 64-bit cells (5). Used to size matrix view table. | Not region payload size. |
| `IsMPMCQueue(record)` | schema | bool | Protocol predicate. | Does not validate queue geometry. |
| `HasSchemaFlag(current,desired)` | flags | bool | Bit test. | Do not compare multi-bit aggregate as equality. |
| `IsKnownSchemaFlags(flags)` | flags | bool | Rejects undefined bits / unassigned sentinel. | Known flags may still be semantically incompatible with protocol. |
| `IsValuePowOfTwoU32(value)` | u32 | bool | Power-of-two ≥2 check for queue records. | Not for arbitrary zero/one power-of-two semantics. |
| `CppTypeToRegionDType<T>()` | template type | optional dtype | Maps C++ type to schema dtype. | Unsupported type returns nullopt; do not reinterpret anyway. |
| `DTypeByteCount(dtype)` | dtype enum | optional<u8> | Element byte width. | `FLOAT16_T` reports 2 bytes but native view mapping is absent. |
| `MatrixByteCount(schema)` | schema | optional<u64> | Checked height×width×dtype size. | Failure can mean invalid dtype/zero geometry/overflow. |
| `MatrixCellCount(schema)` | schema | optional<u32> | Rounds matrix bytes up to 64-bit cells. | Not protocol-stride aligned yet. |
| `RecordStrideCells(schema)` | schema | optional<u32> | Adds MPMC sequence cell where needed then rounds each record to 64-byte region alignment. | Do not compute record address from raw matrix-cell count. |
| `LogicalRecordCount(schema)` | sealed schema | optional<u32> | CellCount ÷ stride; validates divisibility. | Only meaningful after schema is sealed. |
| `AlignRegionCells(cell)` | cell count/index | u32 | Rounds up to 64-byte alignment in 64-bit cells. | Assumes result fits u32. |
| `ValidateStortedRegionSchema(schema,apc_cells,batch)` | stored schema + slot/batch geometry | bool | Validates offset alignment/bounds, BATCHED_LAST_DIM width and protocol record rules. | Use exact source spelling. Does not compare against a model-specific expected schema. |
| `FreshProtocolState(record)` | schema | bool | Checks initial enqueue/dequeue state expected for protocol. | Only a protocol-state check. |
| `SchemaDefinition::SealDesiredSchema(schema,record_count)` | schema + requested protocol record count | bool | Calculates stride/count and sets final `CellCount`; resets schema on invalid input. | Call before trying to place an unsealed custom schema. |
| `MakeDisabledSchemaTable(table)` | schema table | void | Initializes all logical columns with `REGION_DISABLED`. | You still need to configure and seal regions you want active. |

### Internal `SchemaDefinition` helpers

| Helper | Purpose | Expected caller |
| --- | --- | --- |
| `AttachPrivateFloat32ToTable_` | Creates one private FLOAT32 matrix row with selected flags, then seals it. | `GHGFLayerModel::MakeDefaultGHGFStorageProfile`. |
| `GetActiveMaskOfRegionTable_` | Builds active bit mask from non-disabled rows. | Model profile builders / validators. |
| `RequiredCellsForSchemaTable_` | Places enabled regions at aligned offsets and returns required APC capacity. | Model profile builder. |
| `SetRecords_` | Applies protocol-specific record count and queue/double-buffer positions. | `SealDesiredSchema` only. |

---

## 9. Typed region views

### `ResolveRegionBiteView`

Internal byte-level resolution result: `{ Bytes, Schema, RegionOrdinal }`. `IsValid()` requires nonempty bytes, schema pointer and nonzero matrix shape. `ByteCount()` returns `span<byte>::size_bytes()`.

### `APCStorageGeometry`

| Function | Args | Return | Purpose |
| --- | --- | --- | --- |
| `BytesPerLocalAddressUnit()` | none | `size_t` | Returns 8 because APC local addressing is in u64 cells. |
| `ByteOffsetOfLocalIndex(local_idx)` | cell index | bytes | Converts local cell offset to byte offset. |
| `ByteCountOfLOcalSpan(local_span)` | cell count | bytes | Converts cells to bytes. |
| `CanInstallTypedSpan<T>(region)` | resolved byte region | bool | Checks schema dtype, byte divisibility and alignment for T. |
| `CanInstallAtomicSpan<T>(region)` | resolved region | bool | Adds `atomic_ref<T>::required_alignment` checks. |
| `InitializeFreshRegionObject<T>(region)` | resolved region | bool | Placement/default-constructs each T in fresh storage; only for trivially copyable/default-constructible T and freshly initialized region. |

### `RegionView<T>`

A move-held access object combining a typed span with an `APCUseScope`; its lifetime therefore pins the APC generation.

| Function | Args | Return | Who should use | What it does / how not to use |
| --- | --- | --- | --- | --- |
| `IsValid()` | none | bool | Any region client. | Requires held use token and nonempty span. |
| `Size()` | none | size_t | Model/kernel code. | Returns element count only while use token is valid. |
| `GetProtocol()` | none | protocol enum | Protocol-aware caller. | Inspect access policy before choosing mutation operation. |
| `RawMutableSpan()` | none | optional<span<T>> | Single-owner/private region algorithms. | Only succeeds for `PRIVATE_REGION`; intentionally refuses immutable/atomic regions. |
| `AtomicLoad(idx,order)` | element index + memory order | T | Atomic-word region clients. | Returns default T on invalid use/protocol/index; validate the view and index rather than treating default as diagnostic. |
| `AtomicStore(idx,value,order)` | index/value/order | bool | Atomic-word region clients. | False on wrong protocol/index/use. |
| `AtomicCompareExchangeStrong(idx,expected,desired,success,failure)` | CAS args | bool | Atomic-word coordination. | `expected` follows normal CAS semantics and may be updated on failure. |

### `RegionViewConstructor`

| Function | Args | Return | What it does | Caller / misuse |
| --- | --- | --- | --- | --- |
| `ResolveRegionView_` | logical column, record ordinal, output byte view | bool | Maps logical region→compact schema row, validates stored schema, computes selected record stride/data range and returns bytes. | Private implementation. Do not calculate raw addresses in model code instead. |
| `BuildAViewOverRegion<T>` | column, record ordinal=0 | optional<RegionView<T>> | Acquires APC use, resolves schema, validates protocol/type/alignment, returns typed view. | Primary generic data-access API. Rejects MPMC/double-buffer protocols in this snapshot. |
| `ZeroARegion<T>` | column | bool | Builds a view and zeroes private elements directly or atomic-word elements with relaxed stores. | Do not use on immutable, queue, or double-buffered region. |

---

## 10. Fabric-to-APC binding

### `FabricToAPCLinker::SeqLockedOperation`

`FOUND`: a stable requested relation/value was obtained. `NONE`: stable state says no result / invalid for this lookup. `RETRY`: the read could not establish a stable snapshot within the attempts. **RETRY is not corruption and should not be reinterpreted as NONE.**

### `RelationOparation`

Carries an optional `APCUseScope`, the relation locator, and the mutation/read operation result. When you need to distinguish “no relation” from “concurrent instability”, pass this output object to the public find APIs.

| Function | Args | Return | What it does | Who uses / how not to use |
| --- | --- | --- | --- | --- |
| `GetThisSlotIdx()` | none | u32 slot or sentinel | Returns bound slot only if APC is active. | Public/model code; do not cache forever across retirement. |
| `IsActiveAPC()` | none | bool | Checks the facade is Fabric-bound and expected generation is still open. | Public sanity check; not a substitute for holding `APCUseScope` during data access. |
| `IsFabricBound_()` | none | bool | Checks runtime cache pointers/slot/generation are structurally present. | Internal only; does not prove generation is currently open. |
| `AcquireAPCUse_()` | none | `APCUseScope` | CAS-increments active-access count only if generation matches and is not closed. | Internal/model friend code. Always keep the returned scope alive while using raw APC data. |
| `ReleseFabricBindingOnly_()` | none | void | Clears runtime facade cache without changing persistent slot state. | Use for unbinding/cleanup, not as retirement. |
| `BindExternalRawFabricBacking_(raw,owner,slot,generation_cell,expected_generation)` | runtime binding pieces | bool | Validates Fabric/slot/schema context then binds facade cache. | Internal only. Raw pointer must come from current attached Fabric mapping. |
| `InitiateAPCMetaHeader()` | none | bool | Builds/writes default APC metadata header while slot is in construction state. | Creation path only; not a live-node reset. |
| `ReadAPCMetaUnit(meta_idx,out)` | header enum + output | bool | Reads one APC header cell under valid use/bounds. | Prefer higher-level role/lifecycle APIs when available. |

---

## 11. AdaptivePackedCellContainer public facade

`AdaptivePackedCellContainer` is the **generic node-facing API**. It does not own its data. It delegates topology and retirement to the owning Fabric while carrying slot+generation identity.

| Function | Arguments | Returns | Typical caller | What it does | How not to use |
| --- | --- | --- | --- | --- | --- |
| `AddParent` | `parent`, H/V `edge_table`, `max_tries` | bool | Generic graph/model construction or mutable topology code. | Requires both APCs active in the same Fabric; delegates transactional add. | Do not connect across Fabrics. Do not violate `parent_slot < child_slot`. Do not use invalid/non-edge segment. |
| `RemoveParent` | `parent`, H/V table, retries | bool | Topology mutation. | Transactional unlink from child parent row and parent child list. | Do not assume `false` means the relation never existed; it can also reflect contention/validation failure. |
| `ReplaceParent` | old parent, new parent, H/V table, retries | bool | Structural learning/mutation. | Atomically replaces one child-owned relation while repairing both parent child lists. | Old/new/child must share Fabric; old != new; new must obey topological order. |
| `AttachMyChild` | `child`, H/V table, retries | bool | Parent-oriented convenience API. | Equivalent to `child.AddParent(*this,...)`. | Do not think this creates a different edge representation. |
| `DetachMyChild` | `child`, H/V table, retries | bool | Parent-oriented convenience. | Equivalent to child removing this as parent. | Same concurrency/identity restrictions as `RemoveParent`. |
| `FindParent` | axis, relation ordinal, optional `RelationOparation*`, tries=1 | APC facade (possibly empty) | Traversal/model execution. | Stable child→parent lookup; binds returned parent snapshot to correct generation. | If concurrency matters, inspect output operation; empty facade alone cannot distinguish NONE vs RETRY. |
| `FindFirstChild` | axis, optional result, tries=1 | APC facade | Child-list traversal. | Reads parent child-list head and resolves the child relation locator. | Do not retain returned facade through retirement without revalidation/use. |
| `FindLastChild` | axis, optional result, tries=1 | APC facade | Reverse traversal start. | Uses row tail locator. | Same retry semantics. |
| `FindNextChild` | axis, current relation locator, optional result, tries=1 | APC facade | Cursor traversal forward through sibling locators. | Pass the relation locator returned by previous traversal, not a slot index. |
| `FindPreviousChild` | axis, current relation locator, optional result, tries=1 | APC facade | Cursor traversal backward. | Same locator rule. |
| `Retire` | max tries | bool | Owner/application when node is structurally disconnected and no longer needed. | Delegates safe retirement, then unbinds this facade on success. | Retirement requires no parents/children and no active uses; remove graph relations first. |
| `MyAPCPtr` | none | `AdaptivePackedCellContainer*` | Internal convenience/friend code. | Returns `this`. | Not a persistent identifier; never store pointer in slab. |

---

## 12. Fabric core and slab geometry

### `CoreOfFabricCoordinator::FabricCache`

The first fixed 16×u64 worth of metadata describes the complete image. Key fields include format version, APC count/capacity, slab size, record-book and segment-pool boundaries, K parent capacity, edge row width, active region mask/count, batch geometry, alignment, table start indices, and whether a default schema exists.

**Reserved/unfinished:** `RELATION_WIDTH_OF_FABRIC`, `DEVICE_PLANNER_RECORD_LEN`, and `WORK_RECORD_WIDTH_OF_FABRIC` are zero in this snapshot. Do not document the device planner/work queue as operational yet.

| Function / type | Args | Return | Purpose / warning |
| --- | --- | --- | --- |
| `CoreOfFabricCoordinator::IsValidEdgeTable` | segment enum | bool | Accepts only H and V edge tables. |
| `DefaultFabricAlignment16Cell(value)` | cell index/count | size_t | Rounds to 16-cell (128-byte) boundary for major Fabric tables. |
| `GetStartingOfAnyFabricTable_(segment)` | segment enum | u64 | Computes record-book entry index for that segment, not actual segment start. |
| `DetachFabric::operator bool()` | none | bool | True when detached result has nonnull slab and nonzero cells. |
| `RawPackedCellAllocator::AlignBiteCount_` | bytes, alignment | size_t | Rounds allocation byte count. |
| `DefaultAllocateAtomicCells` | cell count, alignment, user | u64* | Aligned allocation + zero fill using `_aligned_malloc` or `aligned_alloc`. |
| `DefaultFreeAtomicCells` | ptr,count,alignment,user | void | Matching platform free. |

### Physical layout produced by `InitializeFabric`

```text

[ FabricCache ]

      ↓ align

[ Record Book ]

[ H Edge Table ]

[ Matrix / Schema Table ]

[ V Edge Table ]

[ Handle / Generation Table ]

[ Compiled DAG Table ]

[ Device Planner ]   current width 0

[ Work Queue ]        current width 0

      ↓ at least DEFAULT_FABRIC_CONTROLIO_LENGTH=512 cells, aligned

[ APC Segment Pool: slot0 | slot1 | ... ]

```

---

## 13. Edge-table representation

Each node has one row in each H/V table. A row contains **two independent control domains** followed by K `ParentRelation` entries:

```text

row for slot X

  cell 0 : CHILD_LIST control/status/sequence/tail

  cell 1 : PARENT_RELATIONS control/status/sequence

  cells 2... : ParentRelation[0..K-1]

  padding to 64-byte/cache-line multiple

```

The child-owned parent array gives bounded O(K) child→parent access. Each occupied child relation also contains previous/next **relation locators**, linking it into the corresponding parent’s circular/doubly-linked child list for parent→child traversal.

### `EdgeBuilder::ParentRelation`

- `ParentHandle`: packed parent slot + generation.
- `SiblingLocators`: packed previous + next relation locator.
- One relation locator packs a **24-bit child slot** and **8-bit relation ordinal**. Therefore the representable slot space of the relation locator is limited to 2^24 slots in this snapshot.

### `EdgeBuilder` API

| Function | Args | Return | What it does / expected user | Misuse |
| --- | --- | --- | --- | --- |
| `EdgeTableRecordWidth(K)` | configured K | u16 cells | Raw row width rounded to a 64-byte multiple. | Do not manually assume `2 + 2K` without padding. |
| `IsValidConfigurableParentCapacity(K)` | u8 | bool | Requires 1..64. | Compiled dirty/parent mask is 64 bits, so >64 is invalid. |
| `IsValidRelationOrdinal(ord,K)` | ordinal + K | bool | Ordinal bounds check. | Does not prove relation occupied. |
| `IsValidRelationLocator(locator,slot_count,K)` | packed locator + geometry | bool | Checks non-null, decoded slot and ordinal bounds. | Does not validate generation or relation contents. |
| `PackRelationLocator(slot,ordinal)` | 24-bit-representable slot + ordinal | u32 | Packs cursor locator. | Caller must ensure slot fits relation-slot bit width. |
| `RelationSlot(locator)` | locator | u32 | Decodes owner child slot. |
| `RelationOrdinal(locator)` | locator | u8 | Decodes child relation ordinal. |
| `MakeParentHandle(slot,generation)` | parent identity | u64 | Packs durable relation identity. |
| `ParentSlot(relation)` | relation | u32 | Decodes slot from `ParentHandle`. |
| `ParentGeneration(relation)` | relation | u32 | Decodes generation. |
| `PreviousLocator` / `NextLocator` | relation | u32 | Decodes sibling cursors. |
| `SetSiblingLocators` | relation,prev,next | void | Updates both sibling cursor halves. |
| `MakeParentRelation` | parent slot/gen, prev,next | relation | Builds full relation record. |
| `IsEmpty` | relation | bool | Both handle and siblings are sentinel. |
| `IsPartiallyEmpty` | relation | bool | Exactly one of handle/sibling pair is sentinel; signals inconsistent record. |
| `Clear` | relation | void | Restores sentinel/default relation. |
| `CanInsertCombinedDAGRelation(parent,child)` | slots | bool | Implements **parent < child** invariant. | This is the DAG-safety rule; bypassing it can allow cycles. |
| `NextSequence(current)` | 30-bit logical sequence | u32 | Wraps sequence inside 30 bits. |
| `PackEdgeHeader(edge)` | `EdgeData` | u64 | Packs tail + 30-bit sequence + 2-bit status. |
| `UnpackEdgeHeader(raw)` | u64 | `EdgeData` | Decodes and validates status/parity/free-tail invariant. |
| `DirtyBit(ordinal)` | 0..63 | u64 mask | Compiled parent occupancy bit. |
| `ControlOffset(domain)` | PARENT_RELATIONS/CHILD_LIST | u16 | Returns row control-cell offset. |
| `RawEdgeTableRecordWidth(K)` | K | u16 cells | Unpadded 2 + K×2-cell relation width. |
| `IsParentEmpty` | relation | bool | Only parent-handle sentinel check. |
| `AreSiblingsEmpty` | relation | bool | Only sibling-locator sentinel check. |

---

## 14. Handle/generation and retirement control

### `HandleOfAPCStatic` 64-bit control word

```text

bits  0..31 : ActiveAccess (32 bits)

bits 32..62 : Generation   (31 bits)

bit      63 : Closed

```

| Function | Args | Return | Purpose / warning |
| --- | --- | --- | --- |
| `MakeControlCell(values)` | generation, active count, closed | u64 | Packs control word. Internal control construction. |
| `ReadControlCell(raw)` | u64 | `ControlValues` | Decodes generation/count/closed. |
| `IsGenerationValid(gen)` | u32 | bool | Accepts generation 1..MAX_GENERATION. |
| `NextGeneration(gen)` | u32 | u32 | Returns gen+1 until max; returns 0 at wrap. | Generation wrap is not silently reused; 0 is invalid. |
| `IsOpenGeneration(raw,expected)` | control + expected gen | bool | Exact generation match and `Closed==false`. |
| `CellOffset(slot)` | slot | size_t | Handle-table row offset for one-cell-per-slot table. |

### `APCRelocationDef::ValidateFabricCache`

Validates a copied/borrowed image before attachment: format version, supplied slab size, APC/slot bounds, 24-bit relation-slot capacity, K, edge row width, active schema mask/count, batch, matrix-row geometry, record-book bounds, segment-pool arithmetic, and exact slab end. **This is format/layout validation, not cryptographic integrity or crash-consistency validation.**

---

## 15. Fabric table constructors

### 15.1 `FabricConstructor` — raw slab primitives

| Function | Arguments | Return | What it does / expected caller | Do not use |
| --- | --- | --- | --- | --- |
| `ReadAFabricU64Directly` | slab index, out | bool | Bounds-checked non-atomic load. Internal construction/quiescent code. | Do not race with concurrent writers. |
| `AtomicallyLoadReadAUnit` | index,out | bool | Acquire `atomic_ref<uint64_t>` load. | Still requires correct higher-level protocol. |
| `DirectlyStoreFabricUnit64` | index,value | void | Bounds-checked direct store. | Never publish concurrent control state through this helper. |
| `AtomicallyStoreU64Fab` | index,value,order | void | Atomic control/data store with requested order. | Memory order must match protocol. |
| `CompareExchangeStrongFromFabric` | index, expected&, desired, orders | bool | Strong atomic CAS on one slab cell. | Expected follows CAS mutation semantics. |
| `CompareExchangeWeakInSlab` | same | bool | Weak CAS for retry loops. | Caller must loop as appropriate. |
| `ForceNxLenMemCopy` | start,count,source | bool | Bounds/overlap checks then `memcpy` into slab. | Not atomic publication; construction/quiescent usage only. |
| `IsDesiredIndexValidInSLab` | index | bool | Tests active cache/pointer and index<slab cells. |
| `SlotBegin_` | slot | size_t | Computes absolute slab-cell start of APC slot. |
| `IsInternalBuffer<T>` | pointer,count | bool | Detects overlap with Fabric slab; GHGF uses it to reject self-aliasing external IO. | Do not interpret false as pointer generally safe; only means not overlapping slab. |

### 15.2 `MatrixViewConstructor`

| Function | Args | Return | Purpose |
| --- | --- | --- | --- |
| `MetrixViewRow_(slot)` | APC slot | span<RegionSchemaRecord> | Returns compact schema row for slot. |
| `ConstructMatrixViewRecords_(begin,end)` | table bounds | bool | Placement-constructs schema records for full table. |
| `PrepareMatrixViewRow_(slot,requested)` | slot + logical schema table | bool | Compacts active schemas, assigns aligned offsets inside slot, validates final stored records and active mask. |
| `ClearMatrixViewRow_(slot)` | slot | void | Resets this slot’s schema row. |
| `InitializeRegionProtocolStorage_(slot)` | slot | bool | Initializes protocol-dependent data metadata; notably MPMC per-record sequence cells. |

### 15.3 `APCHandleAndRetirement`

| Function | Args | Return | Purpose |
| --- | --- | --- | --- |
| `GetAPCGenerationPtr_(slot)` | slot | u64* or null | Pointer to this slot’s generation/control cell. |
| `InitializeAPCGenerationTable_()` | none | bool | Initializes every slot at first valid generation, closed, zero active accesses. |
| `OpenAPCGeneration_(slot,generation)` | slot/gen | bool | CAS closed→open if exact valid generation and zero-active invariant. |
| `CloseAPCGeneration_(slot,generation)` | slot/gen | bool | CAS open→closed only for exact generation and **zero active accesses**. |
| `AdvanceClosedAPCGeneration_(slot,new_gen&)` | slot + output | bool | Requires closed/zero-active; advances generation for reclaimed slot. |
| `ReadFirstFreeAPCIdx_()` | none | optional<u32> | Reads Fabric free-slot cursor. |
| `UpdateFirstFreeIdx_(expected,desired)` | CAS expected reference + desired | void | Attempts to move first-free cursor. |
| `IsOpenAPCGeneration_(slot,gen)` | identity | bool | Acquire-load wrapper around `HandleOfAPCStatic::IsOpenGeneration`. |

### 15.4 `RecordBookConstructor`

| Function | Args | Return | Purpose |
| --- | --- | --- | --- |
| `GetRecordMapCarrierRanges_` | segment, out bounds | bool | Reads record-book begin/end and validates segment range. |
| `IdleAFabricTableClassRangesMemory_` | segment | void | Zeroes/idles one segment range during construction/reset. |
| `WriteARecordBookOfTSCEntry_` | segment, begin,end | void | Writes one begin/end record-book entry. |
| `CheckRecordBookRange_` | segment, expected begin/end | bool | Verifies recorded geometry matches expected layout. |

### 15.5 `APCLifeCycle`

| Function | Args | Return | Purpose / caller |
| --- | --- | --- | --- |
| `GetDescriptionLockIdxInFabric_` | APC slot/description index | optional<u64> | Finds the lifecycle-header cell’s absolute slab index. |
| `GetSegmentPoolRange` | slot | `RangeOfAPC` | Returns absolute begin/end cells of one APC slot. |
| `SwitchDescriptionState` | slot, updated_state(current expected), desired_state, retries | bool | Sequence/state CAS transition using legal-transition table. Reserves with odd sequence and publishes stable states with even sequence; lifecycle constructors/retirement only—do not skip states. |
| `ReadAPCStateAtomically_` | slot | `SeqLockAndStateStruct` | Acquire-loads and decodes APC header lifecycle. |
| `InitAllAPCLifeCycleState` | none | void | Initializes all slot lifecycle cells to FREE with an even sequence. |

### 15.6 `EdgeTableConstructor`

| Function | Args | Return | What it does |
| --- | --- | --- | --- |
| `ReadAnEdgeTableRange_` | H/V, row slot | range | Returns absolute edge-row cell span. |
| `EdgeControlCellIndex_` | H/V,row,domain | size_t or `SIZE_MAX` | Returns absolute control-cell index for parent-relations or child-list domain. |
| `ParentRelations_` | H/V,row | span<ParentRelation> | Typed relation-array span for row. |
| `ConstructParentRelationObjects_` | H/V,row | bool | Placement-constructs K empty relation records. |
| `InitializeEdgeTable_` | H/V | bool | Initializes controls/relations for all rows. Child-list starts FREE/null; parent-relations starts LIVE/empty. |
| `ReadEdgeControl_` | H/V,row,domain,out EdgeData | bool | Acquire-load + unpack/validate chosen domain. |
| `ReadEdgeHeader_` | H/V,row,out | bool | Convenience child-list-domain read. |
| `ReadParentHandle_` | H/V, child, ordinal, out handle, tries | FOUND/NONE/RETRY | Sequence-validates parent-relations control before/after relation read. |
| `ReserveEdgeDomain_` | H/V,row,domain,required status,out before,tries | FOUND/NONE/RETRY | CASes stable control to RESERVED with next odd sequence. |
| `ReserveEdgeRow_` | H/V,row,required status,out before,tries | operation | Child-list-domain wrapper. |
| `StoreReservedParentHandle_` | H/V,child,ordinal,handle | void | Relaxed store to reserved relation record. |
| `StoreReservedSiblingLocators_` | H/V,child,ordinal,packed siblings | void | Relaxed store while owning reservation. |
| `PublishReservedEdgeDomain_` | H/V,row,domain,before,tail,status | void | Publishes final even-sequence control with release semantics. |
| `PublishReservedEdgeRow_` | H/V,row,before,tail,status | void | Child-list-domain wrapper. |

### 15.7 `CompiledDAGTableConstructor`

`CompiledDAGRecord` contains `ValueParentMask` and `VolatileParentMask`, one 64-bit occupancy mask per axis.

| Function | Args | Return | Purpose |
| --- | --- | --- | --- |
| `CompiledDAGRow_(slot)` | slot | record* | Pointer to one compiled row. |
| `InitializeCompiledDAGTAble_()` | none | bool | Constructs/zeros compiled masks. |
| `CompiledDAGRelation_` | axis,child,ordinal,relation | void | Sets/clears occupancy bit according to whether relation is empty. |
| `ReadCompiledDAGParentMask_` | axis,child,out mask,tries | FOUND/NONE/RETRY | Reads occupancy mask under parent-row sequence validation. |

---

## 16. Fabric initialization, save, attach, detach, shutdown

### `SlabToFabricConverterAndCordinator`

| Function | Arguments | Return | What it does | Who should use / misuse |
| --- | --- | --- | --- | --- |
| constructor | none | object | Starts inactive with no slab. | Fabric/model wrapper. |
| destructor | none | void | Calls `ShutDownFabric()`. | Do not manually free owned slab behind it. |
| `IsFabricActive()` | none | bool | Checks published active flag plus core pointers/geometry. | Public health check, not a whole-image verifier. |
| `InitializeFabric` | slot count, cells/slot, `FabricRegionConfig`, K | bool | Computes all table geometry, allocates/zeros slab, constructs cache/record book/schema/handle/edge/compiled/lifecycle state, then publishes active. | Call only with no concurrent users. Existing Fabric is shut down/replaced by implementation path. K must be 1..64; schema geometry must fit. |
| `ShutDownFabric()` | none | void | Quiesces active Fabric, clears runtime state and frees backing only if ownership is OWNED. | May need active `RegionView`/uses to drain. Do not retain facades/views after shutdown. |
| `SaveFabric(destination)` | external `span<u64>` at least slab cells | bool | Quiesces source, memcpy-copies complete image, reopens original live generations, restores source active status if reopen succeeds. | Destination must not overlap internal slab; save is an in-memory image operation, not crash-consistent persistence. |
| `AttachFabric(raw_cells,cell_count,ownership)` | aligned image pointer, cells, BORROWED/OWNED | bool | Validates copied cache/layout, binds slab, requires handle controls closed+zero-use, reopens generations for LIVE APCs, publishes Fabric active. | Attached pointer must remain valid if BORROWED. Do not attach a live/non-quiesced image. |
| `DetachFabric()` | none | `DetachFabric` record | Quiesces then transfers/returns slab pointer+size+ownership while clearing this coordinator state without freeing. | Caller becomes responsible for returned backing according to ownership semantics. |

### Private/protected coordination helpers

| Function | Purpose |
| --- | --- |
| `AllocatePackedCellRaw_` | Uses configured/default raw allocator. |
| `FreeRawPackedCells_` | Matching free. |
| `ResetScalarsofTheFabric_` | Clears coordinator pointers/state/cursors after shutdown/detach/failure. |
| `ValidateAttachedFabricLayout_` | Cross-checks record-book ranges and table geometry after basic cache validation. |
| `QuiesceFabric_` | Sets Fabric inactive, closes each generation control cell, and waits/yields until active-access counts drain. |
| `ReopenLiveAPCGenerations_` | Requires valid closed/zero-use controls; refuses RESERVED header states; reopens only LIVE APC generations. |
| `RegionT_<T>(slot,cell_offset)` | Internal unchecked-ish typed pointer into slot after higher-level geometry is known. |

### Relocation flow

```text

ACTIVE FABRIC

    │ SaveFabric / DetachFabric

    ▼

FabricInitialized = false

    │

close each generation control cell

    │

wait ActiveAccess == 0

    ▼

QUIESCENT IMAGE  ── memcpy / move mapping ──► new address

                                             │

                                             ▼

                                      AttachFabric

                                      validate layout

                                      rebuild runtime base pointers

                                      reopen LIVE generations

                                             │

                                             ▼

                                         ACTIVE

```

---

## 17. Transactional DAG mutation

`DAGMutationConf` implements a small fixed-size transaction for one structural operation. The transaction can reserve up to 3 control rows and modify up to 5 relation records, sufficient for add/remove/replace plus neighbor repair in this representation.

### Transaction structures

| Struct | Fields / meaning |
| --- | --- |
| `ConditionalParentPublication` | Expected/published child parent-row sequence, published ordinal, sequence mismatch flag, opaque context, callback. Used to couple model-specific parameter publication to the topology transaction. |
| `DAGRowParticipant` | slot, domain, before header, working tail, reservation flag. |
| `DAGRelationDelta` | child slot + ordinal, before/work relation values, dirty flags for parent handle/sibling locators. |
| `DAGMutationTransaction` | selected H/V table, fixed arrays of row participants and relation deltas, counts. |

### Transaction functions

| Function | Arguments | Return | What it does / invariant |
| --- | --- | --- | --- |
| `AddRowParticipant_` | transaction, slot, domain | bool | Adds unique row participant in deterministic slot/domain order. The ordering reduces inconsistent reservation ordering/deadlock-style contention. |
| `FindRowParticipant_` | transaction, slot, domain | pointer/null | Locates previously added row. |
| `FindOrInsertRelationDelta_` | transaction, child slot, ordinal | pointer/null | Creates one before/work copy for an edited relation. |
| `EditReservedParentHandle_` | transaction, child, ordinal | delta* | Gets editable relation delta and marks parent-handle dirty. |
| `EditReservedSiblingLocators_` | transaction, owner parent slot, relation locator | delta* | Resolves locator to child/ordinal and marks sibling field dirty. |
| `ReserveAllRows_` | transaction,max tries | bool | Reserves every participant using required LIVE/FREE domain state. Aborts already reserved rows on failure. |
| `AbortRowTransaction_` | transaction | void | Republishes each reserved control with its original state/tail; does not publish work deltas. |
| `CommitRowTransaction_` | transaction | void | Stores dirty relation data while rows are reserved; updates compiled occupancy when needed; runs conditional publication; publishes parent-relation domains before child-list domains; optionally tracks DAG revision. |
| `ValidateConditionalParentPublication_` | transaction, child, publication* | bool | Ensures current reserved parent-row sequence matches caller’s expected sequence; sets mismatch output. |
| `PrepareConditionalParentPublication_` | transaction,child,ordinal,publication* | void | Records to-be-published even row sequence/ordinal for model callback/result. |

### `ConstructDAGOnEachAxis`

| Function | Arguments | Return | What it does | Misuse |
| --- | --- | --- | --- | --- |
| `SameHeader_` | two EdgeData | bool | Exact stable header equivalence helper. | Internal consistency check only. |
| `SameRelation_` | two ParentRelation | bool | Exact relation equality. | Internal validation. |
| `ScanReservedParentRow_` | transaction, child, wanted handle, other handle, out scan | bool | Scans K relation slots after reservation to find existing target, other parent, and first empty ordinal. | Requires transaction already owns relevant row. |
| `AddParentRelation_` | parent slot/gen, child slot/gen, axis, optional publication, tries | bool | Checks identity and parent<child, reserves child parent row + parent child list, inserts relation, updates circular child list and compiled mask, publishes transaction. | Generic lower-level mutation. Prefer `AdaptivePackedCellContainer::AddParent` unless model needs conditional publication. |
| `RemoveParentRelation_` | parent identity, child identity, axis, optional publication, tries | bool | Reserves affected child/parent/neighbor rows, unlinks relation, repairs sibling list, clears child relation, compiled mask. | Relation must exist; caller identities must be current generations. |
| `ReplaceParentRelation_` | old parent identity, new parent identity, child identity, axis, optional publication, tries | bool | Atomically unlinks from old parent list and inserts same child relation into new parent list while preserving ordinal when appropriate and repairing neighbors. | New parent must satisfy parent<child; old/new distinct and generations open. |

### Mutation linearization intuition

Relation words are edited while the affected control domains are `RESERVED` (odd sequence). Readers seeing RESERVED retry. Commit then publishes stable even-sequence controls. Parent-relation domains are published before child-list domains so the child-owned relation identity becomes stable before it is exposed through a parent traversal list.

---

## 18. APCFinilizer

`APCFinilizer` is the final generic authority that turns the raw Fabric machinery into create/get/find/retire semantics for `AdaptivePackedCellContainer`. Model layers should normally inherit/use this boundary rather than reaching into the table constructors.

| Function | Arguments | Return | Who uses it | What it does | How not to use |
| --- | --- | --- | --- | --- | --- |
| `CreateAPC` | desired facade, schema table, retries, `override_table=false` | bool | Generic model/Fabric wrapper. | Chooses FREE/reclaimable slot, prepares/reuses schema, binds facade, initializes header, reserves empty H/V domains, transitions lifecycle RESERVED→LIVE, opens generation, publishes row domains. Rolls back on failure. | Do not pass an already active facade. Do not use an incompatible override merely to change physical geometry under a default Fabric policy. |
| `GetExistingAPC_` | slot, output facade, output use, optional expected generation | bool | Derived model wrappers such as GHGF. | Acquires generation-safe use then binds a snapshot facade. | Keep returned `APCUseScope` alive while using raw/model data. |
| `BindExistingAPCSnapshot_` | slot, facade, optional generation | bool | Traversal/resolution internals. | Binds to current stable generation without returning long-lived active-use token. | Caller must still respect generation validation/use around actual memory access. |
| `GetASlotForNewAPCLink()` | none | optional slot | `CreateAPC`. | Finds FREE slot or reclaims eligible RETIRED slot and advances generation as needed. |
| `ResolveChildLocator_` | parent identity, axis, relation locator, out child | FOUND/NONE/RETRY | Child traversal. | Validates locator and bound child generation. |
| `FindParent_` | child identity, axis, ordinal, optional result, tries | APC facade | Public APC wrapper. | Sequence-validates relation, generation and returns bound parent. |
| `FindFirstChild_` / `FindLastChild_` | parent identity, axis, result, tries | APC facade | Traversal. | Reads stable child-list control/tail and resolves locator. |
| `FindNextChild_` / `FindPreviousChild_` | parent identity, axis, current locator, result, tries | APC facade | Cursor traversal. | Validates current relation still belongs to parent/generation and follows sibling locator. |
| `RetireAPC_` | slot,generation,retries | bool | `AdaptivePackedCellContainer::Retire`. | Requires no H/V parents and no children, reserves four domains, closes generation only after active uses are zero, transitions LIVE→RESERVED→RETIRED, publishes empty domains. | Do not retire a still-linked node. Do not force-close while views/users are active. |
| `ReclaimRetiredSlotTemp_` | slot | bool | Allocation path. | Transitions RETIRED→RESERVED, checks relations remain empty/free, advances closed generation (ABA barrier), clears slot payload/header except lifecycle construction state. | Not public deletion API. |
| `IsNodePolicyReConfigurable_` | schema table | bool | `CreateAPC` override policy. | Allows only compatible physical geometry/dtypes/flags/active pattern; permits limited protocol changes among private/immutable/atomic. | Does not permit arbitrary queue/double-buffer or geometry mutation under default schema. |

### `CreateAPC` high-level flow

```text

request facade + schema

        │

        ▼

choose FREE slot OR reclaim RETIRED slot

        │   (reclaim advances generation)

        ▼

prepare schema row / protocol storage

        │

bind runtime facade cache

        │

initialize APC header

        │

reserve H/V parent + child-list controls

        │

verify rows are empty

        │

header RESERVED → LIVE

        │

open generation cell

        │

publish H/V domains stable

        ▼

LIVE APC

```

---

## 19. End-to-end APC/Fabric flows

### A. Initialize a generic Fabric

1. Build a `FabricRegionConfig` and compatible sealed schema table in the derived/model layer.
2. Call `InitializeFabric(slot_count, slot_cell_count, region_conf, K)`.
3. If using a default schema policy, set `DefaultRegionTable_` from the model wrapper before/around initialization according to that wrapper’s implementation.
4. Fabric computes tables, allocates one slab, initializes every control/table, then becomes active.

### B. Create a node

1. Caller owns an empty `AdaptivePackedCellContainer` facade object.
2. `CreateAPC` selects a slot and binds it.
3. Header/schema/lifecycle/generation become LIVE/open only when construction succeeds.
4. Model-specific node initialization writes its role/state/weights through typed region views.

### C. Add relation

```text

child.AddParent(parent, H or V)

  → facade checks both active + same Fabric

  → AddParentRelation_(slot+generation identities)

  → enforce parent_slot < child_slot

  → reserve affected rows

  → edit child relation + parent sibling list

  → compiled parent mask update

  → publish stable even-sequence controls

```

### D. Read relation

```text

FindParent(child, ordinal)

  load parent-row control BEFORE

  if RESERVED → retry

  read relation payload

  load control AFTER

  if changed → retry

  validate parent generation

  bind returned APC facade

```

### E. Retire / reuse

Remove all H/V links → release all RegionViews/use scopes → `Retire()` → lifecycle RETIRED + generation closed. A later create may reclaim the slot, advance generation, clear slot storage, and construct a new logical node. An old facade still expects the prior generation and therefore fails open-generation validation.

---

## 20. GHGF storage model

GHGF is a **use-case/model layer**, not part of generic Fabric semantics. It maps its mathematics into the generic APC regions and H/V topology.

### `GHGFLearningConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `HCouplingLearningRate` | 1e-3 | Local H-coupling update rate. |
| `DriftLearningRate` | 1e-3 | Tonic drift/bias update rate. |
| `VolatilityLearningRate` | 1e-4 | Tonic log-volatility update rate. |
| `VCouplingLearningRate` | 1e-4 | V-coupling update rate. |
| `AutoConnectionLearningRate` | 1e-4 | Temporal autoregressive connection update rate. |
| `GradientClip` | 10 | Absolute local signal/update clipping scale. |
| `MinTonicLogVolatility` / `MaxTonicLogVolatility` | -20 / 10 | Clamp bounds for tonic log-volatility. |

### GHGF node roles

- `OBSERVATION`: observed/binary output node in current formulation.
- `VALUE`: latent value state.
- `VOLATILE`: latent volatility state.

### State rows

| Row | Purpose |
| --- | --- |
| `MEAN` | Current posterior/state mean. |
| `EXPECTED_MEAN` | Prediction before assimilation. |
| `PRECISION` | Posterior/current precision. |
| `EXPECTED_PRECISION` | Predicted precision. |
| `CONDITIONAL_EXPECTED_PRECISION` | Transition-conditional precision before marginal/effective transformations. |
| `OBSERVED` | Observation gate/mask. |
| `CURRENT_VARIANCE` | Current variance helper. |
| `EFFECTIVE_PRECISION` | Volatility-sensitive effective precision term. |

### Error / parameter / message layout

| Area | Indices |
| --- | --- |
| ERROR | 0 `VALUE_PREDICTION_ERROR`; 1 `VOLATILE_PREDICTION_ERROR`. |
| WEIGHT fixed | 0 `TONIC_VOLATILE`; 1 `TONIC_DRIFT`; 2 `AUTO_CONNECTION`. |
| WEIGHT couplings | `FIRST_COUPLING_INDEX=3`; H occupies next K; V occupies following K. `CouplingIndex()` maps axis+ordinal. |
| BOTTOM_UP / FF | OBSERVATION, VALUE_FACTOR, VALUE_GAIN, VALUE_ERROR, EFFECTIVE_PRECISION, VALUE_LEARNING_SIGNAL, VOLATILE_ERROR. |
| TOP_DOWN / FB | MEAN, EXPECTED_MEAN, EXPECTED_PRECISION. |

### Default GHGF storage profile

`MakeDefaultGHGFStorageProfile` activates exactly BOTTOM_UP, TOP_DOWN, STATE, ERROR, and WEIGHT. FF/FB/STATE/ERROR are private FLOAT32 matrices shaped `[row_count × batch_capacity]` with `BATCHED_LAST_DIM`; WEIGHT is a private FLOAT32 `[1 × parameter_count]` region and is not batched. This is the key vectorization geometry: for a fixed row, batch lanes are contiguous.

### `GHGFLayerModel` helper API

| Function | Args | Return | What it does / warning |
| --- | --- | --- | --- |
| `CouplingIndex(axis,ordinal,K)` | H/V, relation ordinal, capacity | u32 index or sentinel | H maps to `3+ordinal`; V maps to `3+K+ordinal`. | Only call with a valid relation ordinal/axis. |
| `MakeDefaultGHGFStorageProfile(profile,batch,K,is_default)` | output profile + geometry | bool | Builds/seals default schema, active mask, parameter count and required APC cell count. | Profile must be validated before Fabric construction. |
| `IsValidStoregeProfile(profile)` | profile | bool | Checks profile internal consistency/schema geometry. |
| `IsHCouplingParameter(index,K)` | weight index/K | bool | Identifies whether parameter index belongs to H coupling range. |

---

## 21. GHGF classes and API

### 21.1 `GHGFNode`

`GHGFNode` protected-inherits `AdaptivePackedCellContainer`; external code does not receive generic graph APIs directly through this class. `GHGFFabric_` is a runtime pointer to the owning model and is **not relocatable persistent state**.

| Function | Arguments | Return | Caller | What it does | Do not use |
| --- | --- | --- | --- | --- | --- |
| `InitializeGHGFNode` | node role | bool | `GHGFModel::CreateNodeOfGHGF`. | Builds STATE/ERROR/WEIGHT views, initializes all rows/parameters, writes role cell, initializes default couplings, invalidates prepared model. | Do not call on arbitrary non-GHGF APC/schema. |
| `GHGFRole_` | none | optional role | Model internals. | Reads/validates role from APC header. |
| `IsLiveGHGFSlot_` | none | optional role | Model validation. | Acquires APC use then gets role. |
| `ResetAPCGHGFStateRegion_` | none | void | Model reset. | Clears transient FF/FB/state/error and restores initial belief values while preserving learned weights. |
| `PublishFBackwardMessageGHGF_` | batch | void | Prediction node. | Copies state mean/expected mean/expected precision into TOP_DOWN message rows. |
| `PublishFForwardMessageGHGF_` | batch | bool | Update node. | Computes value/error/learning/effective-precision/volatile-error signals for propagation and learning. |
| `PredictGHGFNodenNONVectorized_` | batch, retries | bool | Model prediction loop. | Reads stable parent execution snapshots; predicts observation or latent mean/precision/volatility; publishes backward messages. |
| `UpdateGHGFNodeNONVectorized_` | batch,retries | bool | Reverse update loop for latent nodes. | Assimilates local value error into posterior mean/precision and volatile error, then publishes FF. |
| `PropogateGHGFErrorNONVectorized_` | child slot,batch,retries | bool | Reverse graph propagation. | Reads child H/V parent snapshots and accumulates precision/correction contributions into parent errors/state. |

### 21.2 `GHGFModel`

`GHGFModel` protected-inherits `APCFinilizer`; it is the model-facing wrapper over generic slab/APC mechanics.

| Function | Arguments | Return | Who uses | What it does | How not to use |
| --- | --- | --- | --- | --- | --- |
| `IsGHGFModelReady_` | none | bool | All GHGF public execution APIs. | Requires active Fabric and sealed/prepared model. |
| `GHGFRegion_` | slot, cell offset | float* | Internal math kernels. | Computes typed region base directly from known sealed profile offset. | Never call with arbitrary unvalidated offset. |
| `GHGFStateRow_` / `GHGFErrorRow_` / `GHGFWeight_` / `FFRowGHGF_` / `FBRowGHGF_` | slot + row as applicable | float* | Internal scalar kernels. | Translate profile-cached offsets + row×batch geometry. | Runtime raw pointers require active attached slab. |
| `GetGHGFNode_` | slot, out node, out use | bool | Model loops. | Gets existing APC generation safely, binds GHGF facade, validates role. |
| `InvalidateGHGFModel_` | none | void | Construction/topology changes. | Clears prepared/sealed flag. |
| `GHGFParentMask_` | slot, H/V | u64 | Static/compiled execution. | Returns compiled parent occupancy mask. |
| `GetGHGFParameter_` | slot,index | optional<float> | Fitter/learning internals. | Reads allowed parameter. |
| `SetGHGFParameter_` | slot,index,value | bool | Fitter/internal parameter code. | Checks slot/index/value and observation-node parameter restrictions. |
| `ReadGHGFNodeIdentity_` | slot,out role,out generation | bool | Seal/structural validation. | Reads current role and generation. |
| `ReadGHGFParentExecutionSnapshot_` | child,axis,out snapshot,max tries | FOUND/NONE/RETRY | Prediction/update/structural learning. | **Pay-for-mutation path:** static topology reads compiled mask/relations/weights directly; structural-learning mode validates sequence/control before and after atomic snapshot reads. |
| `ConnectGHGFParent` | `GHGFConnection` | bool | Construction/static model editing. | Validates GHGF role restrictions, adds generic relation, locates child-owned ordinal, stores coupling, invalidates model. | Once model is ready with structural learning active, this direct API is rejected; use structural-learning mutation API. |
| `RemoveParent` | `GHGFConnection` | bool | Construction/static topology edit. | Generic remove + invalidates model. | Same structural-active restriction. |
| `InitializeGHGFFabric` | slot count, validated storage profile | bool | `ConstructGHGFModel`. | Stores profile/default schema, initializes generic Fabric, caches region offsets. |
| `ResetGHGFState` | none | bool | Training/evaluation caller. | Resets every live GHGF node state while retaining weights/topology. |
| `CreateNodeOfGHGF` | node facade, role | bool | Model construction. | Calls generic `CreateAPC` with profile then model-specific node initializer; retires/cleans on failure. |

### 21.3 `GHGFModelConstructor` — public execution API

| Function | Arguments | Return | What it does | Expected usage / warning |
| --- | --- | --- | --- | --- |
| `ConstructGHGFModel` | `GHGFModelConstructionValues&`, profile | bool | Validates node/role spans; initializes Fabric; stores structural-learning mode; creates nodes; connects supplied topology; seals/validates; resets state. On failure shuts down and unbinds nodes. | Primary model constructor. Input node facade objects must not already be active. |
| `PredictModelNONVectorized` | batch, output prediction span | bool | Checks readiness/batch/output/aliasing, runs prediction for all nodes then copies observation predictions. | Call before `UpdateModelNONVectorized`/`TrainModelNONVectorized` for each time step. |
| `UpdateModelNONVectorized` | batch, const observations | bool | Validates external binary observation buffer and assimilates observations/latent nodes without parameter learning. | This does not run prediction first. |
| `TrainModelNONVectorized` | batch, observations, learning config | bool | Runs update, then local parameter learning using already predicted/inferred messages. | Normal step is Predict → Train. With structural learning active, nonzero H/V coupling learning rates are rejected because topology/coupling publication is controlled structurally. |
| `RunGHGFSequence` | observations,time count,batch count,prediction output,reset=true | optional<double> | Convenience sequential evaluation: optional reset; per time step predict, score Bernoulli NLL before observation update, optionally copy predictions, then update. | Not an internal-learning training loop; it calls Update rather than Train. Buffers must not alias slab or each other illegally. |
| `FitGHGFParameters` | observations,time,batch,parameter ranges,passes | optional<double> | Slow replay/coordinate search oracle over bounded parameters using `RunGHGFSequence`. | Validation/reference tool, not production local learner. Rejected while structural learning active. |

### Internal `GHGFModelConstructor` functions

| Function | Purpose |
| --- | --- |
| `SealGHGFModel_` | Performs full invariant pass: nodes/roles/schema/weights/edge controls/compiled masks/parent order/observation restrictions/revision stability, then marks model prepared. |
| `PredictGHGFBatchNONVectorized_` | Iterates slots in ascending topological order and calls each node prediction. |
| `CopyGHGFPredictionNONVectorized_` | Copies observation-node `FB::EXPECTED_MEAN` batches into external output. |
| `UpdateGHGFBatchNONVectorized_` | Loads observations, computes observation prediction errors/FF messages, then walks nodes in reverse slot order to update latents and propagate errors. |
| `LearnGHGFBatchNONVectorized_` | Applies batch-mean clipped local updates for drift, auto connection, tonic volatility, H coupling, and V coupling. |

---

## 22. GHGF prediction/update/learning data flow

### Construction

```text

MakeDefaultGHGFStorageProfile

      │

      ▼

GHGFModelConstructionValues

  nodes span

  role span

  connection span

  StructuralLearningActive

      │

      ▼

ConstructGHGFModel

      ├─ InitializeGHGFFabric

      ├─ CreateNodeOfGHGF for every slot

      ├─ ConnectGHGFParent for supplied edges

      ├─ SealGHGFModel_

      └─ ResetGHGFState

```

### One inference/training time step

```text

             ┌──────────────────────────────┐

             │ PredictModelNONVectorized    │

             └──────────────┬───────────────┘

                            │ slots ascending

                            ▼

          stable parent execution snapshots

                            │

                            ▼

             expected mean / precision

                  publish FB messages

                            │

                            ▼

            external prediction available

                            │

             observations for this time step

                            │

                 ┌──────────┴─────────┐

                 ▼                    ▼

       UpdateModelNONVectorized   TrainModelNONVectorized

                 │                    │

                 └──── assimilation ──┤

                                      │

                               local LearnGHGFBatch

                                      ▼

                               updated parameters

```

### Prediction math in simple words

- **Observation node:** start with tonic drift (bias), add each H parent expected mean times its H coupling, then apply sigmoid. That produces predicted probability.
- **Latent VALUE/VOLATILE node:** predicted mean starts from tonic drift + auto-connection × previous mean + H parent contributions.
- **Volatility contribution:** each V parent modifies log variance/volatility using its mean and uncertainty; exponentiation produces process variance and updates predicted/effective precision.
- The node publishes mean, expected mean and expected precision into TOP_DOWN/FB rows for its children to read.

### Observation assimilation

For a binary observation with predicted probability `p` and observed `y`, the current code uses the Bernoulli marginal `p(1-p)`, stores a scaled prediction error `(y-p)/marginal`, and publishes `VALUE_LEARNING_SIGNAL = marginal × value_error`, which reduces to `y-p` when observed. The prediction is scored before the update in `RunGHGFSequence`.

### Local parameter learning

| Parameter | Current local signal (conceptual) | Important behavior |
| --- | --- | --- |
| tonic drift | mean of child value-learning signal | Bias/intercept learning. |
| auto connection | value-learning signal × previous mean | Temporal strength; clamped to [0,1] in current implementation. |
| tonic log volatility | 0.5 × effective precision × volatile error | Clamped to configured log-volatility range. |
| H coupling | child value-learning signal × parent expected mean | One child-owned scalar per H relation ordinal. |
| V coupling | volatility signal × (parent mean + coupling / parent expected precision) | One child-owned scalar per V relation ordinal. |

All signals are batch-reduced, clipped according to `GradientClip`, and written to the **child-owned `WEIGHT_SLOT`**. Learning occurs after inference/update, not while the same batch’s prediction is still being formed.

---

## 23. Online structural learning

### Mode selection: pay for sequence validation only when needed

`GHGFModelConstructionValues::StructuralLearningActive` defaults to `false`. The constructor stores it in `GHGFCache_`.

**When false (static topology):** `ReadGHGFParentExecutionSnapshot_` reads the compiled parent mask, relation records and child-owned coupling values directly and returns `FOUND` without entering the parent-row sequence-validation loop.

**When true (online structural learning):** the same function loads the parent-relation control word with acquire semantics, rejects/rescans `RESERVED` rows, snapshots mask/handles/couplings with atomic loads, then rechecks control so it can return a coherent snapshot or `RETRY`.

This is the documented **pay-for-mutation boundary**. In this snapshot, the mode is selected at model construction/reconstruction; there is no documented public zero-cost live toggle method that changes the mode in place.

### Structural-learning structs

| Type | Fields | Meaning |
| --- | --- | --- |
| `GHGFConcurrentOperation` | SUCCESS, RETRY, STALE, REJECTED | Separates contention from stale optimistic decision and invalid request. |
| `GHGFStructureMutation` | operation, child, old/new parent, axis, coupling, expected row sequence | Client’s compare-against-snapshot mutation request. |
| `GHGFStructureSnapshot` | child, axis, row sequence, parent mask, valid | Minimal decision snapshot returned to structural learner. |
| `GHGFStructureMutationResult` | result, published row sequence, published ordinal | Outcome plus exact relation ordinal/next sequence after successful commit. |
| `GHGFParentExecutionSnapshot` | parent mask, row sequence, up to 64 handles and couplings | Full execution snapshot consumed by GHGF numeric code. |

### `GHGFStructralLearningModel` API

| Function | Arguments | Return | What it does | How to use / not use |
| --- | --- | --- | --- | --- |
| `ReadStructureSnapshotConcurrently` | child, H/V axis, out snapshot, retries | `GHGFConcurrentOperation` | Available only when model ready + structural learning active. Calls sequence-stable execution snapshot and exposes child/axis sequence+mask for a later optimistic decision. | Treat `RETRY` as retryable contention. A successful snapshot can become stale before mutation. |
| `TryApplyGHGFStructureMutation` | mutation with `ExpectedRowSequence`, retries | `GHGFStructureMutationResult` | Re-reads current snapshot; if sequence differs returns STALE; validates operation/capacity/roles/topological rule; invokes generic conditional Add/Remove/Replace; publishes coupling scalar through transaction callback; reports new row sequence/ordinal. | Do not call with a sequence invented by caller. Read a snapshot, make decision, then submit expected sequence. This is predictive structure mutation, not causal-discovery proof. |
| `PublishCoupling_` | opaque publication context + relation ordinal | void | Transaction callback: maps relation ordinal to H/V weight index and atomically stores child coupling scalar. | Private callback only; topology commit controls when it is invoked. |

### Online structural-learning decision flow

```text

ReadStructureSnapshotConcurrently(child, axis)

        │

        ├── RETRY → read again

        └── SUCCESS(seq, parent mask)

                    │

                    ▼

       model-specific structural criterion

       (e.g. validation score in Test I)

                    │

                    ▼

      GHGFStructureMutation{ExpectedRowSequence=seq}

                    │

                    ▼

        TryApplyGHGFStructureMutation

          ├── STALE → decision snapshot expired; recompute

          ├── RETRY → transaction contention; retry policy

          ├── REJECTED → invalid request/invariant

          └── SUCCESS

                 ├── topology committed

                 ├── coupling scalar published

                 └── new row sequence returned

```

### Interaction with ordinary parameter learning

When structural learning is active, `LearnGHGFBatchNONVectorized_` rejects a configuration with nonzero H- or V-coupling learning rates. This avoids two independent mechanisms concurrently owning the same coupling parameters. Drift, auto-connection and tonic-volatility learning may still be configured subject to their normal validation.

---

## 24. Correct usage recipes

### Recipe A — static GHGF model (lowest structural-read overhead)

```cpp

// 1. Build profile

GHGFLayerModel::GHGFStorageProfile profile{};

GHGFLayerModel::MakeDefaultGHGFStorageProfile(profile, batch, K);

// 2. Prepare node/role/connection arrays

// ...

// 3. Structural learning false

GHGFModelConstructor::GHGFModelConstructionValues values{

    nodes, roles, connections, false

};

model.ConstructGHGFModel(values, profile);

// 4. Per step

model.PredictModelNONVectorized(batch, predictions);

model.TrainModelNONVectorized(batch, observations, learning);

```

Static parent execution uses the direct compiled-table path.

### Recipe B — structural-learning-enabled GHGF

1. Construct with `StructuralLearningActive=true`.
2. Set `HCouplingLearningRate=0` and `VCouplingLearningRate=0` in ordinary local learning if topology/couplings are owned by structural mutation.
3. Read `GHGFStructureSnapshot`.
4. Decide add/remove/replace from a model-specific criterion.
5. Submit mutation with the snapshot’s `ExpectedRowSequence`.
6. On `STALE`, re-read and recompute decision; on `RETRY`, apply bounded retry/backoff policy; on `REJECTED`, fix the request.

### Recipe C — generic APC traversal with retry information

```cpp

FabricToAPCLinker::RelationOparation op{};

auto parent = child.FindParent(FabricSegments::VALUE_PARENT_EDGE_TABLE_H, ordinal, &op);

switch (op.MutationOP_) {

case FabricToAPCLinker::SeqLockedOperation::FOUND: /* use parent */ break;

case FabricToAPCLinker::SeqLockedOperation::RETRY: /* retry policy */ break;

case FabricToAPCLinker::SeqLockedOperation::NONE:  /* no stable relation */ break;

}

```

If you omit the result object, an empty returned facade does not communicate *why* it is empty.

### Recipe D — relocation

1. Stop or coordinate external model work.
2. Ensure temporary `RegionView`/APC use scopes can drain.
3. At Fabric layer, `SaveFabric` into an external cell span **or** `DetachFabric`.
4. Map/copy image at new valid aligned address.
5. `AttachFabric` and validate success.
6. Rebuild/rebind any **runtime facades/model caches/pointers** that are not slab state before using higher-level model APIs.

**Important:** generic Fabric relocation is implemented at `SlabToFabricConverterAndCordinator`; the current GHGF public wrapper does not expose a full public GHGF attach/rebind convenience API. Do not assume a copied C++ `GHGFModelConstructor` object can simply point at a new image without its runtime cache being rebuilt.

---

## 25. Misuse and anti-patterns

| Anti-pattern | Why it is wrong | Correct approach |
| --- | --- | --- |
| Persist `AdaptivePackedCellContainer` or `GHGFNode` C++ object bytes as model storage. | Their runtime caches/pointers are address-specific. | Persist/save the Fabric slab image; rebuild facades. |
| Store raw parent/child pointers in the slab. | Breaks relocation and generation identity. | Store slot+generation handle / relation locator. |
| Mutate edge-table cells directly. | Bypasses reservation sequence, sibling repair, compiled masks and transaction publication. | Use `AddParent`/`RemoveParent`/`ReplaceParent` or conditional structural mutation. |
| Treat `RETRY` as `NONE`. | Conflates concurrent instability with stable absence and can make wrong structural decisions. | Retry or propagate retry status. |
| Use only slot number as identity. | Slot reuse can create ABA bug. | Always validate generation for durable relation identity. |
| Hold `RegionView` forever then call save/retire/shutdown. | The view holds an active-use token and can prevent quiescence. | Bound view lifetime tightly. |
| Call raw direct Fabric store/read in concurrent code. | Non-atomic access can race and bypass control protocol. | Use the correct atomic/transactional layer. |
| Insert parent with slot >= child. | Violates the stored topological order / cycle-prevention invariant. | Allocate/order nodes so every legal parent has lower slot. |
| Use >64 direct parents per axis. | Compiled parent mask is one u64; validator rejects it. | Use K<=64 or redesign compiled representation. |
| Assume `FLOAT16_T` means native half views are supported. | Descriptor exists but `CppTypeToRegionDType` has no half mapping in this snapshot. | Add an explicit half type/mapping and atomic/alignment rules before use. |
| Call `BuildAViewOverRegion` for MPMC/double-buffer and expect it to work. | Generic view switch rejects those protocols. | Add/use dedicated protocol API. |
| Change a default-schema node to arbitrary new geometry through override. | `IsNodePolicyReConfigurable_` intentionally limits overrides. | Create a compatible policy override or a Fabric/profile with desired geometry. |
| Retire a linked node. | Would leave graph references dangling. | Remove H/V parent and child relations first. |
| Run H/V local coupling SGD while structural learning owns couplings. | Two writers would own same logical parameters. | Structural mode explicitly rejects nonzero H/V coupling learning rate. |
| Use `FitGHGFParameters` as production online learning. | It is replay-based bounded coordinate search and expensive. | Use local learner; keep fitter as oracle/validation tool. |
| Assume Fabric relocation implies CPU/GPU coherence. | Address independence is not a heterogeneous memory-order guarantee. | Validate shared memory domain + system-scope atomics on target backend first. |

---

## 26. Current limits / unfinished surfaces

| Limit | Current snapshot |
| --- | --- |
| Direct parents per axis | Configurable 1..64; default 8. |
| Relation locator slot bits | 24 bits → at most 2^24 representable slot values in relation locators; relocation validator enforces count bound. |
| Generation bits | 31 bits; generation 0 invalid. At max generation, `NextGeneration` returns 0 rather than silently wrapping. |
| APC minimum size | 128×u64 cells. |
| Persistent format version | `FORMAT_VERSION=1`. |
| Device planner table | Record length zero; placeholder. |
| Work queue | Record width zero; placeholder. |
| Generic queue/double-buffer API | Schema/storage exists; generic `RegionView<T>` does not expose operations. |
| FLOAT16 view | Dtype exists, native C++ type mapping absent. |
| Vectorized GHGF | Current documented model execution functions are explicitly `NONVectorized`. |
| Multiple numeric writers to one GHGF model | Not promised by current public numeric API; online structural test documents one numeric worker with structural activity. |
| Relocation durability | In-memory address relocation proven by logic/tests; no power-failure/crash-consistent persistence contract. |
| Cross-device atomic quiescence | Research goal, not proven by host `std::atomic_ref` implementation. |

---

## 27. Function index by subsystem

This index is intentionally redundant: use it when you know a symbol name but not which section explains it.

### APC / packing

- `APCDataStructure::CountOfMacroColumn`
- `APCDataStructure::RegionBit`
- `APCDataStructure::ValidRegionMask`
- `APCDataStructure::CompactRegionIndex`
- `APCDataStructure::IsValid32BitAPCUnit`
- `APCDataStructure::IsValidFabricUnit`
- `APCDataStructure::InLimitOfUint8`
- `APCDataStructure::IsCapacityOfAPCValid`
- `APCDataStructure::IsPowerOfTwoValue`
- `APCDataStructure::IsValidEven64`
- `APCUseScope::Release`
- `TwinU32ToU64::PackDoubleUnsigned32In64`
- `TwinU32ToU64::ExtractLow32Of64`
- `TwinU32ToU64::ExtractHigh32Of64`
- `Twin28Plus8::IsCarrierValid`
- `Twin28Plus8::PackValues`
- `Twin28Plus8::UnpackUnitToCarrier`

### Header / schema

- `DescriptionOfAPC::ComposeSeqLockAndState`
- `DescriptionOfAPC::GetSeqLockAndLifeCycle`
- `DescriptionOfAPC::ValidateStateAgainstSeqLock`
- `DescriptionOfAPC::IsTransitionStateLeagal`
- `HeaderOrchestrator::InitializeDefaultHeaderBuffer`
- `SchemaValidator::RegionSchemaCellCount`
- `SchemaValidator::IsMPMCQueue`
- `SchemaValidator::HasSchemaFlag`
- `SchemaValidator::IsKnownSchemaFlags`
- `SchemaValidator::IsValuePowOfTwoU32`
- `SchemaValidator::CppTypeToRegionDType`
- `SchemaValidator::DTypeByteCount`
- `SchemaValidator::MatrixByteCount`
- `SchemaValidator::MatrixCellCount`
- `SchemaValidator::RecordStrideCells`
- `SchemaValidator::LogicalRecordCount`
- `SchemaValidator::AlignRegionCells`
- `SchemaValidator::ValidateStortedRegionSchema`
- `SchemaValidator::FreshProtocolState`
- `SchemaDefinition::SealDesiredSchema`
- `SchemaDefinition::MakeDisabledSchemaTable`

### Views / facade

- `APCStorageGeometry::BytesPerLocalAddressUnit`
- `APCStorageGeometry::ByteOffsetOfLocalIndex`
- `APCStorageGeometry::ByteCountOfLOcalSpan`
- `APCStorageGeometry::CanInstallTypedSpan`
- `APCStorageGeometry::CanInstallAtomicSpan`
- `APCStorageGeometry::InitializeFreshRegionObject`
- `RegionView::IsValid`
- `RegionView::Size`
- `RegionView::GetProtocol`
- `RegionView::RawMutableSpan`
- `RegionView::AtomicLoad`
- `RegionView::AtomicStore`
- `RegionView::AtomicCompareExchangeStrong`
- `RegionViewConstructor::BuildAViewOverRegion`
- `RegionViewConstructor::ZeroARegion`
- `FabricToAPCLinker::GetThisSlotIdx`
- `FabricToAPCLinker::IsActiveAPC`
- `AdaptivePackedCellContainer::AddParent`
- `AdaptivePackedCellContainer::RemoveParent`
- `AdaptivePackedCellContainer::ReplaceParent`
- `AdaptivePackedCellContainer::AttachMyChild`
- `AdaptivePackedCellContainer::DetachMyChild`
- `AdaptivePackedCellContainer::FindParent`
- `AdaptivePackedCellContainer::FindFirstChild`
- `AdaptivePackedCellContainer::FindLastChild`
- `AdaptivePackedCellContainer::FindNextChild`
- `AdaptivePackedCellContainer::FindPreviousChild`
- `AdaptivePackedCellContainer::Retire`
- `AdaptivePackedCellContainer::MyAPCPtr`

### Fabric / edge / lifetime

- `CoreOfFabricCoordinator::IsValidEdgeTable`
- `CoreOfFabricCoordinator::DefaultFabricAlignment16Cell`
- `CoreOfFabricCoordinator::GetStartingOfAnyFabricTable_`
- `RawPackedCellAllocator::AlignBiteCount_`
- `RawPackedCellAllocator::DefaultAllocateAtomicCells`
- `RawPackedCellAllocator::DefaultFreeAtomicCells`
- `EdgeBuilder::EdgeTableRecordWidth`
- `EdgeBuilder::IsValidConfigurableParentCapacity`
- `EdgeBuilder::IsValidRelationOrdinal`
- `EdgeBuilder::IsValidRelationLocator`
- `EdgeBuilder::PackRelationLocator`
- `EdgeBuilder::RelationSlot`
- `EdgeBuilder::RelationOrdinal`
- `EdgeBuilder::MakeParentHandle`
- `EdgeBuilder::ParentSlot`
- `EdgeBuilder::ParentGeneration`
- `EdgeBuilder::PreviousLocator`
- `EdgeBuilder::NextLocator`
- `EdgeBuilder::SetSiblingLocators`
- `EdgeBuilder::MakeParentRelation`
- `EdgeBuilder::IsEmpty`
- `EdgeBuilder::IsPartiallyEmpty`
- `EdgeBuilder::Clear`
- `EdgeBuilder::CanInsertCombinedDAGRelation`
- `EdgeBuilder::NextSequence`
- `EdgeBuilder::PackEdgeHeader`
- `EdgeBuilder::UnpackEdgeHeader`
- `EdgeBuilder::DirtyBit`
- `EdgeBuilder::ControlOffset`
- `EdgeBuilder::RawEdgeTableRecordWidth`
- `EdgeBuilder::IsParentEmpty`
- `EdgeBuilder::AreSiblingsEmpty`
- `HandleOfAPCStatic::MakeControlCell`
- `HandleOfAPCStatic::ReadControlCell`
- `HandleOfAPCStatic::IsGenerationValid`
- `HandleOfAPCStatic::NextGeneration`
- `HandleOfAPCStatic::IsOpenGeneration`
- `HandleOfAPCStatic::CellOffset`
- `APCRelocationDef::ValidateFabricCache`

### Fabric constructor / relocation

- `FabricConstructor::ReadAFabricU64Directly`
- `FabricConstructor::AtomicallyLoadReadAUnit`
- `FabricConstructor::DirectlyStoreFabricUnit64`
- `FabricConstructor::AtomicallyStoreU64Fab`
- `FabricConstructor::CompareExchangeStrongFromFabric`
- `FabricConstructor::CompareExchangeWeakInSlab`
- `FabricConstructor::ForceNxLenMemCopy`
- `MatrixViewConstructor::MetrixViewRow_`
- `MatrixViewConstructor::ConstructMatrixViewRecords_`
- `MatrixViewConstructor::PrepareMatrixViewRow_`
- `MatrixViewConstructor::ClearMatrixViewRow_`
- `MatrixViewConstructor::InitializeRegionProtocolStorage_`
- `APCHandleAndRetirement::GetAPCGenerationPtr_`
- `APCHandleAndRetirement::InitializeAPCGenerationTable_`
- `APCHandleAndRetirement::OpenAPCGeneration_`
- `APCHandleAndRetirement::CloseAPCGeneration_`
- `APCHandleAndRetirement::AdvanceClosedAPCGeneration_`
- `APCHandleAndRetirement::ReadFirstFreeAPCIdx_`
- `APCHandleAndRetirement::UpdateFirstFreeIdx_`
- `RecordBookConstructor::GetRecordMapCarrierRanges_`
- `RecordBookConstructor::IdleAFabricTableClassRangesMemory_`
- `RecordBookConstructor::WriteARecordBookOfTSCEntry_`
- `RecordBookConstructor::CheckRecordBookRange_`
- `APCLifeCycle::GetDescriptionLockIdxInFabric_`
- `APCLifeCycle::GetSegmentPoolRange`
- `APCLifeCycle::SwitchDescriptionState`
- `APCLifeCycle::ReadAPCStateAtomically_`
- `APCLifeCycle::InitAllAPCLifeCycleState`
- `EdgeTableConstructor::ReadAnEdgeTableRange_`
- `EdgeTableConstructor::EdgeControlCellIndex_`
- `EdgeTableConstructor::ParentRelations_`
- `EdgeTableConstructor::ConstructParentRelationObjects_`
- `EdgeTableConstructor::InitializeEdgeTable_`
- `EdgeTableConstructor::ReadEdgeControl_`
- `EdgeTableConstructor::ReadEdgeHeader_`
- `EdgeTableConstructor::ReadParentHandle_`
- `EdgeTableConstructor::ReserveEdgeDomain_`
- `EdgeTableConstructor::ReserveEdgeRow_`
- `EdgeTableConstructor::StoreReservedParentHandle_`
- `EdgeTableConstructor::StoreReservedSiblingLocators_`
- `EdgeTableConstructor::PublishReservedEdgeDomain_`
- `EdgeTableConstructor::PublishReservedEdgeRow_`
- `CompiledDAGTableConstructor::CompiledDAGRow_`
- `CompiledDAGTableConstructor::InitializeCompiledDAGTAble_`
- `CompiledDAGTableConstructor::CompiledDAGRelation_`
- `CompiledDAGTableConstructor::ReadCompiledDAGParentMask_`
- `SlabToFabricConverterAndCordinator::InitializeFabric`
- `SlabToFabricConverterAndCordinator::ShutDownFabric`
- `SlabToFabricConverterAndCordinator::IsFabricActive`
- `SlabToFabricConverterAndCordinator::SaveFabric`
- `SlabToFabricConverterAndCordinator::AttachFabric`
- `SlabToFabricConverterAndCordinator::DetachFabric`

### DAG / APCFinilizer

- `DAGMutationConf::AddRowParticipant_`
- `DAGMutationConf::FindRowParticipant_`
- `DAGMutationConf::FindOrInsertRelationDelta_`
- `DAGMutationConf::EditReservedParentHandle_`
- `DAGMutationConf::EditReservedSiblingLocators_`
- `DAGMutationConf::ReserveAllRows_`
- `DAGMutationConf::AbortRowTransaction_`
- `DAGMutationConf::CommitRowTransaction_`
- `DAGMutationConf::ValidateConditionalParentPublication_`
- `DAGMutationConf::PrepareConditionalParentPublication_`
- `ConstructDAGOnEachAxis::ScanReservedParentRow_`
- `ConstructDAGOnEachAxis::AddParentRelation_`
- `ConstructDAGOnEachAxis::RemoveParentRelation_`
- `ConstructDAGOnEachAxis::ReplaceParentRelation_`
- `APCFinilizer::CreateAPC`
- `APCFinilizer::GetExistingAPC_`
- `APCFinilizer::BindExistingAPCSnapshot_`
- `APCFinilizer::GetASlotForNewAPCLink`
- `APCFinilizer::ResolveChildLocator_`
- `APCFinilizer::FindParent_`
- `APCFinilizer::FindFirstChild_`
- `APCFinilizer::FindLastChild_`
- `APCFinilizer::FindNextChild_`
- `APCFinilizer::FindPreviousChild_`
- `APCFinilizer::RetireAPC_`
- `APCFinilizer::ReclaimRetiredSlotTemp_`
- `APCFinilizer::IsNodePolicyReConfigurable_`

### GHGF

- `GHGFLayerModel::CouplingIndex`
- `GHGFLayerModel::MakeDefaultGHGFStorageProfile`
- `GHGFLayerModel::IsValidStoregeProfile`
- `GHGFLayerModel::IsHCouplingParameter`
- `GHGFModel::ConnectGHGFParent`
- `GHGFModel::RemoveParent`
- `GHGFModel::InitializeGHGFFabric`
- `GHGFModel::ResetGHGFState`
- `GHGFModel::CreateNodeOfGHGF`
- `GHGFModelConstructor::ConstructGHGFModel`
- `GHGFModelConstructor::PredictModelNONVectorized`
- `GHGFModelConstructor::UpdateModelNONVectorized`
- `GHGFModelConstructor::TrainModelNONVectorized`
- `GHGFModelConstructor::RunGHGFSequence`
- `GHGFModelConstructor::FitGHGFParameters`
- `GHGFStructralLearningModel::ReadStructureSnapshotConcurrently`
- `GHGFStructralLearningModel::TryApplyGHGFStructureMutation`
- `GHGFNode::InitializeGHGFNode`

### Type / class / struct index

Use this list for exact-symbol lookup. Detailed behavior is explained in the sections above.

- `AdaptivePackedCellContainer`
- `APCDataStructure`
- `APCDataStructure::RangeOfAPC`
- `APCDataStructure::CacheOfAPC`
- `APCUseScope`
- `TwinU32ToU64`
- `Twin28Plus8`
- `Twin28Plus8::CarrierTwin28`
- `DescriptionOfAPC`
- `DescriptionOfAPC::SeqLockAndStateStruct`
- `HeaderOrchestrator`
- `SchemaOrchestrator`
- `SchemaOrchestrator::RegionSchemaRecord`
- `SchemaOrchestrator::FabricRegionConfig`
- `SchemaValidator`
- `SchemaDefinition`
- `ResolveRegionBiteView`
- `APCStorageGeometry`
- `RegionView<T>`
- `FabricToAPCLinker`
- `FabricToAPCLinker::RelationOparation`
- `RegionViewConstructor`
- `CoreOfFabricCoordinator`
- `CoreOfFabricCoordinator::FabricCache`
- `CoreOfFabricCoordinator::DetachFabric`
- `RecordBookConf`
- `RecordBookConf::FabricSegmentBounds`
- `RawPackedCellAllocator`
- `EdgeBuilder`
- `EdgeBuilder::ParentRelation`
- `EdgeBuilder::EdgeData`
- `HandleOfAPCStatic`
- `HandleOfAPCStatic::ControlValues`
- `APCRelocationDef`
- `FabricConstructor`
- `MatrixViewConstructor`
- `APCHandleAndRetirement`
- `RecordBookConstructor`
- `APCLifeCycle`
- `EdgeTableConstructor`
- `CompiledDAGTableConstructor`
- `CompiledDAGTableConstructor::CompiledDAGRecord`
- `SlabToFabricConverterAndCordinator`
- `DAGMutationConf`
- `DAGMutationConf::ConditionalParentPublication`
- `DAGMutationConf::DAGRowParticipant`
- `DAGMutationConf::DAGRelationDelta`
- `DAGMutationConf::DAGMutationTransaction`
- `ConstructDAGOnEachAxis`
- `ConstructDAGOnEachAxis::ParentRowScan`
- `APCFinilizer`
- `GHGFLearningConfig`
- `StructuralLearningGHGF`
- `StructuralLearningGHGF::GHGFStructureMutation`
- `StructuralLearningGHGF::GHGFStructureSnapshot`
- `StructuralLearningGHGF::GHGFStructureMutationResult`
- `StructuralLearningGHGF::GHGFParentExecutionSnapshot`
- `GHGFLayerModel`
- `GHGFLayerModel::GHGFStorageProfile`
- `GHGFLayerModel::GHGFConnection`
- `GHGFLayerModel::GHGFParameterRange`
- `GHGFLayerModel::GHGFCache`
- `GHGFLayerModel::StorageConst`
- `GHGFNode`
- `GHGFModel`
- `GHGFModel::GHGFModelConstructionValues`
- `GHGFStructralLearningModel`
- `GHGFStructralLearningModel::CouplingPublicationContext`
- `GHGFModelConstructor`

### Complete out-of-line member-function index

The following list was cross-checked against every class-qualified member definition in the APC/Fabric/GHGF production portion of the supplied aggregate source (before `SuperNova.cpp` and the test kits). Inline/static helpers are indexed in the subsystem lists above.

- `AdaptivePackedCellContainer::IsOpenGeneration_`
- `AdaptivePackedCellContainer::AddParent`
- `AdaptivePackedCellContainer::RemoveParent`
- `AdaptivePackedCellContainer::ReplaceParent`
- `AdaptivePackedCellContainer::AttachMyChild`
- `AdaptivePackedCellContainer::DetachMyChild`
- `AdaptivePackedCellContainer::FindParent`
- `AdaptivePackedCellContainer::FindFirstChild`
- `AdaptivePackedCellContainer::FindLastChild`
- `AdaptivePackedCellContainer::FindNextChild`
- `AdaptivePackedCellContainer::FindPreviousChild`
- `AdaptivePackedCellContainer::Retire`
- `FabricToAPCLinker::BindExternalRawFabricBacking_`
- `FabricToAPCLinker::ReleseFabricBindingOnly_`
- `FabricToAPCLinker::InitiateAPCMetaHeader`
- `FabricToAPCLinker::ReadAPCMetaUnit`
- `FabricToAPCLinker::IsFabricBound_`
- `FabricToAPCLinker::AcquireAPCUse_`
- `FabricToAPCLinker::IsActiveAPC`
- `RegionViewConstructor::ResolveRegionView_`
- `DAGMutationConf::ValidateConditionalParentPublication_`
- `DAGMutationConf::PrepareConditionalParentPublication_`
- `CompiledDAGTableConstructor::CompiledDAGRow_`
- `CompiledDAGTableConstructor::InitializeCompiledDAGTAble_`
- `CompiledDAGTableConstructor::CompiledDAGRelation_`
- `CompiledDAGTableConstructor::ReadCompiledDAGParentMask_`
- `DAGMutationConf::AddRowParticipant_`
- `DAGMutationConf::FindRowParticipant_`
- `DAGMutationConf::ReserveAllRows_`
- `DAGMutationConf::FindOrInsertRelationDelta_`
- `DAGMutationConf::EditReservedParentHandle_`
- `DAGMutationConf::EditReservedSiblingLocators_`
- `DAGMutationConf::CommitRowTransaction_`
- `DAGMutationConf::AbortRowTransaction_`
- `ConstructDAGOnEachAxis::ScanReservedParentRow_`
- `ConstructDAGOnEachAxis::AddParentRelation_`
- `ConstructDAGOnEachAxis::RemoveParentRelation_`
- `ConstructDAGOnEachAxis::ReplaceParentRelation_`
- `GHGFModel::IsGHGFModelReady_`
- `GHGFModel::GHGFRegion_`
- `GHGFModel::GHGFStateRow_`
- `GHGFModel::GHGFErrorRow_`
- `GHGFModel::GHGFWeight_`
- `GHGFModel::FFRowGHGF_`
- `GHGFModel::FBRowGHGF_`
- `GHGFModel::InvalidateGHGFModel_`
- `GHGFModel::GHGFParentMask_`
- `GHGFModel::GetGHGFNode_`
- `GHGFModel::ConnectGHGFParent`
- `GHGFModel::RemoveParent`
- `GHGFModel::GetGHGFParameter_`
- `GHGFModel::SetGHGFParameter_`
- `GHGFModel::InitializeGHGFFabric`
- `GHGFModel::ResetGHGFState`
- `GHGFModel::CreateNodeOfGHGF`
- `GHGFModel::ReadGHGFNodeIdentity_`
- `GHGFModel::ReadGHGFParentExecutionSnapshot_`
- `GHGFModelConstructor::SealGHGFModel_`
- `GHGFModelConstructor::UpdateGHGFBatchNONVectorized_`
- `GHGFModelConstructor::CopyGHGFPredictionNONVectorized_`
- `GHGFModelConstructor::PredictModelNONVectorized`
- `GHGFModelConstructor::UpdateModelNONVectorized`
- `GHGFModelConstructor::PredictGHGFBatchNONVectorized_`
- `GHGFModelConstructor::RunGHGFSequence`
- `GHGFModelConstructor::FitGHGFParameters`
- `GHGFModelConstructor::LearnGHGFBatchNONVectorized_`
- `GHGFModelConstructor::TrainModelNONVectorized`
- `GHGFModelConstructor::ConstructGHGFModel`
- `GHGFNode::PublishFBackwardMessageGHGF_`
- `GHGFNode::PublishFForwardMessageGHGF_`
- `GHGFNode::InitializeGHGFNode`
- `GHGFNode::ResetAPCGHGFStateRegion_`
- `GHGFNode::GHGFRole_`
- `GHGFNode::PredictGHGFNodenNONVectorized_`
- `GHGFNode::UpdateGHGFNodeNONVectorized_`
- `GHGFNode::PropogateGHGFErrorNONVectorized_`
- `GHGFStructralLearningModel::PublishCoupling_`
- `GHGFStructralLearningModel::ReadStructureSnapshotConcurrently`
- `GHGFStructralLearningModel::TryApplyGHGFStructureMutation`
- `APCLifeCycle::GetSegmentPoolRange`
- `APCLifeCycle::ReadAPCStateAtomically_`
- `APCLifeCycle::SwitchDescriptionState`
- `APCLifeCycle::GetDescriptionLockIdxInFabric_`
- `APCLifeCycle::InitAllAPCLifeCycleState`
- `EdgeTableConstructor::EdgeControlCellIndex_`
- `EdgeTableConstructor::ReadAnEdgeTableRange_`
- `EdgeTableConstructor::ParentRelations_`
- `EdgeTableConstructor::ConstructParentRelationObjects_`
- `EdgeTableConstructor::InitializeEdgeTable_`
- `EdgeTableConstructor::ReadEdgeControl_`
- `EdgeTableConstructor::ReadEdgeHeader_`
- `EdgeTableConstructor::ReadParentHandle_`
- `EdgeTableConstructor::ReserveEdgeDomain_`
- `EdgeTableConstructor::ReserveEdgeRow_`
- `EdgeTableConstructor::StoreReservedParentHandle_`
- `EdgeTableConstructor::PublishReservedEdgeDomain_`
- `EdgeTableConstructor::PublishReservedEdgeRow_`
- `EdgeTableConstructor::StoreReservedSiblingLocators_`
- `FabricConstructor::ReadAFabricU64Directly`
- `FabricConstructor::AtomicallyLoadReadAUnit`
- `FabricConstructor::DirectlyStoreFabricUnit64`
- `FabricConstructor::AtomicallyStoreU64Fab`
- `FabricConstructor::CompareExchangeStrongFromFabric`
- `FabricConstructor::CompareExchangeWeakInSlab`
- `FabricConstructor::ForceNxLenMemCopy`
- `APCHandleAndRetirement::GetAPCGenerationPtr_`
- `APCHandleAndRetirement::InitializeAPCGenerationTable_`
- `APCHandleAndRetirement::OpenAPCGeneration_`
- `APCHandleAndRetirement::AdvanceClosedAPCGeneration_`
- `APCHandleAndRetirement::CloseAPCGeneration_`
- `APCHandleAndRetirement::ReadFirstFreeAPCIdx_`
- `APCHandleAndRetirement::UpdateFirstFreeIdx_`
- `MatrixViewConstructor::MetrixViewRow_`
- `MatrixViewConstructor::ConstructMatrixViewRecords_`
- `MatrixViewConstructor::PrepareMatrixViewRow_`
- `MatrixViewConstructor::ClearMatrixViewRow_`
- `MatrixViewConstructor::InitializeRegionProtocolStorage_`
- `RecordBookConstructor::CheckRecordBookRange_`
- `RecordBookConstructor::IdleAFabricTableClassRangesMemory_`
- `RecordBookConstructor::GetRecordMapCarrierRanges_`
- `RecordBookConstructor::WriteARecordBookOfTSCEntry_`
- `SlabToFabricConverterAndCordinator::AllocatePackedCellRaw_`
- `SlabToFabricConverterAndCordinator::FreeRawPackedCells_`
- `SlabToFabricConverterAndCordinator::ResetScalarsofTheFabric_`
- `SlabToFabricConverterAndCordinator::ValidateAttachedFabricLayout_`
- `SlabToFabricConverterAndCordinator::QuiesceFabric_`
- `SlabToFabricConverterAndCordinator::ReopenLiveAPCGenerations_`
- `SlabToFabricConverterAndCordinator::InitializeFabric`
- `SlabToFabricConverterAndCordinator::ShutDownFabric`
- `SlabToFabricConverterAndCordinator::SaveFabric`
- `SlabToFabricConverterAndCordinator::AttachFabric`
- `SlabToFabricConverterAndCordinator::DetachFabric`
- `APCFinilizer::BindExistingAPCSnapshot_`
- `APCFinilizer::GetExistingAPC_`
- `APCFinilizer::ResolveChildLocator_`
- `APCFinilizer::FindParent_`
- `APCFinilizer::FindFirstChild_`
- `APCFinilizer::FindLastChild_`
- `APCFinilizer::FindNextChild_`
- `APCFinilizer::FindPreviousChild_`
- `APCFinilizer::CreateAPC`
- `APCFinilizer::GetASlotForNewAPCLink`
- `APCFinilizer::RetireAPC_`
- `APCFinilizer::ReclaimRetiredSlotTemp_`
- `APCFinilizer::IsNodePolicyReConfigurable_`

---

## 28. Glossary

| Term | Meaning in this codebase |
| --- | --- |
| APC | Adaptive Packed Cell: one fixed-capacity slot inside the Fabric segment pool plus its schema/topology/lifecycle identity. |
| Fabric | One relocatable slab containing global metadata, graph tables, handle/lifecycle controls, schemas, compiled topology and APC segment pool. |
| Facade | A C++ runtime object such as `AdaptivePackedCellContainer` that points/binds into slab state but does not own it. |
| Generation | 31-bit logical incarnation of a reusable APC slot; protects against stale handles/ABA. |
| Active use | A scoped count in handle-table control word indicating clients currently pin/use that generation. |
| Closed | Handle control bit preventing new use acquisition while quiescing/retiring. |
| APC lifecycle sequence | Sequence packed with FREE/RESERVED/LIVE/RETIRED/HAULTED header state. |
| Edge-row sequence | Separate 30-bit sequence packed with edge-domain FREE/RESERVED/LIVE state. |
| Relation ordinal | 0..K-1 position in child-owned parent relation array. |
| Relation locator | Packed child slot + relation ordinal used as sibling-list cursor. |
| Compiled DAG mask | u64 occupancy mask indicating which relation ordinals are populated for one child/axis. |
| H axis | Generic Fabric H relation table; interpreted by GHGF as value-parent coupling. |
| V axis | Generic Fabric V relation table; interpreted by GHGF as volatility-parent coupling. |
| Quiescence | Fabric/node state in which new uses are closed and existing active uses have drained. |
| Structural learning | Online selection/mutation of graph relations based on model-specific criteria; current GHGF implementation uses optimistic row sequence snapshots. |
| BATCHED_LAST_DIM | Schema flag requiring matrix width == Fabric batch capacity; GHGF uses it so each row’s batch lanes are contiguous. |
