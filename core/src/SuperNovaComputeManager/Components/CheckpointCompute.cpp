#include <SuperNova>

namespace BidirectionalInMemGraph::ComputerDetail 
{
    bool  CheckpointCompute::SaveCehckpoint(
        const std::filesystem::path& path, 
        uint64_t revision,
        std::span<const std::byte> metadata
    ) noexcept
    {
        if (
            !Runtime_.Binding_.Owner ||
            Runtime_.Residency_.Active() ||
            Runtime_.Session_
        ) 
        {
            return false;
        }

        ExclusiveLease maintanence = Runtime_.NumericGate_().TryMaintenance();
        if (
            !maintanence || Runtime_.NumericGate_().IsPoisoned() ||
            !Runtime_.Close()
        ) 
        {
            return false;
        }

        try 
        {
            APCFinilizer& owner = *Runtime_.Binding_.Owner;
            if (!owner.IsFabricActive()) return false;
            std::span<uint64_t> hot = owner.HotStorageDramUsm();
            std::span<uint64_t> numerical = owner.NumericalStorage();
            if (
                hot.data() != Runtime_.Binding_.Cells.data() ||
                numerical.data() != Runtime_.NumericalBase_() ||
                numerical.size() != Runtime_.SlabCellCount_()
            ) 
            {
                return false;
            }

            std::vector<uint64_t> snapshot(hot.size());
            if (!owner.SaveFabric(snapshot)) 
            {
                return false;
            }

            return CMF::WriteCheckpointToDisk(
                path, Runtime_.Binding_.Id, revision, 
                snapshot, metadata, numerical
            );
        }
        catch (...)
        {
            return false;
        }
    }

    bool CheckpointCompute::LoadCheckpoint(
        APCFinilizer& owner,
        const std::filesystem::path& path,
        const CoreOfFabricCoordinator::FabricInitConf& storege
    ) noexcept
    {
        using CFC = CoreOfFabricCoordinator;
        if (
            !Runtime_.Residency_.Active() || Runtime_.Session_ ||
            !Runtime_.ReleseNumericalStorage()
        )
        {
            return false;
        }

        ExclusiveLease maintanance = owner.ExecutionGate().TryMaintenance();
        if (!maintanance || owner.IsFabricActive()) return false;

        bool attached = false;
        bool complete = false;

        try 
        {
            CMF::FabricCheckpoint image;
            CFC::FabricCache cache{};
            if (
                !image.Open(path, storege.MaxCheckPointCells) ||
                !image.FabSlabCache(cache)
            ) 
            {
                return false;
            }
            const bool paged = !storege.SegmentPoolPath.empty();
            const uint64_t physical_cells = paged ? cache.SegmentPoolBegin_ : cache.SlabCellCount_;

            const RawPackedCellAllocator allocator = storege.HostAlloc ?
                *storege.HostAlloc : RawPackedCellAllocator{};

            PackedCellStorage memory;
            if (
                !memory.Allocate(physical_cells, allocator) ||
                !image.ReadCells(UNSIGNED_ZERO, {memory.Data(), static_cast<size_t>(physical_cells)})
            ) 
            {
                return false;
            }

            if (!owner.AttachFabric(
                memory.Data(), physical_cells, 
                CFC::FabricBackigOwnership::OWNED, storege
            )) 
            {
                return false;
            }

            (void) memory.Relese();
            attached = true;
            if (paged && !image.ReadCells(cache.SegmentPoolBegin_, owner.NumericalStorage())) 
            {
                throw std::runtime_error("numerical checkpoint read");
            }

            std::span<uint64_t> hot = owner.HotStorageDramUsm();
            std::span<uint64_t> numerical = owner.NumericalStorage();
            complete = Runtime_.BindFabric({
                &owner, hot, &owner.ExecutionGate(),
                image.Identity(), numerical, hot.size()
            });

        }
        catch (...)
        {
            return false;
        }

        if (!complete && attached) 
        {
            owner.ShutDownFabric();
            if (!storege.SegmentPoolPath.empty()) 
            {
                std::error_code ignored;
                std::filesystem::remove(storege.SegmentPoolPath, ignored);
            }
        }

        return complete;
    }
}