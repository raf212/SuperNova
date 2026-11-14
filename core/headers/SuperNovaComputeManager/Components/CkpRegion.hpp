#include "SessionC.hpp"

namespace BidirectionalInMemGraph::ComputerDetail 
{
    class CheckpointCompute final
    {
        SessionCompute& Runtime_;
    public:
        explicit CheckpointCompute(SessionCompute& runtime) noexcept :
            Runtime_(runtime)
        {}

        bool  SaveCehckpoint(
            const std::filesystem::path& path, 
            uint64_t revision,
            std::span<const std::byte> metadata = {}
        ) noexcept;

        bool LoadCheckpoint(
            APCFinilizer& owner,
            const std::filesystem::path& path,
            const CoreOfFabricCoordinator::FabricInitConf& storege = {}
        ) noexcept;
    };

    class RegionCompute 
    {
        template<class T>
        using NR = Numerics::NumericalRegionRef<T>;
        
        //Runtime prive ptr
        SessionCompute& Runtime_;
    public:
        using SD = SchemaDefinition;
        template<class T>
        using __Region__ = SNTOComputerBridge<T>;
        
        explicit RegionCompute(SessionCompute& runtime) noexcept :
            Runtime_(runtime)
        {}

        bool EvictRegion(const CMF::RegionRef& ref) noexcept;

        template<class T>
        std::optional<__Region__<T>> PinRegion(
            uint32_t slot,
            MacroColumnOfAPC column,
            uint32_t ordinal = UNSIGNED_ZERO,
            std::optional<uint32_t> generation = std::nullopt
        ) noexcept;

        template<class T>
        std::optional<NR<T>> RegisterNumericRegion(__Region__<T>&& region) noexcept;

        template<class T>
        std::optional<NR<T>> RegisterDiskRegion(
            const CMF::FabricCheckpoint& image,
            const CMF::RegionRef& ref
        ) noexcept;

        template<class T>
        bool PrefetchRegion(const CMF::RegionRef& ref) noexcept;

        template<class T>
        std::optional<CMF::RegionRef> Reference(NR<T> n_ref) noexcept;

        template<class T>
        std::optional<NR<T>> Resolve(const CMF::RegionRef& ref) noexcept;

        template<class T>
        std::optional<NR<T>> Operand(
            NR<T> n_ref, 
            uint32_t row,
            uint64_t offset = UNSIGNED_ZERO,
            uint64_t count = UNSIGNED_ZERO
        ) noexcept;

        template<class T>
        std::optional<NR<T>> Matrix(
            NR<T> n_ref,
            uint32_t first_row,
            uint32_t rows,
            uint32_t first_column,
            uint32_t columns
        ) noexcept;

    };
}