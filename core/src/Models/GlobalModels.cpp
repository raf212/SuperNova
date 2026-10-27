#include <SuperNova>
#include <span>

namespace BidirectionalInMemGraph::GMI
{ 
    constexpr bool ModelCatalog::ValidInfo(const ModelInfo& mi) const noexcept
    {
        if (
            mi.Model == PreBuildModels::NONE ||
            mi.Model > LAST_DEFINED_MODEL_IDX ||
            mi.ConfByteOffset > Dynamics.size() ||
            mi.ConfByteCount > (Dynamics.size() - mi.ConfByteOffset)
        ) 
        {
            return false;
        }
        return true;
    }

    constexpr bool ModelCatalog::ValidateDependency(const ModelDependency& md) const noexcept
    {
        if (
            !ADS::ValidateSlotIdx(md.Producer) ||
            !ADS::ValidateSlotIdx(md.Consumer) ||
            md.Producer == Models.size() ||
            md.Consumer == Models.size() ||
            static_cast<uint8_t>(md.ProducerRegion) >= ADS::CountOfMacroColumn() ||
            static_cast<uint8_t>(md.ConsumerRegion) >= ADS::CountOfMacroColumn()
        ) 
        {
            return false;
        }
        return true;
    }

    constexpr bool ModelCatalog::Validate() const noexcept
    {
        if (Models.size() > ADS::APC_INDEX_BOUND_SENTINAL) 
        {
            return false;
        }

        for (size_t i = 0; i < Models.size(); i++) 
        {
            const ModelInfo& mi = Models[i];
            if (mi.Instance != i + 1 || !ValidInfo(mi)) 
            {
                return false;
            }
        }

        for (const ModelDependency& md : Dependencies) 
        {
            if (!ValidateDependency(md)) return false;
        }
        return true;
    }

    constexpr bool ModelCatalog::Accept(const SchemaDefinition::RegionSchemaRecord& region) noexcept
    {
        if (
            region.ModelInstance > Models.size() ||
            region.DependencyBegin > Dependencies.size() ||
            region.DependencyCount > Dependencies.size() - region.DependencyBegin ||
            (!region.ModelInstance  && region.DependencyCount > UNSIGNED_ZERO)
        ) 
        {
            return false;
        }

        for (uint32_t i = 0; i < region.DependencyCount; i++) 
        {
            const ModelDependency& dp = Dependencies[region.DependencyBegin + i];
            if (
                dp.Consumer != region.ModelInstance ||
                dp.ConsumerRegion != region.Region
            ) 
            {
                return false;
            }
        }
        return true;
    }

}