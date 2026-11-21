#include <SuperNova>

namespace BidirectionalInMemGraph 
{
    bool ModelToFabricLinker::ValidateLayer_(const GMI::ModelIR& ir, std::vector<uint8_t>& mapped) noexcept
    {
        try 
        {
            mapped.resize(FabCache_->CountOfAPC_, UNSIGNED_ZERO);
        } catch (...) 
        {
            return false;
        }

        for (const GMI::IndividualModelDescription& dsc : ir.Description) 
        {
            for (uint32_t i = 0; i < dsc.LayerCount; i++) 
            {
                const GMI::LayerDescription& layer = ir.Layers[dsc.FirstLayer + i];
                for (uint32_t p = 0; p < layer.APCCount; p++) 
                {
                    const GMI::ModelAPC& model = *ir.APCAt(dsc.FirstLayer + i, p);
                    mapped[model.Slot] = 1;
                    const uint64_t* cell = GetAPCGenerationPtr_(model.Slot);
                    if (!cell) return false;
                    const HAS::ControlValues control = HAS::ReadControlCell(*cell);
                    HAS::LifeCycleControl state{};
                    if (
                        ReadAPCStateAtomically_(model.Slot, state) != SeqLockedOperation::FOUND ||
                        (
                            model.Generation != UNSIGNED_ZERO &&
                            !HAS::IsGenerationValid(model.Generation)
                        ) ||
                        (
                            state.State == StateOfAPC::LIVE &&
                            model.Generation != control.Generation
                        )
                    ) 
                    {
                        return false;
                    }

                    for (const SD::RegionSchemaRecord& schema : MatrixViewRow_(model.Slot)) 
                    {
                        if (
                            !SD::HasSchemaFlag(schema.Flags, SD::SchemaFlags::REGION_DISABLED) &&
                            schema.ModelInstance != dsc.Instance
                        ) 
                        {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

}