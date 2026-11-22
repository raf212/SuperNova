#include "Bindings.hpp"
#include "AdaptivePackedCellContainer/APCHandle.hpp"

#include <SuperNova>

#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace SuperNovaBindings
{
    namespace sn = BidirectionalInMemGraph;
    using Fabric = sn::APCFinilizer;
    using Schema = sn::SchemaDefinition;
    using ModelDescriptions = std::vector<sn::GMI::BuildModelDescriptionIndivisually>;
    using Records = std::vector<Schema::RegionSchemaRecord>;

    namespace
    {
        Schema::RegionSchemaTable SchemaTable(const Records& records)
        {
            Schema::RegionSchemaTable table{};
            Schema::MakeDisabledSchemaTable(table);
            for (const auto& schema : records)
            {
                auto& entry = table.at(static_cast<size_t>(schema.Region));
                if (!Schema::HasSchemaFlag(entry.Flags, Schema::SchemaFlags::REGION_DISABLED))
                    throw py::value_error("Each APC column can appear only once.");
                entry = schema;
            }
            return table;
        }
    }

    void BindFabric(py::module_& module)
    {
        py::class_<Fabric, std::shared_ptr<Fabric>>(module, "Fabric")
            .def(
                py::init([](uint32_t slots, const Records& schemas, uint32_t slot_cells,
                             uint32_t batch_capacity, std::optional<uint32_t> max_parents,
                             const std::optional<std::filesystem::path>& segment_pool_path,
                             const ModelDescriptions& models) {
                SchemaTable(schemas); // Validate the column set before building the fabric.
                Schema::FabricRegionConfig config{};
                config.BatchCapacity = batch_capacity;
                for (const auto& record : schemas)
                    config.ActiveRegionMask |= sn::ADS::RegionBit(record.Region);

                sn::GMI::ModelIRBuildStorage ir;
                if (!ir.Build(models, slots))
                    throw py::value_error("Invalid fabric model descriptions.");

                sn::CoreOfFabricCoordinator::FabricInitConf storage{};
                if (segment_pool_path) storage.SegmentPoolPath = *segment_pool_path;
                storage.ModelDescription = ir.View();
                const auto parents = max_parents.value_or(
                    std::min<uint32_t>(slots, sn::ADS::DEFAULT_DIRECTED_PARENT_PER_AXIS));
                auto fabric = std::make_shared<Fabric>();
                if (!fabric->InitializeFabric(slots, slot_cells, config, parents, storage))
                    throw py::value_error("SuperNova rejected the fabric configuration.");
                return fabric;
            }), py::arg("slots"), 
                py::arg("schemas"),
                py::arg("slot_cells") = sn::MINIMUM_APC_CELL_COUNT,
                py::arg("batch_capacity") = 1u,
                py::arg("max_parents") = py::none(),
                py::arg("segment_pool_path") = py::none(),
                py::arg("models") = ModelDescriptions{}
            )

            .def_property_readonly("active", [](Fabric& self) { return self.IsFabricActive(); })

            .def_property_readonly("disk_paged", [](const Fabric& self) { return self.IsDiskPaged(); })

            .def("flush_numerical_backing", [](Fabric& self) { return self.FlushNumericalBacking(); })

            .def("inspect_dense_model", [](Fabric& self, uint32_t instance)
            {
                return sn::ModelRuntime{&self}.InspectDenseModel(instance);
            }, py::arg("instance"))

            .def("dense_structure_revision", [](Fabric& self, uint32_t instance)
            {
                return sn::ModelRuntime{&self}.DenseStructureRevision(instance);
            }, py::arg("instance"))

            .def("open_apc", [](const std::shared_ptr<Fabric>& self, uint32_t slot, uint32_t generation)
            {
                std::unique_ptr apc = std::make_unique<APCHandle>(self);
                if (!self->OpenAPC(slot, generation, *apc)) 
                {
                    throw std::runtime_error("The APC is unavailable or its generation is stale.");
                }
                return apc;
            }, py::arg("slot"), py::arg("generation"))

            .def(
                "create_apc", [](const std::shared_ptr<Fabric>& self, const Records& schemas) {
                auto apc = std::make_unique<APCHandle>(self);
                if (!self->CreateAPC(*apc, SchemaTable(schemas)))
                    throw std::runtime_error("SuperNova could not create the APC.");
                return apc;
                }, py::arg("schemas")
            );

    }
}
