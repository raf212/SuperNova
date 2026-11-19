#include "Bindings.hpp"
#include "AdaptivePackedCellContainer/APCHandle.hpp"
#include "AdaptivePackedCellContainer/RegionCopies.tpp"

#include <SuperNova>

namespace SuperNovaBindings
{
    void BindAPC(py::module_& module)
    {
        namespace sn = BidirectionalInMemGraph;
        using APC = sn::AdaptivePackedCellContainer;
        using Schema = sn::SchemaDefinition;
        using Column = sn::MacroColumnOfAPC;
        using Edge = sn::FabricSegments;
        using DType = Schema::DataTypeOfMacroColumn;
        using Protocol = Schema::SchemaProtocols;
        using Flags = Schema::SchemaFlags;
        using Result = APC::MutationResult;
        using Record = Schema::RegionSchemaRecord;

        py::enum_<Column>(module, "Column")
            .value("BOTTOM_UP", Column::BOTTOM_UP_SLOT)
            .value("TOP_DOWN", Column::TOP_DOWN_SLOT)
            .value("LATERAL", Column::LATERAL_MESAGE)
            .value("STATE", Column::STATE_SLOT)
            .value("ERROR", Column::ERROR_SLOT)
            .value("WEIGHTLESS_LOOKUP", Column::WEIGHTLESS_LOOKUP)
            .value("WEIGHT", Column::WEIGHT_SLOT)
            .value("BIAS", Column::BIAS_SLOT)
            .value("EXTRA", Column::EXTRA_SLOT);

        py::enum_<Edge>(module, "Edge")
            .value("VALUE", Edge::VALUE_PARENT_EDGE_TABLE_H)
            .value("VOLATILE", Edge::VOLATILE_PARENT_EDGE_TABLE_V);

        py::enum_<DType>(module, "DType")
            .value("UINT8", DType::UINT8_T).value("UINT16", DType::UINT16_T)
            .value("UINT32", DType::UINT32_T).value("UINT64", DType::UINT64_T)
            .value("INT8", DType::INT8_T).value("INT16", DType::INT16_T)
            .value("INT32", DType::INT32_T).value("INT64", DType::INT64_T)
            .value("FLOAT16", DType::FLOAT16_T).value("FLOAT32", DType::FLOAT32_T)
            .value("FLOAT64", DType::FLOAT64_T).value("CHAR", DType::CHAR);

        py::enum_<Protocol>(module, "Protocol")
            .value("PRIVATE", Protocol::PRIVATE_REGION)
            .value("IMMUTABLE", Protocol::IMMUTABLE_SNAPSHOT)
            .value("ATOMIC", Protocol::ATOMIC_WORD_ARRAY)
            .value("QUEUE", Protocol::MPMC_FIXED_RECORD_QUEUE)
            .value("DOUBLE_BUFFERED", Protocol::DOUBLE_BUFFERED);

        py::enum_<Result>(module, "MutationResult")
            .value("COMMITTED", Result::COMMITTED)
            .value("INVALID", Result::INVALID)
            .value("RETRY", Result::RETRY);

        py::class_<Record>(module, "RegionSchema")
            .def(py::init([](Column column, DType dtype, uint32_t rows, uint32_t columns,
                             Protocol protocol, uint32_t record_count, bool batched,
                             uint32_t model_instance, uint32_t dependency_begin, uint32_t dependency_count) {
                if (static_cast<size_t>(column) >= sn::ADS::CountOfMacroColumn())
                    throw py::value_error("Invalid APC column.");
                Record schema{};
                schema.Region = column;
                schema.Dtype = dtype;
                schema.Protocol = protocol;
                schema.MatrixHeight = rows;
                schema.MatrixWidth = columns;
                schema.ModelInstance = model_instance;
                schema.DependencyBegin = dependency_begin;
                schema.DependencyCount = dependency_count;
                schema.Flags = batched ? Flags::BATCHED_LAST_DIM : Flags::NONE;
                if (!Schema::SealDesiredSchema(schema, record_count))
                    throw py::value_error("Invalid region geometry or protocol record count.");
                return schema;
            }), py::arg("column"), py::arg("dtype"), py::arg("rows"), py::arg("columns"),
                py::arg("protocol") = Protocol::PRIVATE_REGION,
                py::arg("record_count") = 0u, py::arg("batched") = false,
                py::arg("model_instance") = 0u, py::arg("dependency_begin") = 0u,
                py::arg("dependency_count") = 0u)
            .def_readonly("column", &Record::Region)
            .def_readonly("dtype", &Record::Dtype)
            .def_readonly("rows", &Record::MatrixHeight)
            .def_readonly("columns", &Record::MatrixWidth)
            .def_readonly("protocol", &Record::Protocol)
            .def_readonly("cell_count", &Record::CellCount);

        py::class_<APCHandle>(module, "APC")
            .def_property_readonly("active", [](APCHandle& self) { return self.IsActiveAPC(); })
            .def_property_readonly("slot", [](APCHandle& self) { return self.GetThisSlotIdx(); },
                                   "Native slot index. Inactive handles return INVALID_SLOT.")
            .def("retire", [](APCHandle& self, uint32_t max_tries) {
                return self.Retire(max_tries);
            }, py::arg("max_tries") = DEFAULT_MAX_TRIES)
            .def("add_parent", [](APCHandle& self, APCHandle& parent, Edge edge) {
                return self.AddParent(parent, edge);
            }, py::arg("parent"), py::arg("edge") = Edge::VALUE_PARENT_EDGE_TABLE_H)
            .def("remove_parent", [](APCHandle& self, APCHandle& parent, Edge edge) {
                return self.RemoveParent(parent, edge);
            }, py::arg("parent"), py::arg("edge") = Edge::VALUE_PARENT_EDGE_TABLE_H)
            .def("replace_parent", [](APCHandle& self, APCHandle& old_parent,
                                       APCHandle& new_parent, Edge edge) {
                return self.ReplaceParent(old_parent, new_parent, edge);
            }, py::arg("old_parent"), py::arg("new_parent"),
               py::arg("edge") = Edge::VALUE_PARENT_EDGE_TABLE_H)
            .def("find_parent", [](APCHandle& self, uint32_t ordinal, Edge edge) {
                auto parent = self.FindParent(edge, ordinal);
                return parent.IsActiveAPC()
                    ? std::make_unique<APCHandle>(self.Fabric(), std::move(parent)) : nullptr;
            }, py::arg("ordinal"), py::arg("edge") = Edge::VALUE_PARENT_EDGE_TABLE_H)
            .def("read_region", [](APCHandle& self, Column column, DType dtype) {
                return VisitRegionType(dtype, [&]<class T> { return ReadRegion<T>(self, column); });
            }, py::arg("column"), py::arg("dtype"))
            .def("write_region", [](APCHandle& self, Column column, DType dtype, const py::array& values) {
                VisitRegionType(dtype, [&]<class T> {
                    WriteRegion<T>(self, column, values, self.Fabric()->ExecutionGate());
                });
            }, py::arg("column"), py::arg("dtype"), py::arg("values").noconvert());

        module.attr("INVALID_SLOT") = sn::ADS::APC_INDEX_BOUND_SENTINAL;
    }
}
