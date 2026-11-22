#include "Bindings.hpp"
#include <SuperNova>

#include <pybind11/stl.h>

namespace SuperNovaBindings
{
    void BindModelIR(py::module_& module)
    {
        namespace sn = BidirectionalInMemGraph;
        using Family = sn::GMI::ModelRepresentation;
        using Dependency = sn::GMI::ModelDependency;
        using Assignment = sn::GMI::ModelAPC;
        using Layer = sn::GMI::LayerBuildDescription;
        using Model = sn::GMI::BuildModelDescriptionIndivisually;
        using LSK = sn::GMI::LearningSystemKind;
        using sn::GMI::Dense::TensorRole;
        using TensorBinding = sn::GMI::Dense::TensorBindings;
        using sn::GMI::Dense::TensorDescription;
        using sn::GMI::Dense::ModelSnapshot;
        using sn::GMI::Dense::Operator;
        using sn::GMI::Dense::OperatorGraph;
        using sn::GMI::Dense::OperatorKind;

        py::enum_<Family>(module, "Family")
            .value("torch_model", Family::DENSE_TENSOR_STATE)
            .value("ghgf_graph", Family::GHGF_GRAPH);

        py::enum_<LSK>(module, "LearningSystem")
            .value("backprop", LSK::BACKPROPAGATION)
            .value("ghgf_local", LSK::GHGF_LOCAL);

        py::class_<Dependency>(module, "Dependency")
            .def(py::init<uint16_t, uint16_t, sn::MacroColumnOfAPC, sn::MacroColumnOfAPC, uint32_t>(),
                 py::arg("producer"), py::arg("consumer"), py::arg("producer_region"),
                 py::arg("consumer_region"), py::arg("flags") = 0u);

        py::class_<Assignment>(module, "Assignment")
            .def(py::init([](uint32_t slot, uint32_t role) {
                Assignment assignment{};
                assignment.Slot = slot;
                assignment.Role = role;
                return assignment;
            }), py::arg("slot"), py::arg("role") = 0u)
            
            .def_readonly("slot", &Assignment::Slot)
            .def_readonly("generation", &Assignment::Generation)
            .def_readonly("role", &Assignment::Role);

        py::class_<Layer>(module, "LayerDescription")
            .def(py::init<uint32_t, uint32_t, std::vector<Assignment>>(),
                 py::arg("operator_id"), py::arg("capacity"), py::arg("assignments"))
            .def_readwrite("operator_id", &Layer::OperatorId)
            .def_readwrite("capacity", &Layer::MaxAllowedAPCs)
            .def_readwrite("assignments", &Layer::APCs);

        py::class_<Model>(module, "ModelDescription")
            .def(py::init([](uint32_t instance, Family family, std::vector<Layer> layers,
                             std::vector<Dependency> dependencies, const py::bytes& configuration,
                             bool structural_adaptation) {
                const std::string bytes = configuration;
                const auto config = std::as_bytes(std::span{bytes.data(), bytes.size()});
                Model model{};
                model.Instance = instance;
                model.Model = family;
                model.StructuralADaptationAllowed = structural_adaptation;
                model.Configuration.assign(config.begin(), config.end());
                model.Dependencies = std::move(dependencies);
                model.Layers = std::move(layers);
                return model;
            }), py::arg("instance"), py::arg("family"), py::arg("layers"),
                py::arg("dependencies") = std::vector<Dependency>{},
                py::arg("configuration") = py::bytes(""),
                py::arg("structural_adaptation") = false);


        py::enum_<TensorRole>(module, "TensorRole")
            .value("PARAMETER", TensorRole::Parameter)
            .value("BUFFER", TensorRole::Buffer);

        py::class_<TensorBinding>(module, "TensorBinding")
            .def(py::init<uint32_t, sn::MacroColumnOfAPC, uint64_t, uint32_t, uint32_t>(),
                 py::arg("apc_index"), py::arg("column"), py::arg("tensor_offset"),
                 py::arg("apc_offset"), py::arg("element_count"))
            .def_readonly("apc_index", &TensorBinding::APCIndex)
            .def_readonly("column", &TensorBinding::Column)
            .def_readonly("tensor_offset", &TensorBinding::TensorOffset)
            .def_readonly("apc_offset", &TensorBinding::APCOffset)
            .def_readonly("element_count", &TensorBinding::ElementCount);

        py::class_<TensorDescription>(module, "TensorDescription")
            .def(py::init<std::string, TensorRole, sn::SchemaDefinition::DataTypeOfMacroColumn,
                          uint32_t, std::vector<uint32_t>, std::vector<TensorBinding>>(),
                 py::arg("name"), py::arg("role"), py::arg("dtype"), py::arg("layer_index"),
                 py::arg("shape"), py::arg("bindings"))
            .def_readonly("name", &TensorDescription::Name)
            .def_readonly("role", &TensorDescription::Role)
            .def_readonly("dtype", &TensorDescription::DType)
            .def_readonly("layer_index", &TensorDescription::LayerIndex)
            .def_readonly("shape", &TensorDescription::Shape)
            .def_readonly("bindings", &TensorDescription::Bindings);

        py::enum_<OperatorKind>(module, "OperatorKind")
            .value("LINEAR", OperatorKind::Linear)
            .value("RELU", OperatorKind::ReLU);

        py::class_<Operator>(module, "DenseOperator")
            .def(py::init<OperatorKind, uint32_t, uint32_t, uint32_t>(),
                 py::arg("kind"), py::arg("input"),
                 py::arg("weight") = sn::GMI::Dense::NO_TENSOR, py::arg("bias") = sn::GMI::Dense::NO_TENSOR)
            .def_readonly("kind", &Operator::Kind)
            .def_readonly("input", &Operator::Input)
            .def_readonly("weight", &Operator::Weight)
            .def_readonly("bias", &Operator::Bias);

        py::class_<OperatorGraph>(module, "DenseOperatorGraph")
            .def(py::init<std::vector<Operator>, uint32_t>(), py::arg("operators"), py::arg("output"))
            .def_readonly("operators", &OperatorGraph::Operators)
            .def_readonly("output", &OperatorGraph::Output);

        module.def("encode_dense_ir", [](const std::vector<TensorDescription>& tensors,
                                        const std::string& topology_fingerprint,
                                        const std::optional<OperatorGraph>& graph) {
            const auto bytes = sn::GMI::Dense::DenseModelDynamics{tensors, topology_fingerprint, graph}.Encode();
            return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }, py::arg("tensors"), py::arg("topology_fingerprint") = "", py::arg("graph") = py::none());

        py::class_<ModelSnapshot>(module, "DenseModelSnapshot")
            .def_readonly("structure_revision", &ModelSnapshot::StructureRevision)
            .def_property_readonly("tensors", [](const ModelSnapshot& snapshot) {
                return snapshot.DenseIR.Tensors;
            })
            .def_property_readonly("topology_fingerprint", [](const ModelSnapshot& snapshot) {
                return snapshot.DenseIR.TopologyFingerPrint;
            })
            .def_property_readonly("graph", [](const ModelSnapshot& snapshot) {
                return snapshot.DenseIR.Graph;
            })
            .def_readonly("assignments", &ModelSnapshot::Assignments);
    }
}
