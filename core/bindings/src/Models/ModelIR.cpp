#include "Bindings.hpp"
#include <SuperNova>

#include <pybind11/stl.h>

namespace SuperNovaBindings
{
    void BindModelIR(py::module_& module)
    {
        namespace sn = BidirectionalInMemGraph;
        using Family = sn::PreBuildModels;
        using Dependency = sn::GMI::ModelDependency;
        using Assignment = sn::GMI::ModelAPC;
        using Layer = sn::GMI::LayerBuildDescription;
        using Model = sn::GMI::BuildModelDescriptionIndivisually;

        py::enum_<Family>(module, "Family")
            .value("FNN", Family::FNN)
            .value("RNN", Family::RNN)
            .value("TRANSFORMER", Family::TRANSFORMER)
            .value("STRUCTURAL_ADAPTATION", Family::STRUCTRURAL_ADAPTATION)
            .value("GHGF", Family::GHGF);

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
            }), py::arg("slot"), py::arg("role") = 0u);

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
    }
}
