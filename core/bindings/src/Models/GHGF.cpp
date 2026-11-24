#include "Bindings.hpp"
#include <SuperNova>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace SuperNovaBindings
{
    namespace sn = BidirectionalInMemGraph;
    using Layer = sn::GHGFLayerModel;
    using Model = sn::GHGFModelConstructor;
    using Role = Layer::GHGFNodeRole;
    using Edge = sn::FabricSegments;
    using Connection = Layer::GHGFConnection;

    namespace
    {
        class GHGF final : public Model
        {
        public:
            using Model::GetGHGFParameter_;
            using Model::SetGHGFParameter_;

            uint32_t ObservationCount() const noexcept { return GHGFCache_.ObservationCount_; }
            uint32_t BatchCapacity() const noexcept { return Profile_.BatchCapacity; }
            bool StructuralLearning() const noexcept { return GHGFCache_.StructuralLearningActive_; }
        };

        using FloatArray = py::array_t<float, py::array::c_style>;

        std::span<const float> Observations(const GHGF& model, const FloatArray& input, int rank)
        {
            if (input.ndim() != rank || input.shape(rank - 2) != model.ObservationCount() ||
                input.shape(rank - 1) <= 0 || input.shape(rank - 1) > model.BatchCapacity() ||
                (rank == 3 && (input.shape(0) <= 0 || input.shape(0) > UINT32_MAX)))
                throw py::value_error("Expected (observation, batch) or (time, observation, batch) with the model's dimensions.");
            if (reinterpret_cast<uintptr_t>(input.data()) % alignof(float) != 0)
                throw py::value_error("Expected an aligned float32 array.");
            return {input.data(), static_cast<size_t>(input.size())};
        }

        std::span<float> Output(FloatArray& array)
        {
            return {array.mutable_data(), static_cast<size_t>(array.size())};
        }

        void RequireSuccess(bool success, const char* error)
        {
            if (!success) throw std::runtime_error(error);
        }
    }

    void BindGHGF(py::module_& module)
    {
        using Learning = sn::GHGFLearningConfig;
        using Range = Layer::GHGFParameterRange;
        using Operation = sn::EdgeBuilder::StructureOperation;
        using Status = Layer::GHGFConcurrentOperation;
        using Mutation = Layer::GHGFStructureMutation;
        using Snapshot = Layer::GHGFStructureSnapshot;
        using Result = Layer::GHGFStructureMutationResult;

        py::enum_<Role>(module, "Role")
            .value("OBSERVATION", Role::OBSERVATION)
            .value("VALUE", Role::VALUE)
            .value("VOLATILE", Role::VOLATILE);

        py::class_<Connection>(module, "Connection")
            .def(py::init<uint32_t, uint32_t, Edge, float>(), py::arg("parent"), py::arg("child"),
                 py::arg("edge") = Edge::VALUE_PARENT_EDGE_TABLE_H, py::arg("coupling") = 1.0f)
            .def_readonly("parent", &Connection::Parent)
            .def_readonly("child", &Connection::Child)
            .def_readonly("edge", &Connection::Edge)
            .def_readonly("coupling", &Connection::Coupling);

        py::class_<Learning>(module, "LearningConfig")
            .def(py::init<>())
            .def_readwrite("h_coupling_rate", &Learning::HCouplingLearningRate)
            .def_readwrite("drift_rate", &Learning::DriftLearningRate)
            .def_readwrite("volatility_rate", &Learning::VolatilityLearningRate)
            .def_readwrite("v_coupling_rate", &Learning::VCouplingLearningRate)
            .def_readwrite("auto_connection_rate", &Learning::AutoConnectionLearningRate)
            .def_readwrite("gradient_clip", &Learning::GradientClip)
            .def_readwrite("min_tonic_log_volatility", &Learning::MinTonicLogVolatility)
            .def_readwrite("max_tonic_log_volatility", &Learning::MaxTonicLogVolatility);

        py::class_<Range>(module, "ParameterRange")
            .def(py::init<uint32_t, uint32_t, float, float>(), py::arg("slot"), py::arg("index"),
                 py::arg("lower"), py::arg("upper"))
            .def_readonly("slot", &Range::Slot)
            .def_readonly("index", &Range::Index)
            .def_readonly("lower", &Range::Lower)
            .def_readonly("upper", &Range::Upper);

        py::enum_<Operation>(module, "StructureOperation")
            .value("ADD_PARENT", Operation::ADD_PARENT)
            .value("REMOVE_PARENT", Operation::REMOVE_PARENT)
            .value("REPLACE_PARENT", Operation::REPLACE_PARENT);

        py::enum_<Status>(module, "StructureStatus")
            .value("SUCCESS", Status::SUCCESS).value("REJECTED", Status::REJECTED)
            .value("RETRY", Status::RETRY).value("STALE", Status::STALE);

        py::class_<Mutation>(module, "StructureMutation")
            .def(py::init<>())
            .def_readwrite("operation", &Mutation::Operation)
            .def_readwrite("child", &Mutation::Child)
            .def_readwrite("old_parent", &Mutation::OldParent)
            .def_readwrite("new_parent", &Mutation::NewParent)
            .def_readwrite("edge", &Mutation::Edge)
            .def_readwrite("coupling", &Mutation::Coupling)
            .def_readwrite("expected_row_sequence", &Mutation::ExpectedRowSequence);

        py::class_<Snapshot>(module, "StructureSnapshot")
            .def_readonly("child", &Snapshot::Child)
            .def_readonly("edge", &Snapshot::Edge)
            .def_readonly("row_sequence", &Snapshot::RowSequence)
            .def_readonly("parent_mask", &Snapshot::ParentMask)
            .def_readonly("valid", &Snapshot::IsValid);

        py::class_<Result>(module, "StructureResult")
            .def_readonly("status", &Result::Result)
            .def_readonly("published_row_sequence", &Result::PublishedRowSequence)
            .def_readonly("published_ordinal", &Result::PublishedOrdinal);

        py::class_<GHGF>(module, "Model")
            .def(py::init([](const std::vector<Role>& roles, const std::vector<Connection>& connections,
                             uint32_t batch_capacity, std::optional<uint8_t> max_parents, bool structural_learning) {
                Layer::GHGFStorageProfile profile{};
                const auto parents = max_parents.value_or(static_cast<uint8_t>(
                    std::min<size_t>(roles.size(), sn::ADS::DEFAULT_DIRECTED_PARENT_PER_AXIS)));
                if (parents > roles.size() ||
                    !Layer::MakeDefaultGHGFStorageProfile(profile, batch_capacity, parents))
                    throw py::value_error("Invalid GHGF batch or parent capacity.");
                std::vector<sn::GHGFNode> nodes(roles.size());
                Model::GHGFModelConstructionValues values{nodes, roles, connections, structural_learning};
                auto model = std::make_unique<GHGF>();
                RequireSuccess(model->ConstructGHGFModel(values, profile),
                               "SuperNova rejected the GHGF roles, connections, or storage profile.");
                return model;
            }), py::arg("roles"), py::arg("connections"),
                py::arg("batch_capacity") = Layer::DEFAULT_BATCH_CAPACITY,
                py::arg("max_parents") = py::none(),
                py::arg("structural_learning") = false)
            .def_property_readonly("observation_count", &GHGF::ObservationCount)
            .def_property_readonly("batch_capacity", &GHGF::BatchCapacity)
            .def("reset_state", [](GHGF& self) { return self.ResetGHGFState(); })
            .def("get_parameter", [](GHGF& self, uint32_t slot, uint32_t index) {
                return self.GetGHGFParameter_(slot, index);
            }, py::arg("slot"), py::arg("index"))
            .def("set_parameter", [](GHGF& self, uint32_t slot, uint32_t index, float value) {
                return self.SetGHGFParameter_(slot, index, value);
            }, py::arg("slot"), py::arg("index"), py::arg("value"))
            .def("predict", [](GHGF& model, uint32_t batch) {
                if (batch == 0 || batch > model.BatchCapacity())
                    throw py::value_error("Batch is outside the model's capacity.");
                FloatArray result({static_cast<py::ssize_t>(model.ObservationCount()),
                                   static_cast<py::ssize_t>(batch)});
                RequireSuccess(model.PredictModelNONVectorized(batch, Output(result)), "GHGF prediction failed.");
                return result;
            }, py::arg("batch"))
            .def("update", [](GHGF& model, const FloatArray& input) {
                const auto values = Observations(model, input, 2);
                RequireSuccess(model.UpdateModelNONVectorized(static_cast<uint32_t>(input.shape(1)), values),
                               "GHGF update failed. Observations must be binary.");
            }, py::arg("observations").noconvert())
            .def("train", [](GHGF& model, const FloatArray& input, const Learning& config) {
                const auto values = Observations(model, input, 2);
                if (model.StructuralLearning() &&
                    (config.HCouplingLearningRate != 0 || config.VCouplingLearningRate != 0))
                    throw py::value_error("Structural learning requires zero H and V coupling learning rates.");
                RequireSuccess(model.TrainModelNONVectorized(static_cast<uint32_t>(input.shape(1)), values, config),
                               "GHGF training failed. Check the learning configuration and binary observations.");
            }, py::arg("observations").noconvert(), py::arg("config") = Learning{})
            .def("run_sequence", [](GHGF& model, const FloatArray& input, bool reset_state) {
                const auto values = Observations(model, input, 3);
                FloatArray result({input.shape(0), input.shape(1), input.shape(2)});
                const auto loss = model.RunGHGFSequence(values, static_cast<uint32_t>(input.shape(0)),
                    static_cast<uint32_t>(input.shape(2)), Output(result), reset_state);
                RequireSuccess(loss.has_value(), "GHGF sequence failed. Observations must be binary.");
                return py::make_tuple(*loss, std::move(result));
            }, py::arg("observations").noconvert(), py::arg("reset_state") = true)
            .def("fit_parameters", [](GHGF& model, const FloatArray& input,
                                      const std::vector<Range>& ranges, uint32_t passes) {
                const auto values = Observations(model, input, 3);
                return model.FitGHGFParameters(values, static_cast<uint32_t>(input.shape(0)),
                    static_cast<uint32_t>(input.shape(2)), ranges, passes);
            }, py::arg("observations").noconvert(), py::arg("ranges"), py::arg("passes"))
            .def("structure_snapshot", [](GHGF& model, uint32_t child, Edge edge, uint32_t max_tries) {
                Snapshot snapshot{};
                const auto status = model.ReadStructureSnapshotConcurrently(child, edge, snapshot, max_tries);
                return py::make_tuple(status, snapshot);
            }, py::arg("child"), py::arg("edge") = Edge::VALUE_PARENT_EDGE_TABLE_H,
               py::arg("max_tries") = DEFAULT_MAX_TRIES)
            .def("apply_structure_mutation", [](GHGF& self, const Mutation& mutation, uint32_t max_tries) {
                return self.TryApplyGHGFStructureMutation(mutation, max_tries);
            }, py::arg("mutation"), py::arg("max_tries") = DEFAULT_MAX_TRIES);
    }
}
