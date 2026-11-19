#include "Bindings.hpp"
#include <SuperNova>

#include <pybind11/stl/filesystem.h>

namespace SuperNovaBindings
{
    namespace
    {
        namespace sn = BidirectionalInMemGraph;
        namespace cd = sn::ComputerDetail;

        struct Computer
        {
            // Destroy the native computer before releasing the bound fabric.
            std::shared_ptr<sn::APCFinilizer> Fabric;
            sn::Computer Native;
        };
    }

    void BindComputer(py::module_& module)
    {
        using Backend = cd::BackendC;
        using Config = cd::ConfigC;
        using Info = cd::RuntimeInfoC;
        using Error = cd::ErrorC;
        using Stats = cd::TransferStatics;

        py::enum_<Backend>(module, "Backend")
            .value("AUTOMATIC", Backend::AUTOMATIC)
            .value("CPU", Backend::CPU)
            .value("GPU", Backend::GPU);

        py::enum_<Error>(module, "Error")
            .value("NONE", Error::NONE)
            .value("INVALID_CONFIG", Error::INVALID_CONF)
            .value("CONFIGURED_AFTER_FABRIC_INIT", Error::CONFIGIRED_AFTER_FABRIC_INIT)
            .value("ACCELERATOR_UNAVAILABLE", Error::ACCELERATOR_UNAVAILAVLE)
            .value("SHARED_USM_UNAVAILABLE", Error::SHARED_USM_UNAVAILAVLE)
            .value("CONCURRENT_SHARED_USM_UNAVAILABLE", Error::CONCURRENT_SHARED_USM_UNAVAILAVLE)
            .value("INVALID_REGION", Error::INVALID_REGION)
            .value("INVALID_RANGE", Error::INVALID_RANGE)
            .value("INVALID_DTYPE", Error::INVALIDE_DTYPE)
            .value("DEVICE_EXECUTION_FAILED", Error::DEVICE_EXECUTION_FAILED)
            .value("ATOMIC_CAPABILITY_UNAVAILABLE", Error::ATOMIC_CAPABILITY_UNAVAILAVLE)
            .value("PORTABLE_DETACH_FAILED", Error::PORTABLE_DETACH_FAILED);

        py::class_<Config>(module, "Config")
            .def(py::init<>())
            .def_readwrite("default_backend", &Config::DefaultBackend)
            .def_readwrite("gpu_work_threshold", &Config::GPUWorkThreshold)
            .def_readwrite("allow_cpu_fallback", &Config::AllowCPUFallback);

        py::class_<Info>(module, "RuntimeInfo")
            .def_readonly("default_backend", &Info::DefaultBackend)
            .def_readonly("adaptive_cpp_compiled", &Info::AdaptiveCppCompiled)
            .def_readonly("accelerator_available", &Info::AcceleratorAvailable)
            .def_readonly("device_usm_available", &Info::DeviceUSMAvailable)
            .def_readonly("dual_queue_available", &Info::DualQueueAvailable)
            .def_readonly("configured", &Info::Configured)
            .def_readonly("device_name", &Info::DeviceName);

        py::class_<Stats>(module, "TransferStats")
            .def_readonly("allocations", &Stats::Allocation)
            .def_readonly("uploaded_bytes", &Stats::UploadedBytes)
            .def_readonly("downloaded_bytes", &Stats::DownloadedfBytes)
            .def_readonly("cache_hits", &Stats::CacheHits);

        py::class_<Computer>(module, "Computer")
            .def(py::init<>())
            .def("configure", [](Computer& self, const Config& config) {
                return self.Native.Session().Configure(config);
            }, py::arg("config"))
            .def("bind_fabric", [](Computer& self, const std::shared_ptr<sn::APCFinilizer>& fabric) {
                if (!self.Native.Session().BindFabric(*fabric)) return false;
                self.Fabric = fabric;
                return true;
            }, py::arg("fabric").none(false))
            .def("unbind_fabric", [](Computer& self) {
                if (!self.Native.Session().ReleseNumericalStorage()) return false;
                self.Fabric.reset();
                return true;
            })
            .def_property_readonly("runtime_info", [](const Computer& self) {
                return self.Native.Session().GetRuntimeInfo();
            })
            .def_property_readonly("last_error", [](const Computer& self) {
                return self.Native.Session().LastError();
            })
            .def_property_readonly("transfer_stats", [](const Computer& self) {
                return self.Native.Session().TransferStats();
            })
            .def("synchronize", [](Computer& self) {
                return self.Native.Session().Synchronize();
            })
            .def("save_checkpoint", [](Computer& self, const std::filesystem::path& path,
                                        uint64_t revision, const py::bytes& metadata) {
                if (!self.Fabric || !self.Native.Session().Synchronize()) return false;
                auto& fabric = *self.Fabric;
                std::vector<uint64_t> snapshot(fabric.HotStorageDramUsm().size() +
                                               fabric.NumericalStorage().size());
                // The current CheckpointCompute supplies a hot-only buffer to SaveFabric,
                // which requires the whole slab. Use the native writer with a full snapshot.
                if (!fabric.SaveFabric(snapshot)) return false;
                const std::string bytes = metadata;
                return sn::CheckpointFormat::WriteCheckpointToDisk(
                    path, self.Native.Session().Identity(), revision, snapshot,
                    std::as_bytes(std::span{bytes.data(), bytes.size()}));
            }, py::arg("path"), py::arg("revision"), py::arg("metadata") = py::bytes(""));
    }
}
