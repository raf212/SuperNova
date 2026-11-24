#pragma once

#include <pybind11/pybind11.h>

namespace SuperNovaBindings
{
    namespace py = pybind11;

    void BindAPC(py::module_& module);
    void BindFabric(py::module_& module);
    void BindModelIR(py::module_& module);
    void BindGHGF(py::module_& module);
    void BindComputer(py::module_& module);
}
