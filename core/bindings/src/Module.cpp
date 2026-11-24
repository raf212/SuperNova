#include "Bindings.hpp"

PYBIND11_MODULE(SuperNovaBind, module)
{
    using namespace SuperNovaBindings;

    module.doc() = "Python bindings for the SuperNova C++ core.";
    auto apc = module.def_submodule("apc", "APC schemas, edges, and numeric regions.");
    auto fabric = module.def_submodule("fabric", "Fabric allocation and APC creation.");
    auto models = module.def_submodule("models", "SuperNova model entry points.");
    auto compute = module.def_submodule("compute", "Compute runtime and checkpoints.");

    BindAPC(apc);
    BindModelIR(models);
    BindFabric(fabric);
    auto ghgf = models.def_submodule("ghgf", "GHGF inference and learning.");
    BindGHGF(ghgf);
    BindComputer(compute);
}
