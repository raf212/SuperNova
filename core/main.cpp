#include <SuperNova>
#include "../external/UseCase/GHGFTestKit.hpp"
#include "../external/UseCase/DenseMutation.hpp"
#include "tests/StaticAsserts.hpp"
int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    (void) argc;
    (void) argv;
    return GHGFTestKit::RunAll();
}