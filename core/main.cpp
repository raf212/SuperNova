#include <SuperNova>
#include "../external/UseCase/GHGFTestKit.hpp"
#include "tests/StaticAsserts.hpp"
int main()
{
    return APCDAGTests::Test08_FabricRelocation::Run() == APCDAGTests::Result::PASS ? 0 : 1;
}