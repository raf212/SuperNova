#include <SuperNova>
#include "../external/SuperNova_UseCase/DemoUseV1/GHGFTestKit.hpp"
int main()
{
    return APCDAGTests::Test08_FabricRelocation::Run() == APCDAGTests::Result::PASS ? 0 : 1;
}