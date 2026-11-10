#include <SuperNova>
#include "../external/UseCase/GHGFTestKit.hpp"
int main()
{
    return APCDAGTests::Test08_FabricRelocation::Run() == APCDAGTests::Result::PASS ? 0 : 1;
}