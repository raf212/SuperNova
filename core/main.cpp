#include <SuperNova>
#include "TestFiles/StaticAsserts.hpp"
#include "TestFiles/GHGFTestKit.hpp"
#include "TestFiles/TestKit.hpp"
#include "TestFiles/SuperNovaTest2ARepeat.hpp"
int main()
{
    return APCDAGTests::Test08_FabricRelocation::Run() == APCDAGTests::Result::PASS ? 0 : 1;
}