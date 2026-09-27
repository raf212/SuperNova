#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "Models/GHGF/GHGFModelOfAPC.hpp"
#include "TestFiles/GHGFTestKit.hpp"
#include "TestFiles/TestKit.hpp"
#include "TestFiles/SuperNovaTest2ARepeat.hpp"
int main()
{
    return APCDAGTests::RunAll() + GHGFTestKit::RunAll();
}