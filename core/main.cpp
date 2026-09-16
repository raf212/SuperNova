#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "Models/GHGF/GHGFModelOfAPC.hpp"
#include "TestFiles/GHGFTestKit.hpp"
#include "TestFiles/TestKit.hpp"

int main()
{
    APCDAGTests::RunAll();
    return GHGFTestKit::RunAll();
}