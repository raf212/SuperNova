#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "Models/GHGF/GHGFModelOfAPC.hpp"
#include "TestFiles/GHGFTestKit.hpp"
#include "TestFiles/TestKit.hpp"
#include "TestFiles/SuperNovaTest2ARepeat.hpp"
int main()
{
    return SuperNovaTest2ARepeat::Run(
        100000,
        16,
        "SuperNova_Test2A_LongRun.txt"
    );
}