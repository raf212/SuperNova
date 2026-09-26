#pragma once

#include "TestKit.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <thread>

namespace SuperNovaTest2ARepeat
{
namespace T = APCDAGTests;
namespace B = APCDAGTests::BenchmarkCore;

inline int Run(
    std::size_t repeat_count = 10u,
    std::size_t max_writers = 0u,
    const char* output_file = "SuperNovaTest2ARepeat_results.txt")
{
    const std::size_t hardware_threads =
        static_cast<std::size_t>(std::thread::hardware_concurrency());

    if (max_writers == 0u)
    {
        // Match the current default TestKit configuration:
        // usable threads = 18 -> Test 2 writer sweep = 1..16.
        max_writers = hardware_threads > 2u
            ? std::min<std::size_t>(16u, hardware_threads - 2u)
            : 1u;
    }

    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out)
        return 2;

    const auto cases =
        T::MakeBenchmarkCases(100u, 10'000u, 4u, 32u);

    std::size_t failed_repetitions = 0u;
    std::size_t failed_cases = 0u;

    out
        << "SUPERNOVA FABRIC TEST 2A REPEAT TEST\n"
        << "repeats=" << repeat_count << '\n'
        << "writer_sweep=1.." << max_writers << '\n'
        << "measured_runs_per_point="
        << T::ConcurrencyConfig::MEASURED_RUNS << "\n\n";
    out.flush();

    for (std::size_t repeat = 1u; repeat <= repeat_count; ++repeat)
    {
        bool repeat_failed = false;

        for (std::size_t case_index = 0u;
             case_index < cases.size();
             ++case_index)
        {
            const T::BenchmarkCase& config = cases[case_index];

            const auto maybe_scenario =
                B::MakeMutationScenario(
                    config,
                    B::MutationLocality::HOTSPOT,
                    max_writers);

            if (!maybe_scenario.has_value())
            {
                ++failed_cases;
                repeat_failed = true;

                out
                    << "FAIL"
                    << " repeat=" << repeat
                    << " case=" << (case_index + 1u)
                    << " N=" << config.NodeCount
                    << " K=" << static_cast<unsigned>(config.ParentCapacity)
                    << " reason=SCENARIO_BUILD_FAILED\n";
                out.flush();
                continue;
            }

            const B::MutationScenario scenario =
                maybe_scenario.value();

            const B::MutationSchedule schedule =
                B::BuildMutationSchedule(scenario);

            for (std::size_t writers = 1u;
                 writers <= max_writers;
                 ++writers)
            {
                for (std::size_t sample = 1u;
                     sample <= T::ConcurrencyConfig::MEASURED_RUNS;
                     ++sample)
                {
                    T::RuntimeAPCFabricBackend backend{};

                    const bool build_ok =
                        B::BuildMutationBackend(
                            backend,
                            scenario,
                            writers);

                    B::MutationSweepResult result{};
                    bool verify_ok = false;
                    bool verify_ran = false;

                    if (build_ok)
                    {
                        result =
                            B::RunMutationWorkers(
                                backend,
                                scenario,
                                schedule,
                                writers);

                        if (result.Ok)
                        {
                            verify_ran = true;
                            verify_ok =
                                B::VerifyMutationScenario(
                                    backend,
                                    scenario,
                                    schedule,
                                    writers);
                        }
                    }

                    if (build_ok && result.Ok && verify_ok)
                        continue;

                    ++failed_cases;
                    repeat_failed = true;

                    const double retry_per_success =
                        result.Success == 0u
                            ? 0.0
                            : static_cast<double>(result.Retries) /
                              static_cast<double>(result.Success);

                    out
                        << "FAIL"
                        << " repeat=" << repeat
                        << " case=" << (case_index + 1u)
                        << " N=" << config.NodeCount
                        << " K=" << static_cast<unsigned>(config.ParentCapacity)
                        << " writers=" << writers
                        << " sample=" << sample
                        << " build=" << (build_ok ? "PASS" : "FAIL")
                        << " mutation=" << (result.Ok ? "PASS" : "FAIL")
                        << " verify="
                        << (verify_ran
                            ? (verify_ok ? "PASS" : "FAIL")
                            : "SKIPPED")
                        << " success=" << result.Success
                        << " retries=" << result.Retries
                        << " retry/success="
                        << std::fixed << std::setprecision(4)
                        << retry_per_success
                        << '\n';

                    // Keep every discovered failure safe during a long run.
                    out.flush();
                }
            }
        }

        if (repeat_failed)
            ++failed_repetitions;
    }

    out
        << "\n============================================================\n"
        << "FINAL SUMMARY\n"
        << "repeats=" << repeat_count << '\n'
        << "failed_repetitions=" << failed_repetitions << '\n'
        << "passed_repetitions="
        << (repeat_count - failed_repetitions) << '\n'
        << "failed_cases=" << failed_cases << '\n'
        << "result=" << (failed_cases == 0u ? "PASS" : "FAIL") << '\n'
        << "============================================================\n";

    out.flush();
    return failed_cases == 0u ? 0 : 1;
}

} // namespace SuperNovaTest2ARepeat
