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
        static_cast<std::size_t>(
            std::thread::hardware_concurrency());

    if (max_writers == 0u)
    {
        max_writers = hardware_threads > 2u
            ? std::min<std::size_t>(
                16u,
                hardware_threads - 2u)
            : 1u;
    }

    std::ofstream out(
        output_file,
        std::ios::out | std::ios::trunc);

    if (!out)
        return 2;

    const auto cases =
        T::MakeBenchmarkCases(
            100u,
            10'000u,
            4u,
            32u);

    std::size_t invalid_repetitions = 0u;
    std::size_t retry_repetitions = 0u;

    std::uint64_t invalid_cases = 0u;
    std::uint64_t retry_cases = 0u;
    std::uint64_t committed_cases = 0u;

    out
        << "SUPERNOVA FABRIC TEST 2A MUTATION-RESULT REPEAT TEST\n"
        << "repeats=" << repeat_count << '\n'
        << "writer_sweep=1.." << max_writers << '\n'
        << "measured_runs_per_point="
        << T::ConcurrencyConfig::MEASURED_RUNS << '\n'
        << "INVALID=FAIL\n"
        << "RETRY=COUNTED_NOT_FAILURE\n\n";
    out.flush();

    for (
        std::size_t repeat = 1u;
        repeat <= repeat_count;
        ++repeat)
    {
        bool repeat_invalid = false;
        bool repeat_retry = false;

        for (
            std::size_t case_index = 0u;
            case_index < cases.size();
            ++case_index)
        {
            const T::BenchmarkCase& config =
                cases[case_index];

            const auto maybe_scenario =
                B::MakeMutationScenario(
                    config,
                    B::MutationLocality::HOTSPOT,
                    max_writers);

            if (!maybe_scenario.has_value())
            {
                ++invalid_cases;
                repeat_invalid = true;

                out
                    << "INVALID"
                    << " repeat=" << repeat
                    << " case=" << (case_index + 1u)
                    << " N=" << config.NodeCount
                    << " K="
                    << static_cast<unsigned>(
                        config.ParentCapacity)
                    << " reason=SCENARIO_BUILD_FAILED\n";
                out.flush();
                continue;
            }

            const B::MutationScenario scenario =
                maybe_scenario.value();

            const B::MutationSchedule schedule =
                B::BuildMutationSchedule(scenario);

            for (
                std::size_t writers = 1u;
                writers <= max_writers;
                ++writers)
            {
                for (
                    std::size_t sample = 1u;
                    sample <=
                        T::ConcurrencyConfig::MEASURED_RUNS;
                    ++sample)
                {
                    T::RuntimeAPCFabricBackend backend{};

                    if (
                        !B::BuildMutationBackend(
                            backend,
                            scenario,
                            writers))
                    {
                        ++invalid_cases;
                        repeat_invalid = true;

                        out
                            << "INVALID"
                            << " repeat=" << repeat
                            << " case=" << (case_index + 1u)
                            << " N=" << config.NodeCount
                            << " K="
                            << static_cast<unsigned>(
                                config.ParentCapacity)
                            << " writers=" << writers
                            << " sample=" << sample
                            << " reason=BACKEND_BUILD_FAILED\n";
                        out.flush();
                        continue;
                    }

                    const B::MutationSweepResult result =
                        B::RunMutationWorkers(
                            backend,
                            scenario,
                            schedule,
                            writers);

                    const double retry_per_success =
                        result.Success == 0u
                            ? 0.0
                            : static_cast<double>(
                                result.Retries) /
                              static_cast<double>(
                                result.Success);

                    if (
                        result.Status ==
                        T::MutationResult::INVALID)
                    {
                        ++invalid_cases;
                        repeat_invalid = true;

                        out
                            << "INVALID"
                            << " repeat=" << repeat
                            << " case=" << (case_index + 1u)
                            << " N=" << config.NodeCount
                            << " K="
                            << static_cast<unsigned>(
                                config.ParentCapacity)
                            << " writers=" << writers
                            << " sample=" << sample
                            << " success=" << result.Success
                            << " retries=" << result.Retries
                            << " invalid="
                            << result.InvalidFailures
                            << " retry-exhaust="
                            << result.RetryExhaustions
                            << " retry/success="
                            << std::fixed
                            << std::setprecision(4)
                            << retry_per_success
                            << '\n';

                        out.flush();
                        continue;
                    }

                    if (
                        result.Status ==
                        T::MutationResult::RETRY)
                    {
                        ++retry_cases;
                        repeat_retry = true;

                        out
                            << "RETRY"
                            << " repeat=" << repeat
                            << " case=" << (case_index + 1u)
                            << " N=" << config.NodeCount
                            << " K="
                            << static_cast<unsigned>(
                                config.ParentCapacity)
                            << " writers=" << writers
                            << " sample=" << sample
                            << " success=" << result.Success
                            << " retries=" << result.Retries
                            << " invalid="
                            << result.InvalidFailures
                            << " retry-exhaust="
                            << result.RetryExhaustions
                            << " retry/success="
                            << std::fixed
                            << std::setprecision(4)
                            << retry_per_success
                            << '\n';

                        out.flush();
                        continue;
                    }

                    const bool verify_ok =
                        B::VerifyMutationScenario(
                            backend,
                            scenario,
                            schedule,
                            writers);

                    if (!verify_ok)
                    {
                        ++invalid_cases;
                        repeat_invalid = true;

                        out
                            << "INVALID"
                            << " repeat=" << repeat
                            << " case=" << (case_index + 1u)
                            << " N=" << config.NodeCount
                            << " K="
                            << static_cast<unsigned>(
                                config.ParentCapacity)
                            << " writers=" << writers
                            << " sample=" << sample
                            << " reason=FINAL_VERIFY_FAILED"
                            << " success=" << result.Success
                            << " retries=" << result.Retries
                            << '\n';

                        out.flush();
                        continue;
                    }

                    ++committed_cases;
                }
            }
        }

        if (repeat_invalid)
            ++invalid_repetitions;

        if (repeat_retry)
            ++retry_repetitions;
    }

    out
        << "\n============================================================\n"
        << "FINAL SUMMARY\n"
        << "repeats=" << repeat_count << '\n'
        << "invalid_repetitions="
        << invalid_repetitions << '\n'
        << "retry_repetitions="
        << retry_repetitions << '\n'
        << "committed_cases="
        << committed_cases << '\n'
        << "invalid_cases="
        << invalid_cases << '\n'
        << "retry_cases="
        << retry_cases << '\n'
        << "result="
        << (invalid_cases == 0u ? "PASS" : "FAIL")
        << '\n'
        << "============================================================\n";

    out.flush();

    // Only INVALID / structural verification failure is a test failure.
    // RETRY is reported as a progress/contention event.
    return invalid_cases == 0u ? 0 : 1;
}



inline int RunDeltaRecursion(
    std::size_t repeat_count = 10u,
    std::size_t max_writers = 0u,
    std::uint32_t delta_recursion = 1u,
    std::uint32_t transaction_attempt_limit = 1'000u)
{
    constexpr std::uint32_t BASE_INTERNAL_RECURSION =
        BidirectionalInMemGraph::
        AdaptivePackedCellContainer::
        INTERNAL_RECURSION;

    if (
        delta_recursion == 0u ||
        transaction_attempt_limit == 0u)
    {
        return 3;
    }

    const std::size_t hardware_threads =
        static_cast<std::size_t>(
            std::thread::hardware_concurrency());

    if (max_writers == 0u)
    {
        max_writers = hardware_threads > 2u
            ? std::min<std::size_t>(
                16u,
                hardware_threads - 2u)
            : 1u;
    }
    // Creates a standard std::string
    std::string output_file_str = "SuperNovaTest2ARepeat_dl_recursion_LowerBound" + std::to_string(repeat_count) + ".txt";

    // If you absolutely need a const char* for an older API:
    const char* output_file = output_file_str.c_str();
    std::ofstream out(
        output_file,
        std::ios::out | std::ios::trunc);

    if (!out)
        return 2;

    const auto cases =
        T::MakeBenchmarkCases(
            100u,
            10'000u,
            4u,
            32u);

    std::size_t invalid_repetitions = 0u;
    std::size_t retry_repetitions = 0u;
    std::size_t escalated_repetitions = 0u;

    std::uint64_t invalid_cases = 0u;
    std::uint64_t retry_cases = 0u;
    std::uint64_t committed_cases = 0u;
    std::uint64_t escalated_cases = 0u;

    std::uint64_t total_stage_exhaustions = 0u;
    std::uint64_t total_escalated_commits = 0u;
    std::uint64_t total_terminal_retries = 0u;

    std::uint32_t global_max_internal_recursion =
        BASE_INTERNAL_RECURSION;
    std::uint32_t global_max_stage = 1u;

    out
        << "SUPERNOVA FABRIC TEST 2A CUMULATIVE DELTA-RECURSION TEST\n"
        << "repeats=" << repeat_count << '\n'
        << "writer_sweep=1.." << max_writers << '\n'
        << "measured_runs_per_point="
        << T::ConcurrencyConfig::MEASURED_RUNS << '\n'
        << "transaction_attempt_limit_per_stage="
        << transaction_attempt_limit << '\n'
        << "base_internal_recursion="
        << BASE_INTERNAL_RECURSION << '\n'
        << "delta_recursion="
        << delta_recursion << '\n'
        << "default_sequence=1,2,3,6,12,24,48,...\n"
        << "INVALID=FAIL\n"
        << "RETRY=ARITHMETIC_SAFETY_EXIT_ONLY\n\n";

    out.flush();

    for (
        std::size_t repeat = 1u;
        repeat <= repeat_count;
        ++repeat)
    {
        bool repeat_invalid = false;
        bool repeat_retry = false;
        bool repeat_escalated = false;

        for (
            std::size_t case_index = 0u;
            case_index < cases.size();
            ++case_index)
        {
            const T::BenchmarkCase& config =
                cases[case_index];

            const auto maybe_scenario =
                B::MakeMutationScenario(
                    config,
                    B::MutationLocality::HOTSPOT,
                    max_writers);

            if (!maybe_scenario.has_value())
            {
                ++invalid_cases;
                repeat_invalid = true;

                out
                    << "INVALID"
                    << " repeat=" << repeat
                    << " case=" << (case_index + 1u)
                    << " N=" << config.NodeCount
                    << " K="
                    << static_cast<unsigned>(
                        config.ParentCapacity)
                    << " reason=SCENARIO_BUILD_FAILED\n";
                out.flush();
                continue;
            }

            const B::MutationScenario scenario =
                maybe_scenario.value();

            const B::MutationSchedule schedule =
                B::BuildMutationSchedule(scenario);

            for (
                std::size_t writers = 1u;
                writers <= max_writers;
                ++writers)
            {
                for (
                    std::size_t sample = 1u;
                    sample <= T::ConcurrencyConfig::MEASURED_RUNS;
                    ++sample)
                {
                    T::RuntimeAPCFabricBackend backend{};

                    if (
                        !B::BuildMutationBackend(
                            backend,
                            scenario,
                            writers))
                    {
                        ++invalid_cases;
                        repeat_invalid = true;

                        out
                            << "INVALID"
                            << " repeat=" << repeat
                            << " case=" << (case_index + 1u)
                            << " N=" << config.NodeCount
                            << " K="
                            << static_cast<unsigned>(
                                config.ParentCapacity)
                            << " writers=" << writers
                            << " sample=" << sample
                            << " reason=BACKEND_BUILD_FAILED\n";
                        out.flush();
                        continue;
                    }

                    const B::MutationSweepResult result =
                        B::RunMutationWorkersDeltaRecursion(
                            backend,
                            scenario,
                            schedule,
                            writers,
                            delta_recursion,
                            transaction_attempt_limit);

                    total_stage_exhaustions +=
                        result.RecursionEscalations;
                    total_escalated_commits +=
                        result.EscalatedCommits;
                    total_terminal_retries +=
                        result.RetryExhaustions;

                    global_max_internal_recursion =
                        std::max(
                            global_max_internal_recursion,
                            result.MaxInternalRecursion);
                    global_max_stage =
                        std::max(
                            global_max_stage,
                            result.MaxEscalationStage);

                    if (result.RecursionEscalations != 0u)
                    {
                        ++escalated_cases;
                        repeat_escalated = true;
                    }

                    const double retry_per_success =
                        result.Success == 0u
                            ? 0.0
                            : static_cast<double>(
                                result.Retries) /
                              static_cast<double>(
                                result.Success);

                    if (
                        result.Status ==
                        T::MutationResult::INVALID)
                    {
                        ++invalid_cases;
                        repeat_invalid = true;

                        out
                            << "INVALID"
                            << " repeat=" << repeat
                            << " case=" << (case_index + 1u)
                            << " N=" << config.NodeCount
                            << " K="
                            << static_cast<unsigned>(
                                config.ParentCapacity)
                            << " writers=" << writers
                            << " sample=" << sample
                            << " success=" << result.Success
                            << " retries=" << result.Retries
                            << " invalid=" << result.InvalidFailures
                            << " stage-exhaustions="
                            << result.RecursionEscalations
                            << " escalated-commits="
                            << result.EscalatedCommits
                            << " max-internal-recursion="
                            << result.MaxInternalRecursion
                            << " max-stage="
                            << result.MaxEscalationStage
                            << " terminal-retry="
                            << result.RetryExhaustions
                            << " retry/success="
                            << std::fixed
                            << std::setprecision(4)
                            << retry_per_success
                            << '\n';

                        out.flush();
                        continue;
                    }

                    if (
                        result.Status ==
                        T::MutationResult::RETRY)
                    {
                        ++retry_cases;
                        repeat_retry = true;

                        out
                            << "RETRY"
                            << " repeat=" << repeat
                            << " case=" << (case_index + 1u)
                            << " N=" << config.NodeCount
                            << " K="
                            << static_cast<unsigned>(
                                config.ParentCapacity)
                            << " writers=" << writers
                            << " sample=" << sample
                            << " success=" << result.Success
                            << " retries=" << result.Retries
                            << " invalid=" << result.InvalidFailures
                            << " stage-exhaustions="
                            << result.RecursionEscalations
                            << " escalated-commits="
                            << result.EscalatedCommits
                            << " max-internal-recursion="
                            << result.MaxInternalRecursion
                            << " max-stage="
                            << result.MaxEscalationStage
                            << " terminal-retry="
                            << result.RetryExhaustions
                            << " retry/success="
                            << std::fixed
                            << std::setprecision(4)
                            << retry_per_success
                            << '\n';

                        out.flush();
                        continue;
                    }

                    const bool verify_ok =
                        B::VerifyMutationScenario(
                            backend,
                            scenario,
                            schedule,
                            writers);

                    if (!verify_ok)
                    {
                        ++invalid_cases;
                        repeat_invalid = true;

                        out
                            << "INVALID"
                            << " repeat=" << repeat
                            << " case=" << (case_index + 1u)
                            << " N=" << config.NodeCount
                            << " K="
                            << static_cast<unsigned>(
                                config.ParentCapacity)
                            << " writers=" << writers
                            << " sample=" << sample
                            << " reason=FINAL_VERIFY_FAILED"
                            << " success=" << result.Success
                            << " retries=" << result.Retries
                            << " max-internal-recursion="
                            << result.MaxInternalRecursion
                            << " max-stage="
                            << result.MaxEscalationStage
                            << '\n';

                        out.flush();
                        continue;
                    }

                    ++committed_cases;

                    if (result.RecursionEscalations != 0u)
                    {
                        out
                            << "ESCALATED"
                            << " repeat=" << repeat
                            << " case=" << (case_index + 1u)
                            << " N=" << config.NodeCount
                            << " K="
                            << static_cast<unsigned>(
                                config.ParentCapacity)
                            << " writers=" << writers
                            << " sample=" << sample
                            << " success=" << result.Success
                            << " retries=" << result.Retries
                            << " stage-exhaustions="
                            << result.RecursionEscalations
                            << " escalated-commits="
                            << result.EscalatedCommits
                            << " max-internal-recursion="
                            << result.MaxInternalRecursion
                            << " max-stage="
                            << result.MaxEscalationStage
                            << " terminal-retry=0"
                            << " retry/success="
                            << std::fixed
                            << std::setprecision(4)
                            << retry_per_success
                            << '\n';

                        out.flush();
                    }
                }
            }
        }

        if (repeat_invalid)
            ++invalid_repetitions;
        if (repeat_retry)
            ++retry_repetitions;
        if (repeat_escalated)
            ++escalated_repetitions;
    }

    out
        << "\n============================================================\n"
        << "FINAL SUMMARY\n"
        << "repeats=" << repeat_count << '\n'
        << "transaction_attempt_limit_per_stage="
        << transaction_attempt_limit << '\n'
        << "base_internal_recursion="
        << BASE_INTERNAL_RECURSION << '\n'
        << "delta_recursion="
        << delta_recursion << '\n'
        << "invalid_repetitions="
        << invalid_repetitions << '\n'
        << "retry_repetitions="
        << retry_repetitions << '\n'
        << "escalated_repetitions="
        << escalated_repetitions << '\n'
        << "committed_cases="
        << committed_cases << '\n'
        << "invalid_cases="
        << invalid_cases << '\n'
        << "retry_cases="
        << retry_cases << '\n'
        << "escalated_cases="
        << escalated_cases << '\n'
        << "total_stage_exhaustions="
        << total_stage_exhaustions << '\n'
        << "total_escalated_commits="
        << total_escalated_commits << '\n'
        << "terminal_retry_exhaustions="
        << total_terminal_retries << '\n'
        << "max_internal_recursion_reached="
        << global_max_internal_recursion << '\n'
        << "max_escalation_stage_reached="
        << global_max_stage << '\n'
        << "result="
        << (invalid_cases == 0u ? "PASS" : "FAIL")
        << '\n'
        << "============================================================\n";

    out.flush();

    return invalid_cases == 0u ? 0 : 1;
}


} // namespace SuperNovaTest2ARepeat
