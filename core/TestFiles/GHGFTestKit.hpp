#ifndef APC_DAG_TEST_EXTERNAL_TYPES
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"
#include "Models/GHGF/GHGFModelOfAPC.hpp"

#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace APCDAGTests
{
    namespace GHGFTest01
    {
        using GM = GHGFLayerModel;
        using Role = GM::GHGFNodeRole;

        static constexpr uint32_t BATCH = 2u;
        static constexpr uint32_t TRAIN_STEPS = 128u;
        static constexpr uint32_t TEST_STEPS = 64u;

        static constexpr std::size_t CONSTRUCTION_SAMPLES = 31u;
        static constexpr std::size_t TIMING_SAMPLES = 31u;

        inline bool ConstructModel(
            GHGFModelConstructor& model,
            const GM::GHGFStorageProfile& profile
        ) noexcept
        {
            std::array<GHGFNode, 3u> nodes{};

            const std::array roles{
                Role::VOLATILE,
                Role::VALUE,
                Role::OBSERVATION
            };

            const std::array<GM::GHGFConnection, 2u> connections{{
                {
                    0u,
                    1u,
                    FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
                    1.0f
                },
                {
                    1u,
                    2u,
                    FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                    1.0f
                }
            }};

            GHGFModelConstructor::GHGFModelConstructionValues values{
                nodes,
                roles,
                connections
            };

            return model.ConstructGHGFModel(
                values,
                profile
            );
        }


        inline void BuildDataset(
            std::vector<float>& training,
            std::vector<float>& testing
        )
        {
            for (
                uint32_t time = 0u;
                time < TRAIN_STEPS + TEST_STEPS;
                ++time
            )
            {
                const float first =
                    time % 10u < 8u
                        ? 1.0f
                        : 0.0f;

                std::vector<float>& destination =
                    time < TRAIN_STEPS
                        ? training
                        : testing;

                const uint32_t local_time =
                    time < TRAIN_STEPS
                        ? time
                        : time - TRAIN_STEPS;

                destination[
                    static_cast<std::size_t>(local_time) * BATCH
                ] = first;

                destination[
                    static_cast<std::size_t>(local_time) * BATCH + 1u
                ] = 1.0f - first;
            }
        }


        inline double ReferenceLogLoss(
            std::span<const float> observations,
            uint32_t time_count,
            const std::array<double, BATCH>& probability
        ) noexcept
        {
            double loss = 0.0;

            for (uint32_t time = 0u; time < time_count; ++time)
            {
                for (uint32_t lane = 0u; lane < BATCH; ++lane)
                {
                    const double p = std::clamp(
                        probability[lane],
                        1.0e-12,
                        1.0 - 1.0e-12
                    );

                    const float observed =
                        observations[
                            static_cast<std::size_t>(time) * BATCH + lane
                        ];

                    loss -= observed == 1.0f
                        ? std::log(p)
                        : std::log1p(-p);
                }
            }

            return loss /
                static_cast<double>(
                    static_cast<std::size_t>(time_count) * BATCH
                );
        }


        inline std::array<double, BATCH> TrainingFrequency(
            std::span<const float> observations
        ) noexcept
        {
            std::array<double, BATCH> probability{};

            for (uint32_t lane = 0u; lane < BATCH; ++lane)
            {
                double sum = 0.0;

                for (uint32_t time = 0u; time < TRAIN_STEPS; ++time)
                {
                    sum += observations[
                        static_cast<std::size_t>(time) * BATCH + lane
                    ];
                }

                probability[lane] =
                    sum / static_cast<double>(TRAIN_STEPS);
            }

            return probability;
        }


        inline double BinaryAccuracy(
            std::span<const float> observations,
            std::span<const float> predictions
        ) noexcept
        {
            if (
                observations.size() != predictions.size() ||
                observations.empty()
            )
            {
                return 0.0;
            }

            std::size_t correct = 0u;

            for (std::size_t i = 0u; i < observations.size(); ++i)
            {
                const float predicted =
                    predictions[i] >= 0.5f ? 1.0f : 0.0f;

                correct +=
                    predicted == observations[i]
                        ? 1u
                        : 0u;
            }

            return static_cast<double>(correct) /
                static_cast<double>(observations.size());
        }


        inline bool ValidPredictions(
            std::span<const float> predictions
        ) noexcept
        {
            return std::all_of(
                predictions.begin(),
                predictions.end(),
                [](float value) noexcept
                {
                    return
                        std::isfinite(value) &&
                        value > 0.0f &&
                        value < 1.0f;
                }
            );
        }


        inline double MaxAbsoluteDifference(
            std::span<const float> first,
            std::span<const float> second
        ) noexcept
        {
            if (first.size() != second.size())
            {
                return std::numeric_limits<double>::infinity();
            }

            double maximum = 0.0;

            for (std::size_t i = 0u; i < first.size(); ++i)
            {
                maximum = std::max(
                    maximum,
                    std::abs(
                        static_cast<double>(first[i]) -
                        static_cast<double>(second[i])
                    )
                );
            }

            return maximum;
        }


        template <std::size_t SAMPLE_COUNT, typename Operation>
        std::optional<double> MedianNanoseconds(
            Operation&& operation
        )
        {
            std::array<double, SAMPLE_COUNT> samples{};

            for (std::size_t sample = 0u;
                sample < SAMPLE_COUNT;
                ++sample)
            {
                const auto begin = Clock::now();

                if (!operation())
                {
                    return std::nullopt;
                }

                const auto end = Clock::now();

                samples[sample] =
                    std::chrono::duration<double, std::nano>(
                        end - begin
                    ).count();
            }

            return Median(samples);
        }


        inline Result Run()
        {
            Banner(
                "TEST 8 - GHGF SCALAR QUALITY / FIT / TIMING COMPARISON"
            );

            GM::GHGFStorageProfile profile{};

            if (!GM::MakeDefaultGHGFStorageProfile(
                profile,
                BATCH,
                2u
            ))
            {
                std::cout << "  profile construction: FAIL\n";
                return Result::FAIL;
            }


            std::vector<float> training(
                static_cast<std::size_t>(TRAIN_STEPS) * BATCH
            );

            std::vector<float> testing(
                static_cast<std::size_t>(TEST_STEPS) * BATCH
            );

            BuildDataset(training, testing);


            // --------------------------------------------------------
            // Very small reference baselines.
            // --------------------------------------------------------

            const std::array<double, BATCH> uniform_probability{
                0.5,
                0.5
            };

            const std::array<double, BATCH> empirical_probability =
                TrainingFrequency(training);

            const double uniform_test_loss =
                ReferenceLogLoss(
                    testing,
                    TEST_STEPS,
                    uniform_probability
                );

            const double empirical_test_loss =
                ReferenceLogLoss(
                    testing,
                    TEST_STEPS,
                    empirical_probability
                );


            // --------------------------------------------------------
            // Construction timing.
            // --------------------------------------------------------

            const std::optional<double> construction_ns =
                MedianNanoseconds<CONSTRUCTION_SAMPLES>(
                    [&]() noexcept -> bool
                    {
                        GHGFModelConstructor candidate{};

                        if (!ConstructModel(candidate, profile))
                        {
                            return false;
                        }

                        candidate.ShutDownFabric();
                        return true;
                    }
                );

            if (!construction_ns.has_value())
            {
                std::cout << "  construction benchmark: FAIL\n";
                return Result::FAIL;
            }


            // --------------------------------------------------------
            // Real test model.
            // --------------------------------------------------------

            GHGFModelConstructor model{};

            if (!ConstructModel(model, profile))
            {
                std::cout << "  model construction: FAIL\n";
                return Result::FAIL;
            }


            // --------------------------------------------------------
            // Unfitted model.
            // --------------------------------------------------------

            const auto baseline_train_begin = Clock::now();

            const std::optional<double> baseline_train_loss =
                model.RunGHGFSequence(
                    training,
                    TRAIN_STEPS,
                    BATCH,
                    {},
                    true
                );

            const auto baseline_train_end = Clock::now();

            if (
                !baseline_train_loss.has_value() ||
                !std::isfinite(baseline_train_loss.value())
            )
            {
                std::cout << "  baseline training replay: FAIL\n";
                model.ShutDownFabric();
                return Result::FAIL;
            }


            std::vector<float> baseline_predictions(
                static_cast<std::size_t>(TEST_STEPS) * BATCH
            );

            const auto baseline_test_begin = Clock::now();

            const std::optional<double> baseline_test_loss =
                model.RunGHGFSequence(
                    testing,
                    TEST_STEPS,
                    BATCH,
                    baseline_predictions,
                    false
                );

            const auto baseline_test_end = Clock::now();

            if (
                !baseline_test_loss.has_value() ||
                !std::isfinite(baseline_test_loss.value()) ||
                !ValidPredictions(baseline_predictions)
            )
            {
                std::cout << "  baseline held-out inference: FAIL\n";
                model.ShutDownFabric();
                return Result::FAIL;
            }


            // --------------------------------------------------------
            // Fit exactly the same parameter as your original Test 8.
            // --------------------------------------------------------

            const std::array<GM::GHGFParameterRange, 1u> parameters{{
                {
                    1u,
                    static_cast<uint32_t>(
                        GM::GHGFErrorValueIndexing::TONIC_VOLATILE
                    ),
                    -6.0f,
                    -2.0f
                }
            }};


            const auto fit_begin = Clock::now();

            const std::optional<double> fitted_train_loss =
                model.FitGHGFParameters(
                    training,
                    TRAIN_STEPS,
                    BATCH,
                    parameters,
                    5u
                );

            const auto fit_end = Clock::now();

            if (
                !fitted_train_loss.has_value() ||
                !std::isfinite(fitted_train_loss.value())
            )
            {
                std::cout << "  parameter fitting: FAIL\n";
                model.ShutDownFabric();
                return Result::FAIL;
            }


            // --------------------------------------------------------
            // Held-out evaluation after fitting.
            // FitGHGFParameters leaves the training posterior active.
            // --------------------------------------------------------

            std::vector<float> fitted_predictions(
                static_cast<std::size_t>(TEST_STEPS) * BATCH
            );

            const auto fitted_test_begin = Clock::now();

            const std::optional<double> fitted_test_loss =
                model.RunGHGFSequence(
                    testing,
                    TEST_STEPS,
                    BATCH,
                    fitted_predictions,
                    false
                );

            const auto fitted_test_end = Clock::now();

            if (
                !fitted_test_loss.has_value() ||
                !std::isfinite(fitted_test_loss.value()) ||
                !ValidPredictions(fitted_predictions)
            )
            {
                std::cout << "  fitted held-out inference: FAIL\n";
                model.ShutDownFabric();
                return Result::FAIL;
            }


            // --------------------------------------------------------
            // Determinism / replay proof.
            // Reconstruct exactly the same training posterior and
            // repeat held-out inference with the fitted parameters.
            // --------------------------------------------------------

            if (!model.RunGHGFSequence(
                training,
                TRAIN_STEPS,
                BATCH,
                {},
                true
            ).has_value())
            {
                std::cout << "  deterministic replay preparation: FAIL\n";
                model.ShutDownFabric();
                return Result::FAIL;
            }

            std::vector<float> replay_predictions(
                fitted_predictions.size()
            );

            const std::optional<double> replay_loss =
                model.RunGHGFSequence(
                    testing,
                    TEST_STEPS,
                    BATCH,
                    replay_predictions,
                    false
                );

            if (!replay_loss.has_value())
            {
                std::cout << "  deterministic replay: FAIL\n";
                model.ShutDownFabric();
                return Result::FAIL;
            }

            const double replay_difference =
                MaxAbsoluteDifference(
                    fitted_predictions,
                    replay_predictions
                );

            const bool deterministic =
                replay_difference <= 1.0e-6 &&
                std::abs(
                    replay_loss.value() -
                    fitted_test_loss.value()
                ) <= 1.0e-9;


            // --------------------------------------------------------
            // Median fitted scalar replay timing.
            // --------------------------------------------------------

            const std::optional<double> train_replay_ns =
                MedianNanoseconds<TIMING_SAMPLES>(
                    [&]() noexcept -> bool
                    {
                        const std::optional<double> loss =
                            model.RunGHGFSequence(
                                training,
                                TRAIN_STEPS,
                                BATCH,
                                {},
                                true
                            );

                        return
                            loss.has_value() &&
                            std::isfinite(loss.value());
                    }
                );

            if (!train_replay_ns.has_value())
            {
                std::cout << "  training replay benchmark: FAIL\n";
                model.ShutDownFabric();
                return Result::FAIL;
            }


            std::array<double, TIMING_SAMPLES> test_samples{};

            for (std::size_t sample = 0u;
                sample < TIMING_SAMPLES;
                ++sample)
            {
                // Recreate the same training posterior.
                if (!model.RunGHGFSequence(
                    training,
                    TRAIN_STEPS,
                    BATCH,
                    {},
                    true
                ).has_value())
                {
                    std::cout << "  held-out benchmark preparation: FAIL\n";
                    model.ShutDownFabric();
                    return Result::FAIL;
                }

                const auto begin = Clock::now();

                const std::optional<double> loss =
                    model.RunGHGFSequence(
                        testing,
                        TEST_STEPS,
                        BATCH,
                        {},
                        false
                    );

                const auto end = Clock::now();

                if (!loss.has_value())
                {
                    std::cout << "  held-out benchmark: FAIL\n";
                    model.ShutDownFabric();
                    return Result::FAIL;
                }

                test_samples[sample] =
                    std::chrono::duration<double, std::nano>(
                        end - begin
                    ).count();
            }

            const double test_replay_ns =
                Median(test_samples);


            // --------------------------------------------------------
            // Derived metrics.
            // --------------------------------------------------------

            const double baseline_train_ns =
                std::chrono::duration<double, std::nano>(
                    baseline_train_end - baseline_train_begin
                ).count();

            const double baseline_test_ns =
                std::chrono::duration<double, std::nano>(
                    baseline_test_end - baseline_test_begin
                ).count();

            const double fit_ns =
                std::chrono::duration<double, std::nano>(
                    fit_end - fit_begin
                ).count();

            const double fitted_test_ns =
                std::chrono::duration<double, std::nano>(
                    fitted_test_end - fitted_test_begin
                ).count();

            const double training_items =
                static_cast<double>(TRAIN_STEPS) *
                static_cast<double>(BATCH);

            const double testing_items =
                static_cast<double>(TEST_STEPS) *
                static_cast<double>(BATCH);

            const double train_ns_per_item =
                train_replay_ns.value() / training_items;

            const double test_ns_per_item =
                test_replay_ns / testing_items;

            const double train_improvement =
                baseline_train_loss.value() > 0.0
                    ? 100.0 *
                        (
                            baseline_train_loss.value() -
                            fitted_train_loss.value()
                        ) /
                        baseline_train_loss.value()
                    : 0.0;

            const double test_improvement =
                baseline_test_loss.value() > 0.0
                    ? 100.0 *
                        (
                            baseline_test_loss.value() -
                            fitted_test_loss.value()
                        ) /
                        baseline_test_loss.value()
                    : 0.0;

            const double baseline_accuracy =
                BinaryAccuracy(
                    testing,
                    baseline_predictions
                );

            const double fitted_accuracy =
                BinaryAccuracy(
                    testing,
                    fitted_predictions
                );


            const bool training_non_regression =
                fitted_train_loss.value() <=
                    baseline_train_loss.value() + 1.0e-12;

            const bool heldout_improved =
                fitted_test_loss.value() <
                    baseline_test_loss.value();

            const bool beats_uniform =
                fitted_test_loss.value() <
                    uniform_test_loss;

            const bool beats_empirical =
                fitted_test_loss.value() <
                    empirical_test_loss;


            // --------------------------------------------------------
            // Report.
            // --------------------------------------------------------

            std::cout
                << std::fixed
                << std::setprecision(6)

                << "\nREFERENCE BASELINES\n"
                << "  uniform 0.5 held-out loss       : "
                << uniform_test_loss
                << '\n'

                << "  training-frequency probabilities: ["
                << empirical_probability[0u]
                << ", "
                << empirical_probability[1u]
                << "]\n"

                << "  frequency held-out loss         : "
                << empirical_test_loss
                << '\n'

                << "\nGHGF QUALITY\n"
                << "  unfitted training loss          : "
                << baseline_train_loss.value()
                << '\n'

                << "  fitted training loss            : "
                << fitted_train_loss.value()
                << '\n'

                << "  training improvement            : "
                << train_improvement
                << " %\n"

                << "  unfitted held-out loss          : "
                << baseline_test_loss.value()
                << '\n'

                << "  fitted held-out loss            : "
                << fitted_test_loss.value()
                << '\n'

                << "  held-out improvement            : "
                << test_improvement
                << " %\n"

                << "  unfitted held-out accuracy      : "
                << baseline_accuracy * 100.0
                << " %\n"

                << "  fitted held-out accuracy        : "
                << fitted_accuracy * 100.0
                << " %\n"

                << "  fitted beats 0.5 baseline       : "
                << (beats_uniform ? "YES" : "NO")
                << '\n'

                << "  fitted beats frequency baseline : "
                << (beats_empirical ? "YES" : "NO")
                << '\n'

                << "  fitting improves held-out       : "
                << (heldout_improved ? "YES" : "NO")
                << '\n'

                << "  deterministic replay max diff   : "
                << replay_difference
                << "  "
                << (deterministic ? "PASS" : "FAIL")
                << '\n'

                << "\nTIMING\n"
                << std::setprecision(2)

                << "  model construction median       : "
                << construction_ns.value() / 1000.0
                << " us\n"

                << "  one unfitted training replay    : "
                << baseline_train_ns / 1000.0
                << " us\n"

                << "  one unfitted held-out replay    : "
                << baseline_test_ns / 1000.0
                << " us\n"

                << "  parameter fit total             : "
                << fit_ns / 1.0e6
                << " ms\n"

                << "  one fitted held-out replay      : "
                << fitted_test_ns / 1000.0
                << " us\n"

                << "  fitted train replay median      : "
                << train_replay_ns.value() / 1000.0
                << " us\n"

                << "  fitted held-out replay median   : "
                << test_replay_ns / 1000.0
                << " us\n"

                << "  fitted train scalar cost        : "
                << train_ns_per_item
                << " ns/scored-item\n"

                << "  fitted test scalar cost         : "
                << test_ns_per_item
                << " ns/scored-item\n";


            const bool ok =
                training_non_regression &&
                deterministic &&
                ValidPredictions(baseline_predictions) &&
                ValidPredictions(fitted_predictions) &&
                std::isfinite(uniform_test_loss) &&
                std::isfinite(empirical_test_loss);


            std::cout
                << "\nTEST 8 MECHANICAL RESULT: "
                << (ok ? "PASS" : "FAIL")
                << '\n'

                << "TEST 8 QUALITY SIGNAL   : "
                << (
                    beats_empirical
                        ? "GHGF BEATS TRIVIAL FREQUENCY BASELINE"
                        : beats_uniform
                            ? "GHGF BEATS 0.5 BUT NOT FREQUENCY BASELINE"
                            : "GHGF DOES NOT YET BEAT 0.5 BASELINE"
                )
                << '\n';


            model.ShutDownFabric();

            return ok
                ? Result::PASS
                : Result::FAIL;
        }
    }
}