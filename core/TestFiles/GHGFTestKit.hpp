// GHGFTestKit.hpp
// Internal-learning validation for SuperNova GHGF (C++20)
//
// Standalone runner:
//
//   #include "GHGFTestKit.hpp"
//
//   int main()
//   {
//       return GHGFTestKit::RunAll();
//   }
//
// The tests use only the public GHGFModelConstructor API. They deliberately do
// not expose or mutate private WEIGHT_SLOT data. Where a slow reference is
// needed, FitGHGFParameters() is used as the global fitting oracle.
//
// Paper use:
//   * compile with the same C++20 release flags on every compared platform;
//   * report the compiler, CPU, OS, flags, and repeated-run distribution;
//   * call concurrent results aggregate throughput, not per-call latency;
//   * Test I overlaps one numeric worker with structural mutation and snapshot
//     reads; it does not claim that multiple numeric writers are safe;
//   * Test J measures independent-model throughput, not same-model latency.
// ============================================================================

#ifndef APC_DAG_TEST_EXTERNAL_TYPES
#include "Models/GHGF/GHGFModelOfAPC.hpp"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace GHGFTestKit
{
using namespace BidirectionalInMemGraph;

using GM = GHGFLayerModel;
using Role = GM::GHGFNodeRole;

static constexpr std::uint8_t PARENT_CAPACITY = 2u;
static constexpr double LOSS_EPS = 1.0e-10;
static constexpr double DELTA_EPS = 1.0e-12;

struct TestConst final
{
    static constexpr float BINARY_ZERO = 0.0f;
    static constexpr float BINARY_ONE = 1.0f;
    static constexpr double HALF = 0.5;
    static constexpr double PROBABILITY_CLIP = 1.0e-12;
    static constexpr double API_LOSS_TOLERANCE = 1.0e-9;
    static constexpr float DEFAULT_GRADIENT_CLIP = 10.0f;
    static constexpr float DEFAULT_MIN_TONIC_LOG_VOLATILITY = -20.0f;
    static constexpr float DEFAULT_MAX_TONIC_LOG_VOLATILITY = 10.0f;
    static constexpr int REPORT_WIDTH = 48;
    static constexpr int SHORT_PRECISION = 6;
    static constexpr int LONG_PRECISION = 9;
    static constexpr int SUMMARY_PRECISION = 3;
    static constexpr int THREAD_COLUMN_WIDTH = 2;
    static constexpr int THROUGHPUT_COLUMN_WIDTH = 12;
    static constexpr int SPEEDUP_COLUMN_WIDTH = 10;
    static constexpr int SINGLE_CANDIDATE_PARENT_COUNT = 1;

    static constexpr std::uint32_t VALUE_SLOT = 0u;
    static constexpr std::uint32_t SECOND_VALUE_SLOT = 1u;
    static constexpr std::uint32_t TWO_NODE_OBSERVATION_SLOT = 1u;
    static constexpr std::uint32_t THREE_NODE_OBSERVATION_SLOT = 2u;
    static constexpr std::size_t TWO_NODE_COUNT = 2u;
    static constexpr std::size_t THREE_NODE_COUNT = 3u;
    static constexpr std::size_t ONE_EDGE_COUNT = 1u;
    static constexpr std::size_t TWO_EDGE_COUNT = 2u;
    static constexpr std::size_t ONE_PARAMETER_COUNT = 1u;
};

static_assert(std::is_trivially_copyable_v<GHGFLearningConfig>);

// ----------------------------------------------------------------------------
// Small helpers
// ----------------------------------------------------------------------------

struct ScopedModel final
{
    GHGFModelConstructor Model{};

    ScopedModel() = default;
    ScopedModel(const ScopedModel&) = delete;
    ScopedModel& operator=(const ScopedModel&) = delete;

    ~ScopedModel()
    {
        if (Model.IsFabricActive())
            Model.ShutDownFabric();
    }
};

struct Evaluation final
{
    bool Valid = false;
    double Loss = std::numeric_limits<double>::infinity();
    std::vector<float> Predictions{};
};

inline void Banner(const char* title)
{
    std::cout
        << "\n================================================================================\n"
        << title
        << "\n================================================================================\n";
}

inline void Report(
    const char* name,
    bool passed
)
{
    std::cout
        << "  " << std::left << std::setw(TestConst::REPORT_WIDTH) << name
        << (passed ? "PASS" : "FAIL") << '\n';
}

inline GHGFLearningConfig ZeroLearning() noexcept
{
    GHGFLearningConfig learning{};
    learning.HCouplingLearningRate = TestConst::BINARY_ZERO;
    learning.DriftLearningRate = TestConst::BINARY_ZERO;
    learning.VolatilityLearningRate = TestConst::BINARY_ZERO;
    learning.VCouplingLearningRate = TestConst::BINARY_ZERO;
    learning.AutoConnectionLearningRate = TestConst::BINARY_ZERO;
    learning.GradientClip = TestConst::DEFAULT_GRADIENT_CLIP;
    learning.MinTonicLogVolatility =
        TestConst::DEFAULT_MIN_TONIC_LOG_VOLATILITY;
    learning.MaxTonicLogVolatility =
        TestConst::DEFAULT_MAX_TONIC_LOG_VOLATILITY;
    return learning;
}

inline bool MakeProfile(
    GM::GHGFStorageProfile& profile,
    std::uint32_t batch
) noexcept
{
    return
        batch != 0u &&
        GM::MakeDefaultGHGFStorageProfile(
            profile,
            batch,
            PARENT_CAPACITY
        );
}

inline bool ConstructValueObservationModel(
    GHGFModelConstructor& model,
    std::uint32_t batch,
    float h_coupling
) noexcept
{
    if (!std::isfinite(h_coupling))
        return false;

    GM::GHGFStorageProfile profile{};
    if (!MakeProfile(profile, batch))
        return false;

    std::array<GHGFNode, TestConst::TWO_NODE_COUNT> nodes{};

    const std::array roles{
        Role::VALUE,
        Role::OBSERVATION
    };

    const std::array<GM::GHGFConnection, TestConst::ONE_EDGE_COUNT> connections{{
        {
            TestConst::VALUE_SLOT,
            TestConst::TWO_NODE_OBSERVATION_SLOT,
            FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
            h_coupling
        }
    }};

    GHGFModelConstructor::GHGFModelConstructionValues values{
        nodes,
        roles,
        connections
    };

    return model.ConstructGHGFModel(values, profile);
}

inline bool ConstructVolatileValueObservationModel(
    GHGFModelConstructor& model,
    std::uint32_t batch,
    float v_coupling,
    float h_coupling
) noexcept
{
    if (!std::isfinite(v_coupling) || !std::isfinite(h_coupling))
        return false;

    GM::GHGFStorageProfile profile{};
    if (!MakeProfile(profile, batch))
        return false;

    std::array<GHGFNode, TestConst::THREE_NODE_COUNT> nodes{};

    const std::array roles{
        Role::VOLATILE,
        Role::VALUE,
        Role::OBSERVATION
    };

    const std::array<GM::GHGFConnection, TestConst::TWO_EDGE_COUNT> connections{{
        {
            TestConst::VALUE_SLOT,
            TestConst::SECOND_VALUE_SLOT,
            FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
            v_coupling
        },
        {
            TestConst::SECOND_VALUE_SLOT,
            TestConst::THREE_NODE_OBSERVATION_SLOT,
            FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
            h_coupling
        }
    }};

    GHGFModelConstructor::GHGFModelConstructionValues values{
        nodes,
        roles,
        connections
    };

    return model.ConstructGHGFModel(values, profile);
}

enum class PredictiveStructure : std::uint8_t
{
    SHALLOW = 0u,
    HIERARCHICAL = 1u
};

inline bool ConstructPredictiveStructure(
    GHGFModelConstructor& model,
    std::uint32_t batch,
    PredictiveStructure structure,
    float hidden_coupling,
    float observation_coupling,
    bool structural_learning_active = false
) noexcept
{
    if (
        !std::isfinite(hidden_coupling) ||
        !std::isfinite(observation_coupling)
    )
    {
        return false;
    }

    GM::GHGFStorageProfile profile{};
    if (!MakeProfile(profile, batch))
        return false;

    std::array<GHGFNode, TestConst::THREE_NODE_COUNT> nodes{};
    const std::array roles{
        Role::VALUE,
        Role::VALUE,
        Role::OBSERVATION
    };

    const GM::GHGFConnection hidden_edge{
        TestConst::VALUE_SLOT,
        TestConst::SECOND_VALUE_SLOT,
        FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
        hidden_coupling
    };

    const GM::GHGFConnection observation_edge{
        TestConst::SECOND_VALUE_SLOT,
        TestConst::THREE_NODE_OBSERVATION_SLOT,
        FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
        observation_coupling
    };

    const std::array<GM::GHGFConnection, TestConst::TWO_EDGE_COUNT>
        hierarchical_edges{{hidden_edge, observation_edge}};

    const std::array<GM::GHGFConnection, TestConst::ONE_EDGE_COUNT>
        shallow_edges{{observation_edge}};

    const std::span<const GM::GHGFConnection> edges =
        structure == PredictiveStructure::HIERARCHICAL
            ? std::span<const GM::GHGFConnection>(hierarchical_edges)
            : std::span<const GM::GHGFConnection>(shallow_edges);

    GHGFModelConstructor::GHGFModelConstructionValues values{
        nodes,
        roles,
        edges,
        structural_learning_active
    };

    return model.ConstructGHGFModel(values, profile);
}

inline std::vector<float> ConstantDataset(
    std::uint32_t steps,
    std::uint32_t batch,
    float value
)
{
    return std::vector<float>(
        static_cast<std::size_t>(steps) * batch,
        value
    );
}

inline std::vector<float> FixedBernoulliBatchDataset(
    std::uint32_t steps,
    std::uint32_t batch,
    std::uint32_t ones_per_batch
)
{
    std::vector<float> data(
        static_cast<std::size_t>(steps) * batch,
        TestConst::BINARY_ZERO
    );

    ones_per_batch = std::min(ones_per_batch, batch);

    for (std::uint32_t t = 0u; t < steps; ++t)
    {
        for (std::uint32_t lane = 0u; lane < ones_per_batch; ++lane)
            data[static_cast<std::size_t>(t) * batch + lane] =
                TestConst::BINARY_ONE;
    }

    return data;
}

inline std::vector<float> AlternatingDataset(
    std::uint32_t steps,
    std::uint32_t batch,
    std::uint32_t time_offset = 0u
)
{
    std::vector<float> data(
        static_cast<std::size_t>(steps) * batch,
        TestConst::BINARY_ZERO
    );

    for (std::uint32_t t = 0u; t < steps; ++t)
    {
        const float value =
            ((t + time_offset) & 1u) != 0u
                ? TestConst::BINARY_ONE
                : TestConst::BINARY_ZERO;

        for (std::uint32_t lane = 0u; lane < batch; ++lane)
            data[static_cast<std::size_t>(t) * batch + lane] = value;
    }

    return data;
}

inline std::vector<float> BlockDataset(
    std::uint32_t steps,
    std::uint32_t batch,
    std::uint32_t block_length,
    std::uint32_t time_offset = 0u
)
{
    std::vector<float> data(
        static_cast<std::size_t>(steps) * batch,
        TestConst::BINARY_ZERO
    );

    if (block_length == 0u)
        return data;

    for (std::uint32_t t = 0u; t < steps; ++t)
    {
        const std::uint32_t global_time = t + time_offset;
        const float value =
            ((global_time / block_length) & 1u) != 0u
                ? TestConst::BINARY_ONE
                : TestConst::BINARY_ZERO;

        for (std::uint32_t lane = 0u; lane < batch; ++lane)
            data[static_cast<std::size_t>(t) * batch + lane] = value;
    }

    return data;
}

inline bool ValidPredictions(
    std::span<const float> predictions
) noexcept
{
    if (predictions.empty())
        return false;

    return std::all_of(
        predictions.begin(),
        predictions.end(),
        [](float value) noexcept
        {
            return
                std::isfinite(value) &&
                value > TestConst::BINARY_ZERO &&
                value < TestConst::BINARY_ONE;
        }
    );
}

inline double BinaryLogLoss(
    std::span<const float> observations,
    std::span<const float> predictions
) noexcept
{
    if (
        observations.empty() ||
        observations.size() != predictions.size()
    )
    {
        return std::numeric_limits<double>::infinity();
    }

    double loss = 0.0;

    for (std::size_t i = 0u; i < observations.size(); ++i)
    {
        const double p = std::clamp(
            static_cast<double>(predictions[i]),
            TestConst::PROBABILITY_CLIP,
            1.0 - TestConst::PROBABILITY_CLIP
        );

        loss -=
            observations[i] == TestConst::BINARY_ONE
                ? std::log(p)
                : std::log1p(-p);
    }

    return loss /
        static_cast<double>(observations.size());
}

inline double MaxAbsoluteDifference(
    std::span<const float> first,
    std::span<const float> second
) noexcept
{
    if (first.size() != second.size())
        return std::numeric_limits<double>::infinity();

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

inline double DeltaSquared(
    std::span<const float> base,
    std::span<const float> changed
) noexcept
{
    if (base.size() != changed.size())
        return 0.0;

    double sum = 0.0;

    for (std::size_t i = 0u; i < base.size(); ++i)
    {
        const double delta =
            static_cast<double>(changed[i]) -
            static_cast<double>(base[i]);

        sum += delta * delta;
    }

    return sum;
}

inline double DirectionDot(
    std::span<const float> base,
    std::span<const float> local,
    std::span<const float> oracle
) noexcept
{
    if (
        base.size() != local.size() ||
        base.size() != oracle.size()
    )
    {
        return 0.0;
    }

    double dot = 0.0;

    for (std::size_t i = 0u; i < base.size(); ++i)
    {
        const double local_delta =
            static_cast<double>(local[i]) -
            static_cast<double>(base[i]);

        const double oracle_delta =
            static_cast<double>(oracle[i]) -
            static_cast<double>(base[i]);

        dot += local_delta * oracle_delta;
    }

    return dot;
}

inline Evaluation Evaluate(
    GHGFModelConstructor& model,
    std::span<const float> observations,
    std::uint32_t steps,
    std::uint32_t batch,
    bool reset_state = true
)
{
    Evaluation result{};

    const std::size_t expected =
        static_cast<std::size_t>(steps) * batch;

    if (
        steps == 0u ||
        batch == 0u ||
        observations.size() != expected ||
        !std::all_of(
            observations.begin(),
            observations.end(),
            [](float value) noexcept
            {
                return
                    value == TestConst::BINARY_ZERO ||
                    value == TestConst::BINARY_ONE;
            }
        )
    )
    {
        return result;
    }

    result.Predictions.resize(expected);

    const std::optional<double> loss =
        model.RunGHGFSequence(
            observations,
            steps,
            batch,
            result.Predictions,
            reset_state
        );

    if (
        !loss.has_value() ||
        !std::isfinite(loss.value()) ||
        !ValidPredictions(result.Predictions)
    )
    {
        return result;
    }

    const double independently_computed_loss =
        BinaryLogLoss(observations, result.Predictions);

    if (
        !std::isfinite(independently_computed_loss) ||
        std::abs(loss.value() - independently_computed_loss) >
            TestConst::API_LOSS_TOLERANCE
    )
    {
        return result;
    }

    result.Valid = true;
    result.Loss = independently_computed_loss;
    return result;
}

inline bool TrainSequence(
    GHGFModelConstructor& model,
    std::span<const float> observations,
    std::uint32_t steps,
    std::uint32_t batch,
    const GHGFLearningConfig& learning,
    bool reset_state = true
)
{
    const std::size_t step_size = batch;

    if (
        steps == 0u ||
        batch == 0u ||
        observations.size() !=
            static_cast<std::size_t>(steps) * step_size ||
        !std::all_of(
            observations.begin(),
            observations.end(),
            [](float value) noexcept
            {
                return
                    value == TestConst::BINARY_ZERO ||
                    value == TestConst::BINARY_ONE;
            }
        )
    )
    {
        return false;
    }

    if (reset_state && !model.ResetGHGFState())
        return false;

    std::vector<float> predictions(step_size);

    for (std::uint32_t t = 0u; t < steps; ++t)
    {
        const std::size_t begin =
            static_cast<std::size_t>(t) * step_size;

        if (!model.PredictModelNONVectorized(
            batch,
            predictions
        ) || !ValidPredictions(predictions))
        {
            return false;
        }

        if (!model.TrainModelNONVectorized(
            batch,
            observations.subspan(begin, step_size),
            learning
        ))
        {
            return false;
        }
    }

    return true;
}

inline bool FitOneParameter(
    GHGFModelConstructor& model,
    std::span<const float> observations,
    std::uint32_t steps,
    std::uint32_t batch,
    std::uint32_t slot,
    std::uint32_t index,
    float lower,
    float upper,
    std::uint32_t passes
)
{
    const std::array<GM::GHGFParameterRange, TestConst::ONE_PARAMETER_COUNT>
        parameter{{
        {
            slot,
            index,
            lower,
            upper
        }
    }};

    const std::optional<double> loss =
        model.FitGHGFParameters(
            observations,
            steps,
            batch,
            parameter,
            passes
        );

    return
        loss.has_value() &&
        std::isfinite(loss.value());
}

// ----------------------------------------------------------------------------
// TEST A - zero-rate identity
// Predict + Train(zero rates) must be identical to Predict + Update.
// ----------------------------------------------------------------------------

inline bool TestA_ZeroRateIdentity()
{
    Banner("TEST A - ZERO-RATE TRAINING IDENTITY");

    static constexpr std::uint32_t BATCH = 4u;
    static constexpr std::uint32_t STEPS = 64u;
    static constexpr std::uint32_t BLOCK_LENGTH = 3u;
    static constexpr float H_COUPLING = 0.75f;
    static constexpr double MAX_IDENTITY_DIFFERENCE = 1.0e-7;

    ScopedModel update_model{};
    ScopedModel train_model{};

    if (
        !ConstructValueObservationModel(
            update_model.Model,
            BATCH,
            H_COUPLING
        ) ||
        !ConstructValueObservationModel(
            train_model.Model,
            BATCH,
            H_COUPLING
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    const std::vector<float> data =
        BlockDataset(
            STEPS,
            BATCH,
            BLOCK_LENGTH
        );

    const GHGFLearningConfig zero =
        ZeroLearning();

    std::vector<float> update_prediction(BATCH);
    std::vector<float> train_prediction(BATCH);

    double maximum_difference = 0.0;

    for (std::uint32_t t = 0u; t < STEPS; ++t)
    {
        const std::size_t begin =
            static_cast<std::size_t>(t) * BATCH;

        if (
            !update_model.Model.PredictModelNONVectorized(
                BATCH,
                update_prediction
            ) ||
            !train_model.Model.PredictModelNONVectorized(
                BATCH,
                train_prediction
            )
        )
        {
            Report("paired prediction", false);
            return false;
        }

        maximum_difference =
            std::max(
                maximum_difference,
                MaxAbsoluteDifference(
                    update_prediction,
                    train_prediction
                )
            );

        const std::span<const float> observation{
            data.data() + begin,
            BATCH
        };

        if (
            !update_model.Model.UpdateModelNONVectorized(
                BATCH,
                observation
            ) ||
            !train_model.Model.TrainModelNONVectorized(
                BATCH,
                observation,
                zero
            )
        )
        {
            Report("paired update/train", false);
            return false;
        }
    }

    if (
        !update_model.Model.PredictModelNONVectorized(
            BATCH,
            update_prediction
        ) ||
        !train_model.Model.PredictModelNONVectorized(
            BATCH,
            train_prediction
        )
    )
    {
        Report("final prediction", false);
        return false;
    }

    maximum_difference =
        std::max(
            maximum_difference,
            MaxAbsoluteDifference(
                update_prediction,
                train_prediction
            )
        );

    const bool ok =
        std::isfinite(maximum_difference) &&
        maximum_difference <= MAX_IDENTITY_DIFFERENCE;

    std::cout
        << std::fixed
        << std::setprecision(TestConst::LONG_PRECISION)
        << "  maximum prediction difference                 "
        << maximum_difference << '\n';

    Report("zero-rate Train == Update", ok);
    return ok;
}

// ----------------------------------------------------------------------------
// TEST B - observation bias / tonic drift
// With H coupling zero, an 80/20 Bernoulli target can only be represented by
// the observation node's TONIC_DRIFT (logit bias).
// ----------------------------------------------------------------------------

inline bool TestB_ObservationBias()
{
    Banner("TEST B - OBSERVATION BIAS / TONIC-DRIFT LEARNING");

    static constexpr std::uint32_t BATCH = 10u;
    static constexpr std::uint32_t TRAIN_STEPS = 160u;
    static constexpr std::uint32_t TEST_STEPS = 40u;
    static constexpr std::uint32_t ONES = 8u;
    static constexpr float DISABLED_H_COUPLING = 0.0f;
    static constexpr float DRIFT_LEARNING_RATE = 0.25f;
    static constexpr float GRADIENT_CLIP = 2.0f;
    static constexpr double TARGET_PROBABILITY = 0.8;
    static constexpr double PROBABILITY_TOLERANCE = 0.05;
    static constexpr double MINIMUM_LOSS_IMPROVEMENT = 0.05;

    ScopedModel model{};

    if (!ConstructValueObservationModel(
        model.Model,
        BATCH,
        DISABLED_H_COUPLING
    ))
    {
        Report("model construction", false);
        return false;
    }

    const std::vector<float> training =
        FixedBernoulliBatchDataset(
            TRAIN_STEPS,
            BATCH,
            ONES
        );

    const std::vector<float> testing =
        FixedBernoulliBatchDataset(
            TEST_STEPS,
            BATCH,
            ONES
        );

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.DriftLearningRate = DRIFT_LEARNING_RATE;
    learning.GradientClip = GRADIENT_CLIP;

    if (!TrainSequence(
        model.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        learning,
        true
    ))
    {
        Report("internal drift training", false);
        return false;
    }

    const Evaluation held_out = Evaluate(
        model.Model,
        testing,
        TEST_STEPS,
        BATCH,
        true
    );

    if (!held_out.Valid)
    {
        Report("held-out evaluation", false);
        return false;
    }

    double mean_probability = 0.0;

    for (float value : held_out.Predictions)
        mean_probability += value;

    mean_probability /=
        static_cast<double>(held_out.Predictions.size());

    const double initial_loss = -std::log(TestConst::HALF);

    const bool probability_ok =
        std::abs(mean_probability - TARGET_PROBABILITY) <
            PROBABILITY_TOLERANCE;

    const bool loss_ok =
        held_out.Loss < initial_loss - MINIMUM_LOSS_IMPROVEMENT;

    std::cout
        << std::fixed
        << std::setprecision(TestConst::SHORT_PRECISION)
        << "  learned mean probability                     "
        << mean_probability << '\n'
        << "  target probability                           "
        << TARGET_PROBABILITY << '\n'
        << "  initial 0.5 loss                             "
        << initial_loss << '\n'
        << "  learned loss                                 "
        << held_out.Loss << '\n';

    Report("bias approaches empirical Bernoulli rate", probability_ok);
    Report("bias learning lowers binary loss", loss_ok);

    return probability_ok && loss_ok;
}

// ----------------------------------------------------------------------------
// TEST C - H coupling quality
// Only H coupling is learnable. A persistent all-one stream makes the latent
// parent useful, so the learned decoder coupling should improve cold-start loss.
// ----------------------------------------------------------------------------

inline bool TestC_HCouplingLearning()
{
    Banner("TEST C - H-COUPLING INTERNAL LEARNING");

    static constexpr std::uint32_t BATCH = 1u;
    static constexpr std::uint32_t TRAIN_STEPS = 128u;
    static constexpr std::uint32_t TEST_STEPS = 16u;
    static constexpr float OBSERVATION = 1.0f;
    static constexpr float INITIAL_H_COUPLING = 0.20f;
    static constexpr float H_LEARNING_RATE = 0.05f;
    static constexpr float GRADIENT_CLIP = 5.0f;
    static constexpr double MINIMUM_IMPROVEMENT = 1.0e-5;

    const std::vector<float> training =
        ConstantDataset(
            TRAIN_STEPS,
            BATCH,
            OBSERVATION
        );

    const std::vector<float> testing =
        ConstantDataset(
            TEST_STEPS,
            BATCH,
            OBSERVATION
        );

    ScopedModel baseline{};
    ScopedModel learned{};

    if (
        !ConstructValueObservationModel(
            baseline.Model,
            BATCH,
            INITIAL_H_COUPLING
        ) ||
        !ConstructValueObservationModel(
            learned.Model,
            BATCH,
            INITIAL_H_COUPLING
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.HCouplingLearningRate = H_LEARNING_RATE;
    learning.GradientClip = GRADIENT_CLIP;

    if (!TrainSequence(
        learned.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        learning,
        true
    ))
    {
        Report("H-coupling training", false);
        return false;
    }

    const Evaluation before =
        Evaluate(
            baseline.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    const Evaluation after =
        Evaluate(
            learned.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    const bool ok =
        before.Valid &&
        after.Valid &&
        after.Loss + MINIMUM_IMPROVEMENT < before.Loss;

    std::cout
        << std::fixed
        << std::setprecision(TestConst::LONG_PRECISION)
        << "  frozen coupling loss                         "
        << before.Loss << '\n'
        << "  internally learned coupling loss             "
        << after.Loss << '\n';

    Report("H coupling lowers cold-start loss", ok);
    return ok;
}

// ----------------------------------------------------------------------------
// TEST D - H coupling finite-difference direction
// The H parameter itself is private. We therefore construct +/- epsilon models
// to obtain the numerical loss direction, then verify that internal learning
// moves the model's predictions toward the better finite-difference side.
// ----------------------------------------------------------------------------

inline bool TestD_HCouplingFiniteDifference()
{
    Banner("TEST D - H-COUPLING FINITE-DIFFERENCE DIRECTION");

    static constexpr std::uint32_t BATCH = 1u;
    static constexpr std::uint32_t TRAIN_STEPS = 48u;
    static constexpr std::uint32_t PROBE_STEPS = 24u;
    static constexpr float CENTER = 0.50f;
    static constexpr float EPSILON = 0.10f;
    static constexpr float OBSERVATION = 1.0f;
    static constexpr float H_LEARNING_RATE = 0.02f;
    static constexpr float GRADIENT_CLIP = 5.0f;
    static constexpr double CENTRAL_DIFFERENCE_DENOMINATOR = 2.0;
    static constexpr double MINIMUM_RESOLVABLE_LOSS_DELTA = 1.0e-8;

    const std::vector<float> training =
        ConstantDataset(
            TRAIN_STEPS,
            BATCH,
            OBSERVATION
        );

    const std::vector<float> probe =
        ConstantDataset(
            PROBE_STEPS,
            BATCH,
            OBSERVATION
        );

    ScopedModel minus_model{};
    ScopedModel center_model{};
    ScopedModel plus_model{};
    ScopedModel local_model{};

    if (
        !ConstructValueObservationModel(
            minus_model.Model,
            BATCH,
            CENTER - EPSILON
        ) ||
        !ConstructValueObservationModel(
            center_model.Model,
            BATCH,
            CENTER
        ) ||
        !ConstructValueObservationModel(
            plus_model.Model,
            BATCH,
            CENTER + EPSILON
        ) ||
        !ConstructValueObservationModel(
            local_model.Model,
            BATCH,
            CENTER
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    const Evaluation minus =
        Evaluate(
            minus_model.Model,
            probe,
            PROBE_STEPS,
            BATCH,
            true
        );

    const Evaluation center =
        Evaluate(
            center_model.Model,
            probe,
            PROBE_STEPS,
            BATCH,
            true
        );

    const Evaluation plus =
        Evaluate(
            plus_model.Model,
            probe,
            PROBE_STEPS,
            BATCH,
            true
        );

    if (!minus.Valid || !center.Valid || !plus.Valid)
    {
        Report("finite-difference evaluation", false);
        return false;
    }

    const double numerical_loss_gradient =
        (plus.Loss - minus.Loss) /
        (CENTRAL_DIFFERENCE_DENOMINATOR * static_cast<double>(EPSILON));

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.HCouplingLearningRate = H_LEARNING_RATE;
    learning.GradientClip = GRADIENT_CLIP;

    if (!TrainSequence(
        local_model.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        learning,
        true
    ))
    {
        Report("internal H training", false);
        return false;
    }

    const Evaluation local =
        Evaluate(
            local_model.Model,
            probe,
            PROBE_STEPS,
            BATCH,
            true
        );

    if (!local.Valid)
    {
        Report("local evaluation", false);
        return false;
    }

    const Evaluation& preferred =
        plus.Loss < minus.Loss
            ? plus
            : minus;

    const double agreement =
        DirectionDot(
            center.Predictions,
            local.Predictions,
            preferred.Predictions
        );

    const bool finite_difference_has_direction =
        std::abs(plus.Loss - minus.Loss) >
            MINIMUM_RESOLVABLE_LOSS_DELTA;

    const bool direction_ok =
        agreement > DELTA_EPS;

    const bool quality_ok =
        local.Loss < center.Loss;

    std::cout
        << std::fixed
        << std::setprecision(TestConst::LONG_PRECISION)
        << "  loss(k-eps)                                  "
        << minus.Loss << '\n'
        << "  loss(k)                                      "
        << center.Loss << '\n'
        << "  loss(k+eps)                                  "
        << plus.Loss << '\n'
        << "  numerical dLoss/dk                           "
        << numerical_loss_gradient << '\n'
        << "  local-vs-preferred prediction dot            "
        << agreement << '\n'
        << "  local learned loss                           "
        << local.Loss << '\n';

    Report(
        "finite difference has a resolvable direction",
        finite_difference_has_direction
    );

    Report(
        "local H update moves toward better side",
        direction_ok
    );

    Report(
        "local H update lowers probe loss",
        quality_ok
    );

    return
        finite_difference_has_direction &&
        direction_ok &&
        quality_ok;
}

// ----------------------------------------------------------------------------
// TEST E - drift/bias direction against global fitter
// FitGHGFParameters() is the slow global oracle. Only observation TONIC_DRIFT
// is fitted/trained, so both mechanisms should move predictions in the same
// functional direction.
// ----------------------------------------------------------------------------

inline bool TestE_DriftOracleAgreement()
{
    Banner("TEST E - DRIFT / BIAS DIRECTION VS GLOBAL FIT ORACLE");

    static constexpr std::uint32_t BATCH = 1u;
    static constexpr std::uint32_t TRAIN_STEPS = 48u;
    static constexpr std::uint32_t TEST_STEPS = 16u;
    static constexpr float OBSERVATION = 1.0f;
    static constexpr float DISABLED_H_COUPLING = 0.0f;
    static constexpr float DRIFT_LEARNING_RATE = 0.10f;
    static constexpr float GRADIENT_CLIP = 2.0f;
    static constexpr float SEARCH_LOWER = -2.0f;
    static constexpr float SEARCH_UPPER = 2.0f;
    static constexpr std::uint32_t SEARCH_PASSES = 2u;

    const std::vector<float> training =
        ConstantDataset(
            TRAIN_STEPS,
            BATCH,
            OBSERVATION
        );

    const std::vector<float> testing =
        ConstantDataset(
            TEST_STEPS,
            BATCH,
            OBSERVATION
        );

    ScopedModel baseline{};
    ScopedModel local{};
    ScopedModel oracle{};

    if (
        !ConstructValueObservationModel(
            baseline.Model,
            BATCH,
            DISABLED_H_COUPLING
        ) ||
        !ConstructValueObservationModel(
            local.Model,
            BATCH,
            DISABLED_H_COUPLING
        ) ||
        !ConstructValueObservationModel(
            oracle.Model,
            BATCH,
            DISABLED_H_COUPLING
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.DriftLearningRate = DRIFT_LEARNING_RATE;
    learning.GradientClip = GRADIENT_CLIP;

    if (!TrainSequence(
        local.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        learning,
        true
    ))
    {
        Report("local drift training", false);
        return false;
    }

    const std::uint32_t drift_index =
        static_cast<std::uint32_t>(
            GM::GHGFErrorValueIndexing::TONIC_DRIFT
        );

    if (!FitOneParameter(
        oracle.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        TestConst::TWO_NODE_OBSERVATION_SLOT,
        drift_index,
        SEARCH_LOWER,
        SEARCH_UPPER,
        SEARCH_PASSES
    ))
    {
        Report("global drift fit", false);
        return false;
    }

    const Evaluation base_eval =
        Evaluate(
            baseline.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    const Evaluation local_eval =
        Evaluate(
            local.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    const Evaluation oracle_eval =
        Evaluate(
            oracle.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    if (
        !base_eval.Valid ||
        !local_eval.Valid ||
        !oracle_eval.Valid
    )
    {
        Report("evaluation", false);
        return false;
    }

    const double agreement =
        DirectionDot(
            base_eval.Predictions,
            local_eval.Predictions,
            oracle_eval.Predictions
        );

    const bool local_improved =
        local_eval.Loss < base_eval.Loss;

    const bool oracle_improved =
        oracle_eval.Loss < base_eval.Loss;

    const bool direction_ok =
        agreement > DELTA_EPS;

    std::cout
        << std::fixed
        << std::setprecision(TestConst::LONG_PRECISION)
        << "  baseline loss                                "
        << base_eval.Loss << '\n'
        << "  local drift loss                             "
        << local_eval.Loss << '\n'
        << "  global-fit drift loss                        "
        << oracle_eval.Loss << '\n'
        << "  local/global prediction-direction dot        "
        << agreement << '\n';

    Report("local drift lowers loss", local_improved);
    Report("global drift fit lowers loss", oracle_improved);
    Report("local drift agrees with global-fit direction", direction_ok);

    return
        local_improved &&
        oracle_improved &&
        direction_ok;
}

// ----------------------------------------------------------------------------
// TEST F - temporal / AUTO_CONNECTION
// Deterministic alternation punishes persistence. AUTO_CONNECTION begins at 1,
// so both local learning and global fitting should learn a less persistent
// temporal model.
// ----------------------------------------------------------------------------

inline bool TestF_TemporalParameter()
{
    Banner("TEST F - TEMPORAL / AUTO-CONNECTION LEARNING");

    static constexpr std::uint32_t BATCH = 1u;
    static constexpr std::uint32_t TRAIN_STEPS = 256u;
    static constexpr std::uint32_t TEST_STEPS = 64u;
    static constexpr float INITIAL_H_COUPLING = 1.0f;
    static constexpr float TEMPORAL_LEARNING_RATE = 0.05f;
    static constexpr float GRADIENT_CLIP = 2.0f;
    static constexpr float SEARCH_LOWER = 0.0f;
    static constexpr float SEARCH_UPPER = 1.0f;
    static constexpr std::uint32_t SEARCH_PASSES = 4u;
    static constexpr double MINIMUM_IMPROVEMENT = 1.0e-6;

    const std::vector<float> training =
        AlternatingDataset(
            TRAIN_STEPS,
            BATCH
        );

    const std::vector<float> testing =
        AlternatingDataset(
            TEST_STEPS,
            BATCH,
            TRAIN_STEPS
        );

    ScopedModel baseline{};
    ScopedModel local{};
    ScopedModel oracle{};

    if (
        !ConstructValueObservationModel(
            baseline.Model,
            BATCH,
            INITIAL_H_COUPLING
        ) ||
        !ConstructValueObservationModel(
            local.Model,
            BATCH,
            INITIAL_H_COUPLING
        ) ||
        !ConstructValueObservationModel(
            oracle.Model,
            BATCH,
            INITIAL_H_COUPLING
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.AutoConnectionLearningRate = TEMPORAL_LEARNING_RATE;
    learning.GradientClip = GRADIENT_CLIP;

    if (!TrainSequence(
        local.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        learning,
        true
    ))
    {
        Report("local temporal training", false);
        return false;
    }

    const std::uint32_t temporal_index =
        static_cast<std::uint32_t>(
            GM::GHGFErrorValueIndexing::AUTO_CONNECTION
        );

    if (!FitOneParameter(
        oracle.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        TestConst::VALUE_SLOT,
        temporal_index,
        SEARCH_LOWER,
        SEARCH_UPPER,
        SEARCH_PASSES
    ))
    {
        Report("global temporal fit", false);
        return false;
    }

    const Evaluation base_eval =
        Evaluate(
            baseline.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    const Evaluation local_eval =
        Evaluate(
            local.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    const Evaluation oracle_eval =
        Evaluate(
            oracle.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    if (
        !base_eval.Valid ||
        !local_eval.Valid ||
        !oracle_eval.Valid
    )
    {
        Report("evaluation", false);
        return false;
    }

    const double agreement =
        DirectionDot(
            base_eval.Predictions,
            local_eval.Predictions,
            oracle_eval.Predictions
        );

    const bool local_improved =
        local_eval.Loss + MINIMUM_IMPROVEMENT < base_eval.Loss;

    const bool oracle_non_regression =
        oracle_eval.Loss <= base_eval.Loss + LOSS_EPS;

    const bool direction_ok =
        agreement > DELTA_EPS;

    std::cout
        << std::fixed
        << std::setprecision(TestConst::LONG_PRECISION)
        << "  fixed lambda loss                             "
        << base_eval.Loss << '\n'
        << "  locally learned lambda loss                   "
        << local_eval.Loss << '\n'
        << "  globally fitted lambda loss                   "
        << oracle_eval.Loss << '\n'
        << "  local/global prediction-direction dot         "
        << agreement << '\n';

    Report("temporal learning lowers alternating loss", local_improved);
    Report("global temporal fit is non-regressive", oracle_non_regression);
    Report("local temporal direction agrees with fitter", direction_ok);

    return
        local_improved &&
        oracle_non_regression &&
        direction_ok;
}

// ----------------------------------------------------------------------------
// TEST G - tonic volatility
// Block switches create repeated surprise at regime boundaries. The only local
// parameter enabled is TONIC_VOLATILE on the VALUE node.
// ----------------------------------------------------------------------------

inline bool TestG_TonicVolatility()
{
    Banner("TEST G - TONIC-VOLATILITY INTERNAL LEARNING");

    static constexpr std::uint32_t BATCH = 1u;
    static constexpr std::uint32_t TRAIN_STEPS = 256u;
    static constexpr std::uint32_t TEST_STEPS = 96u;
    static constexpr std::uint32_t BLOCK = 8u;
    static constexpr float INITIAL_H_COUPLING = 1.0f;
    static constexpr float VOLATILITY_LEARNING_RATE = 0.01f;
    static constexpr float GRADIENT_CLIP = 2.0f;
    static constexpr float MIN_TONIC_LOG_VOLATILITY = -10.0f;
    static constexpr float MAX_TONIC_LOG_VOLATILITY = 2.0f;
    static constexpr float SEARCH_LOWER = -8.0f;
    static constexpr float SEARCH_UPPER = 0.0f;
    static constexpr std::uint32_t SEARCH_PASSES = 4u;
    static constexpr double MINIMUM_MEAN_SQUARED_EFFECT = 1.0e-8;
    static constexpr double MINIMUM_LOSS_IMPROVEMENT = 1.0e-6;

    const std::vector<float> training =
        BlockDataset(
            TRAIN_STEPS,
            BATCH,
            BLOCK
        );

    const std::vector<float> testing =
        BlockDataset(
            TEST_STEPS,
            BATCH,
            BLOCK,
            TRAIN_STEPS
        );

    ScopedModel baseline{};
    ScopedModel local{};
    ScopedModel oracle{};

    if (
        !ConstructValueObservationModel(
            baseline.Model,
            BATCH,
            INITIAL_H_COUPLING
        ) ||
        !ConstructValueObservationModel(
            local.Model,
            BATCH,
            INITIAL_H_COUPLING
        ) ||
        !ConstructValueObservationModel(
            oracle.Model,
            BATCH,
            INITIAL_H_COUPLING
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.VolatilityLearningRate = VOLATILITY_LEARNING_RATE;
    learning.GradientClip = GRADIENT_CLIP;
    learning.MinTonicLogVolatility = MIN_TONIC_LOG_VOLATILITY;
    learning.MaxTonicLogVolatility = MAX_TONIC_LOG_VOLATILITY;

    if (!TrainSequence(
        local.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        learning,
        true
    ))
    {
        Report("local volatility training", false);
        return false;
    }

    const std::uint32_t volatility_index =
        static_cast<std::uint32_t>(
            GM::GHGFErrorValueIndexing::TONIC_VOLATILE
        );

    if (!FitOneParameter(
        oracle.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        TestConst::VALUE_SLOT,
        volatility_index,
        SEARCH_LOWER,
        SEARCH_UPPER,
        SEARCH_PASSES
    ))
    {
        Report("global volatility fit", false);
        return false;
    }

    const Evaluation base_eval =
        Evaluate(
            baseline.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    const Evaluation local_eval =
        Evaluate(
            local.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    const Evaluation oracle_eval =
        Evaluate(
            oracle.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    if (
        !base_eval.Valid ||
        !local_eval.Valid ||
        !oracle_eval.Valid
    )
    {
        Report("evaluation", false);
        return false;
    }

    const double local_change =
        DeltaSquared(
            base_eval.Predictions,
            local_eval.Predictions
        );

    const double agreement =
        DirectionDot(
            base_eval.Predictions,
            local_eval.Predictions,
            oracle_eval.Predictions
        );

    const double mean_squared_effect =
        local_change / static_cast<double>(local_eval.Predictions.size());

    const bool local_changed_meaningfully =
        mean_squared_effect > MINIMUM_MEAN_SQUARED_EFFECT;

    const bool local_improved =
        local_eval.Loss + MINIMUM_LOSS_IMPROVEMENT < base_eval.Loss;

    const bool oracle_non_regression =
        oracle_eval.Loss <= base_eval.Loss + LOSS_EPS;

    // Diagnostic only: FitGHGFParameters() optimizes complete replay loss,
    // while the internal update is a local prospective rule. They need not
    // produce the same long-horizon direction on every dataset.
    const bool oracle_direction_same =
        agreement > DELTA_EPS;

    std::cout
        << std::fixed
        << std::setprecision(TestConst::LONG_PRECISION)
        << "  fixed volatility loss                         "
        << base_eval.Loss << '\n'
        << "  locally learned volatility loss               "
        << local_eval.Loss << '\n'
        << "  globally fitted volatility loss               "
        << oracle_eval.Loss << '\n'
        << std::scientific
        << "  local prediction delta^2                      "
        << local_change << '\n'
        << "  local mean squared prediction effect          "
        << mean_squared_effect << '\n'
        << "  local/global prediction-direction dot         "
        << agreement << '\n'
        << std::fixed;

    Report("tonic volatility has meaningful held-out effect", local_changed_meaningfully);
    Report("local volatility lowers held-out loss", local_improved);
    Report("global volatility fit is non-regressive", oracle_non_regression);

    std::cout
        << "  global-fit direction diagnostic               "
        << (oracle_direction_same ? "SAME" : "DIFFERENT")
        << " (not a hard failure)\n";

    return
        local_changed_meaningfully &&
        local_improved &&
        oracle_non_regression;
}

// ----------------------------------------------------------------------------
// TEST H - V coupling
// A volatility parent modulates a VALUE node that predicts the Bernoulli child.
// Only the V coupling is learnable. The global fitter is again used as oracle.
// ----------------------------------------------------------------------------

inline bool TestH_VCoupling()
{
    Banner("TEST H - V-COUPLING INTERNAL LEARNING");

    static constexpr std::uint32_t BATCH = 1u;
    static constexpr std::uint32_t TRAIN_STEPS = 384u;
    static constexpr std::uint32_t TEST_STEPS = 128u;
    static constexpr std::uint32_t BLOCK = 8u;
    static constexpr float INITIAL_V = 0.25f;
    static constexpr float INITIAL_H = 1.0f;
    static constexpr float V_LEARNING_RATE = 0.01f;
    static constexpr float GRADIENT_CLIP = 2.0f;
    static constexpr float SEARCH_LOWER = -2.0f;
    static constexpr float SEARCH_UPPER = 2.0f;
    static constexpr std::uint32_t SEARCH_PASSES = 4u;
    static constexpr std::uint8_t FIRST_RELATION_ORDINAL = 0u;
    static constexpr double MINIMUM_MEAN_SQUARED_EFFECT = 1.0e-8;
    static constexpr double MINIMUM_LOSS_IMPROVEMENT = 1.0e-6;

    const std::vector<float> training =
        BlockDataset(
            TRAIN_STEPS,
            BATCH,
            BLOCK
        );

    const std::vector<float> testing =
        BlockDataset(
            TEST_STEPS,
            BATCH,
            BLOCK,
            TRAIN_STEPS
        );

    ScopedModel baseline{};
    ScopedModel local{};
    ScopedModel oracle{};

    if (
        !ConstructVolatileValueObservationModel(
            baseline.Model,
            BATCH,
            INITIAL_V,
            INITIAL_H
        ) ||
        !ConstructVolatileValueObservationModel(
            local.Model,
            BATCH,
            INITIAL_V,
            INITIAL_H
        ) ||
        !ConstructVolatileValueObservationModel(
            oracle.Model,
            BATCH,
            INITIAL_V,
            INITIAL_H
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.VCouplingLearningRate = V_LEARNING_RATE;
    learning.GradientClip = GRADIENT_CLIP;

    if (!TrainSequence(
        local.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        learning,
        true
    ))
    {
        Report("local V-coupling training", false);
        return false;
    }

    const std::uint32_t v_index =
        GM::CouplingIndex(
            FabricSegments::VOLATILE_PARENT_EDGE_TABLE_V,
            FIRST_RELATION_ORDINAL,
            PARENT_CAPACITY
        );

    if (!FitOneParameter(
        oracle.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        TestConst::SECOND_VALUE_SLOT,
        v_index,
        SEARCH_LOWER,
        SEARCH_UPPER,
        SEARCH_PASSES
    ))
    {
        Report("global V-coupling fit", false);
        return false;
    }

    const Evaluation base_eval =
        Evaluate(
            baseline.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    const Evaluation local_eval =
        Evaluate(
            local.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    const Evaluation oracle_eval =
        Evaluate(
            oracle.Model,
            testing,
            TEST_STEPS,
            BATCH,
            true
        );

    if (
        !base_eval.Valid ||
        !local_eval.Valid ||
        !oracle_eval.Valid
    )
    {
        Report("evaluation", false);
        return false;
    }

    const double local_change =
        DeltaSquared(
            base_eval.Predictions,
            local_eval.Predictions
        );

    const double agreement =
        DirectionDot(
            base_eval.Predictions,
            local_eval.Predictions,
            oracle_eval.Predictions
        );

    const double mean_squared_effect =
        local_change / static_cast<double>(local_eval.Predictions.size());

    const bool local_changed_meaningfully =
        mean_squared_effect > MINIMUM_MEAN_SQUARED_EFFECT;

    const bool local_improved =
        local_eval.Loss + MINIMUM_LOSS_IMPROVEMENT < base_eval.Loss;

    const bool oracle_non_regression =
        oracle_eval.Loss <= base_eval.Loss + LOSS_EPS;

    // Diagnostic only for the same reason as Test G: the global coordinate
    // fitter and the local prospective V update optimize through different
    // computational paths.
    const bool oracle_direction_same =
        agreement > DELTA_EPS;

    std::cout
        << std::fixed
        << std::setprecision(TestConst::LONG_PRECISION)
        << "  fixed V-coupling loss                         "
        << base_eval.Loss << '\n'
        << "  locally learned V-coupling loss               "
        << local_eval.Loss << '\n'
        << "  globally fitted V-coupling loss               "
        << oracle_eval.Loss << '\n'
        << std::scientific
        << "  local prediction delta^2                      "
        << local_change << '\n'
        << "  local mean squared prediction effect          "
        << mean_squared_effect << '\n'
        << "  local/global prediction-direction dot         "
        << agreement << '\n'
        << std::fixed;

    Report("V coupling has meaningful held-out effect", local_changed_meaningfully);
    Report("local V learning lowers held-out loss", local_improved);
    Report("global V fit is non-regressive", oracle_non_regression);

    std::cout
        << "  global-fit direction diagnostic               "
        << (oracle_direction_same ? "SAME" : "DIFFERENT")
        << " (not a hard failure)\n";

    return
        local_changed_meaningfully &&
        local_improved &&
        oracle_non_regression;
}

// ----------------------------------------------------------------------------
// TEST I (current) - online concurrent predictive-structure learning
//
// Three claims are kept separate: optimistic mutation semantics, a
// validation-only structural decision on one live model, and concurrent
// progress/integrity. This is predictive structure selection, not causal
// discovery. Only one worker writes numeric state; the public API does not
// promise that multiple numeric writers can share a model.
// ----------------------------------------------------------------------------

inline double BICStylePerSamplePenalty(
    std::uint32_t edge_count,
    std::size_t sample_count
) noexcept
{
    static constexpr double HALF_BIC_SCALE = TestConst::HALF;

    if (edge_count == 0u || sample_count < 2u)
        return 0.0;

    return
        HALF_BIC_SCALE *
        static_cast<double>(edge_count) *
        std::log(static_cast<double>(sample_count)) /
        static_cast<double>(sample_count);
}

inline GM::GHGFStructureMutation MakeHiddenEdgeMutation(
    EdgeBuilder::StructureOperation operation,
    std::uint32_t expected_sequence,
    float coupling
) noexcept
{
    GM::GHGFStructureMutation mutation{};
    mutation.Operation = operation;
    mutation.Child = TestConst::SECOND_VALUE_SLOT;
    mutation.OldParent = TestConst::VALUE_SLOT;
    mutation.NewParent = TestConst::VALUE_SLOT;
    mutation.Edge = FabricSegments::VALUE_PARENT_EDGE_TABLE_H;
    mutation.Coupling = coupling;
    mutation.ExpectedRowSequence = expected_sequence;
    return mutation;
}

struct ConcurrentStructureMeasurement final
{
    std::uint64_t NumericSuccesses = 0u;
    std::uint64_t NumericFailedAttempts = 0u;
    std::uint64_t SnapshotSuccesses = 0u;
    std::uint64_t SnapshotRetries = 0u;
    std::uint64_t SnapshotRejected = 0u;
    std::uint64_t SnapshotInvariantFailures = 0u;
    std::uint64_t MutationSuccesses = 0u;
    std::uint64_t MutationRetries = 0u;
    std::uint64_t MutationStale = 0u;
    std::uint64_t MutationRejected = 0u;
    std::uint64_t MutationPublicationFailures = 0u;
    double Seconds = 0.0;
};

inline ConcurrentStructureMeasurement MeasureConcurrentStructureLearning(
    GHGFModelConstructor& model,
    std::span<const float> observations,
    std::uint32_t batch,
    float hidden_coupling
)
{
    using Concurrent = GM::GHGFConcurrentOperation;

    static constexpr std::uint32_t NUMERIC_ATTEMPTS = 32768u;
    static constexpr std::uint32_t MUTATION_ATTEMPTS = 32768u;
    static constexpr std::uint32_t SNAPSHOT_ATTEMPTS = 65536u;
    static constexpr std::uint32_t CONCURRENT_MAX_TRIES = 32u;
    static constexpr std::uint32_t SEQUENCE_WRITE_BIT_MASK = 1u;
    static constexpr std::ptrdiff_t WORKER_COUNT = 3;
    static constexpr std::ptrdiff_t MAIN_PARTICIPANT_COUNT = 1;
    static constexpr std::ptrdiff_t BARRIER_PARTICIPANTS =
        WORKER_COUNT + MAIN_PARTICIPANT_COUNT;

    ConcurrentStructureMeasurement measurement{};
    if (batch == 0u || observations.size() < batch)
        return measurement;

    const GHGFLearningConfig zero_learning = ZeroLearning();
    const std::size_t observation_step_count = observations.size() / batch;
    std::vector<float> prediction(batch);
    std::chrono::steady_clock::time_point begin{};

    std::barrier start_gate(
        BARRIER_PARTICIPANTS,
        [&begin]() noexcept { begin = std::chrono::steady_clock::now(); }
    );

    std::thread numeric_worker([&]()
    {
        start_gate.arrive_and_wait();
        for (std::uint32_t attempt = 0u; attempt < NUMERIC_ATTEMPTS; ++attempt)
        {
            const std::size_t step = attempt % observation_step_count;
            const std::span<const float> observation = observations.subspan(
                step * batch,
                batch
            );
            const bool predicted =
                model.PredictModelNONVectorized(batch, prediction);
            const bool valid_prediction =
                predicted && ValidPredictions(prediction);
            const bool trained =
                valid_prediction &&
                model.TrainModelNONVectorized(
                    batch,
                    observation,
                    zero_learning
                );

            if (trained)
                ++measurement.NumericSuccesses;
            else
                ++measurement.NumericFailedAttempts;
        }
    });

    std::thread mutation_worker([&]()
    {
        start_gate.arrive_and_wait();
        for (std::uint32_t attempt = 0u; attempt < MUTATION_ATTEMPTS; ++attempt)
        {
            GM::GHGFStructureSnapshot snapshot{};
            const Concurrent read = model.ReadStructureSnapshotConcurrently(
                TestConst::SECOND_VALUE_SLOT,
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                snapshot,
                CONCURRENT_MAX_TRIES
            );

            if (read == Concurrent::RETRY)
            {
                ++measurement.MutationRetries;
                continue;
            }
            if (read != Concurrent::SUCCESS || !snapshot.IsValid)
            {
                ++measurement.MutationRejected;
                continue;
            }

            const EdgeBuilder::StructureOperation operation =
                snapshot.ParentMask == 0u
                    ? EdgeBuilder::StructureOperation::ADD_PARENT
                    : EdgeBuilder::StructureOperation::REMOVE_PARENT;
            const GM::GHGFStructureMutation mutation = MakeHiddenEdgeMutation(
                operation,
                snapshot.RowSequence,
                hidden_coupling
            );
            const GM::GHGFStructureMutationResult result =
                model.TryApplyGHGFStructureMutation(
                    mutation,
                    CONCURRENT_MAX_TRIES
                );

            switch (result.Result)
            {
            case Concurrent::SUCCESS:
                ++measurement.MutationSuccesses;
                if (
                    result.PublishedRowSequence == UINT32_MAX ||
                    result.PublishedOrdinal == UINT8_MAX
                )
                {
                    ++measurement.MutationPublicationFailures;
                }
                break;
            case Concurrent::RETRY:
                ++measurement.MutationRetries;
                break;
            case Concurrent::STALE:
                ++measurement.MutationStale;
                break;
            case Concurrent::REJECTED:
                ++measurement.MutationRejected;
                break;
            }
        }
    });

    std::thread snapshot_worker([&]()
    {
        std::uint32_t last_sequence = 0u;
        start_gate.arrive_and_wait();
        for (std::uint32_t attempt = 0u; attempt < SNAPSHOT_ATTEMPTS; ++attempt)
        {
            GM::GHGFStructureSnapshot snapshot{};
            const Concurrent read = model.ReadStructureSnapshotConcurrently(
                TestConst::SECOND_VALUE_SLOT,
                FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
                snapshot,
                CONCURRENT_MAX_TRIES
            );

            if (read == Concurrent::SUCCESS)
            {
                ++measurement.SnapshotSuccesses;
                const bool valid_mask =
                    std::popcount(snapshot.ParentMask) <=
                    TestConst::SINGLE_CANDIDATE_PARENT_COUNT;
                const bool published_sequence =
                    (snapshot.RowSequence & SEQUENCE_WRITE_BIT_MASK) == 0u;
                const bool monotonic_sequence =
                    snapshot.RowSequence >= last_sequence;
                if (
                    !snapshot.IsValid ||
                    snapshot.Child != TestConst::SECOND_VALUE_SLOT ||
                    snapshot.Edge != FabricSegments::VALUE_PARENT_EDGE_TABLE_H ||
                    !valid_mask ||
                    !published_sequence ||
                    !monotonic_sequence
                )
                {
                    ++measurement.SnapshotInvariantFailures;
                }
                last_sequence = snapshot.RowSequence;
            }
            else if (read == Concurrent::RETRY)
            {
                ++measurement.SnapshotRetries;
            }
            else
            {
                ++measurement.SnapshotRejected;
            }
        }
    });

    start_gate.arrive_and_wait();
    numeric_worker.join();
    mutation_worker.join();
    snapshot_worker.join();

    const auto end = std::chrono::steady_clock::now();
    measurement.Seconds = std::chrono::duration<double>(end - begin).count();
    return measurement;
}

inline bool TestI_OnlineConcurrentStructuralLearning()
{
    Banner("TEST I - ONLINE CONCURRENT PREDICTIVE-STRUCTURE LEARNING");

    static constexpr std::uint32_t BATCH = 1u;
    static constexpr std::uint32_t TRAIN_STEPS = 503u;
    static constexpr std::uint32_t VALIDATION_STEPS = 251u;
    static constexpr std::uint32_t TEST_STEPS = 257u;
    static constexpr std::uint32_t TOTAL_STEPS =
        TRAIN_STEPS + VALIDATION_STEPS + TEST_STEPS;
    static constexpr std::uint32_t BLOCK_LENGTH = 16u;
    static constexpr std::uint32_t SHALLOW_EDGE_COUNT = 1u;
    static constexpr std::uint32_t HIERARCHICAL_EDGE_COUNT = 2u;
    static constexpr std::uint32_t API_MAX_TRIES = 128u;
    static constexpr float CANDIDATE_HIDDEN_COUPLING = 0.5f;
    static constexpr float INITIAL_OBSERVATION_COUPLING = 1.0f;
    static constexpr float DRIFT_LEARNING_RATE = 0.02f;
    static constexpr float TEMPORAL_LEARNING_RATE = 0.005f;
    static constexpr float GRADIENT_CLIP = 5.0f;
    static constexpr double MINIMUM_VALIDATION_MARGIN = 1.0e-3;
    static constexpr double MINIMUM_TEST_IMPROVEMENT = 1.0e-3;

    const std::vector<float> data = BlockDataset(
        TOTAL_STEPS,
        BATCH,
        BLOCK_LENGTH
    );
    const std::span<const float> all{data};
    const std::span<const float> training =
        all.first(static_cast<std::size_t>(TRAIN_STEPS) * BATCH);
    const std::span<const float> validation = all.subspan(
        static_cast<std::size_t>(TRAIN_STEPS) * BATCH,
        static_cast<std::size_t>(VALIDATION_STEPS) * BATCH
    );
    const std::span<const float> testing = all.subspan(
        static_cast<std::size_t>(TRAIN_STEPS + VALIDATION_STEPS) * BATCH,
        static_cast<std::size_t>(TEST_STEPS) * BATCH
    );

    GHGFLearningConfig learning = ZeroLearning();
    learning.DriftLearningRate = DRIFT_LEARNING_RATE;
    learning.AutoConnectionLearningRate = TEMPORAL_LEARNING_RATE;
    learning.GradientClip = GRADIENT_CLIP;

    ScopedModel model{};
    if (
        !ConstructPredictiveStructure(
            model.Model,
            BATCH,
            PredictiveStructure::SHALLOW,
            CANDIDATE_HIDDEN_COUPLING,
            INITIAL_OBSERVATION_COUPLING,
            true
        ) ||
        !TrainSequence(
            model.Model,
            training,
            TRAIN_STEPS,
            BATCH,
            learning,
            true
        )
    )
    {
        Report("structural model construction and training", false);
        return false;
    }

    const Evaluation shallow_validation = Evaluate(
        model.Model, validation, VALIDATION_STEPS, BATCH, true);
    const double shallow_score = shallow_validation.Loss +
        BICStylePerSamplePenalty(SHALLOW_EDGE_COUNT, validation.size());

    GM::GHGFStructureSnapshot initial_snapshot{};
    const GM::GHGFConcurrentOperation initial_read =
        model.Model.ReadStructureSnapshotConcurrently(
            TestConst::SECOND_VALUE_SLOT,
            FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
            initial_snapshot,
            API_MAX_TRIES
        );
    const GM::GHGFStructureMutation add_mutation = MakeHiddenEdgeMutation(
        EdgeBuilder::StructureOperation::ADD_PARENT,
        initial_snapshot.RowSequence,
        CANDIDATE_HIDDEN_COUPLING
    );
    const GM::GHGFStructureMutationResult add_result =
        model.Model.TryApplyGHGFStructureMutation(add_mutation, API_MAX_TRIES);

    GM::GHGFStructureSnapshot added_snapshot{};
    const GM::GHGFConcurrentOperation added_read =
        model.Model.ReadStructureSnapshotConcurrently(
            TestConst::SECOND_VALUE_SLOT,
            FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
            added_snapshot,
            API_MAX_TRIES
        );
    const GM::GHGFStructureMutationResult stale_replay =
        model.Model.TryApplyGHGFStructureMutation(add_mutation, API_MAX_TRIES);

    const Evaluation hierarchical_validation = Evaluate(
        model.Model, validation, VALIDATION_STEPS, BATCH, true);
    const double hierarchical_score = hierarchical_validation.Loss +
        BICStylePerSamplePenalty(HIERARCHICAL_EDGE_COUNT, validation.size());
    const double validation_margin = shallow_score - hierarchical_score;
    const bool selected_hierarchical =
        shallow_validation.Valid &&
        hierarchical_validation.Valid &&
        validation_margin > MINIMUM_VALIDATION_MARGIN;

    // The test split is read only after the validation decision is fixed.
    const Evaluation hierarchical_test = Evaluate(
        model.Model, testing, TEST_STEPS, BATCH, true);

    const GM::GHGFStructureMutation remove_mutation = MakeHiddenEdgeMutation(
        EdgeBuilder::StructureOperation::REMOVE_PARENT,
        added_snapshot.RowSequence,
        CANDIDATE_HIDDEN_COUPLING
    );
    const GM::GHGFStructureMutationResult remove_result =
        model.Model.TryApplyGHGFStructureMutation(remove_mutation, API_MAX_TRIES);
    const Evaluation shallow_test = Evaluate(
        model.Model, testing, TEST_STEPS, BATCH, true);

    GM::GHGFStructureSnapshot removed_snapshot{};
    const GM::GHGFConcurrentOperation removed_read =
        model.Model.ReadStructureSnapshotConcurrently(
            TestConst::SECOND_VALUE_SLOT,
            FabricSegments::VALUE_PARENT_EDGE_TABLE_H,
            removed_snapshot,
            API_MAX_TRIES
        );
    const GM::GHGFStructureMutation restore_mutation = MakeHiddenEdgeMutation(
        EdgeBuilder::StructureOperation::ADD_PARENT,
        removed_snapshot.RowSequence,
        CANDIDATE_HIDDEN_COUPLING
    );
    const GM::GHGFStructureMutationResult restore_result =
        model.Model.TryApplyGHGFStructureMutation(
            restore_mutation,
            API_MAX_TRIES
        );

    const bool transaction_contract =
        initial_read == GM::GHGFConcurrentOperation::SUCCESS &&
        initial_snapshot.IsValid &&
        initial_snapshot.ParentMask == 0u &&
        add_result.Result == GM::GHGFConcurrentOperation::SUCCESS &&
        add_result.PublishedRowSequence != UINT32_MAX &&
        add_result.PublishedOrdinal != UINT8_MAX &&
        added_read == GM::GHGFConcurrentOperation::SUCCESS &&
        added_snapshot.IsValid &&
        std::popcount(added_snapshot.ParentMask) ==
            TestConst::SINGLE_CANDIDATE_PARENT_COUNT &&
        added_snapshot.RowSequence == add_result.PublishedRowSequence &&
        stale_replay.Result == GM::GHGFConcurrentOperation::STALE &&
        remove_result.Result == GM::GHGFConcurrentOperation::SUCCESS &&
        removed_read == GM::GHGFConcurrentOperation::SUCCESS &&
        removed_snapshot.IsValid &&
        removed_snapshot.ParentMask == 0u &&
        restore_result.Result == GM::GHGFConcurrentOperation::SUCCESS;
    const bool held_out_improvement =
        hierarchical_test.Valid &&
        shallow_test.Valid &&
        hierarchical_test.Loss + MINIMUM_TEST_IMPROVEMENT < shallow_test.Loss;

    const ConcurrentStructureMeasurement concurrent =
        MeasureConcurrentStructureLearning(
            model.Model,
            testing,
            BATCH,
            CANDIDATE_HIDDEN_COUPLING
        );

    std::vector<float> recovery_prediction(BATCH);
    const bool recovered_after_contention =
        model.Model.ResetGHGFState() &&
        model.Model.PredictModelNONVectorized(BATCH, recovery_prediction) &&
        ValidPredictions(recovery_prediction) &&
        model.Model.UpdateModelNONVectorized(BATCH, testing.first(BATCH));
    const bool concurrent_integrity =
        concurrent.Seconds > 0.0 &&
        concurrent.NumericSuccesses > 0u &&
        concurrent.SnapshotSuccesses > 0u &&
        concurrent.MutationSuccesses > 0u &&
        concurrent.SnapshotInvariantFailures == 0u &&
        concurrent.SnapshotRejected == 0u &&
        concurrent.MutationRejected == 0u &&
        concurrent.MutationPublicationFailures == 0u &&
        recovered_after_contention;

    const double numeric_attempts_per_second =
        static_cast<double>(
            concurrent.NumericSuccesses + concurrent.NumericFailedAttempts) /
        concurrent.Seconds;
    const double mutation_attempts_per_second =
        static_cast<double>(
            concurrent.MutationSuccesses +
            concurrent.MutationRetries +
            concurrent.MutationStale +
            concurrent.MutationRejected) /
        concurrent.Seconds;

    std::cout
        << std::fixed
        << std::setprecision(TestConst::LONG_PRECISION)
        << "  shallow validation loss                       "
        << shallow_validation.Loss << '\n'
        << "  shallow penalized score                       "
        << shallow_score << '\n'
        << "  hierarchical validation loss                  "
        << hierarchical_validation.Loss << '\n'
        << "  hierarchical penalized score                  "
        << hierarchical_score << '\n'
        << "  validation decision margin                    "
        << validation_margin << '\n'
        << "  hierarchical held-out test loss               "
        << hierarchical_test.Loss << '\n'
        << "  shallow held-out test loss                    "
        << shallow_test.Loss << '\n'
        << "  concurrent elapsed seconds                    "
        << concurrent.Seconds << '\n'
        << "  numeric successes / failed attempts           "
        << concurrent.NumericSuccesses << " / "
        << concurrent.NumericFailedAttempts << '\n'
        << "  snapshot success / retry / rejected           "
        << concurrent.SnapshotSuccesses << " / "
        << concurrent.SnapshotRetries << " / "
        << concurrent.SnapshotRejected << '\n'
        << "  mutation success / retry / stale / rejected   "
        << concurrent.MutationSuccesses << " / "
        << concurrent.MutationRetries << " / "
        << concurrent.MutationStale << " / "
        << concurrent.MutationRejected << '\n'
        << "  numeric attempts per second                   "
        << numeric_attempts_per_second << '\n'
        << "  mutation attempts per second                  "
        << mutation_attempts_per_second << '\n';

    Report("optimistic mutation contract is coherent", transaction_contract);
    Report("validation accepts live hierarchical edge", selected_hierarchical);
    Report("accepted edge improves untouched test data", held_out_improvement);
    Report("same-model concurrent stress preserves invariants", concurrent_integrity);

    std::cout
        << "  NOTE: failed numeric attempts are reported, not relabeled as retries;\n"
        << "        the numeric API returns bool and does not expose retry status.\n";

    return
        transaction_contract &&
        selected_hierarchical &&
        held_out_improvement &&
        concurrent_integrity;
}

// ----------------------------------------------------------------------------
// TEST J - timing and independent-model parallel throughput
//
// Reported ns/step values are amortized wall-clock throughput measurements, not
// single-call latency. Each parallel worker owns a separate model; this measures
// independent-model parallelism and does not claim same-model thread safety.
// ----------------------------------------------------------------------------

using BenchmarkClock = std::chrono::steady_clock;
static_assert(BenchmarkClock::is_steady);

struct WorkloadMeasurement final
{
    bool Valid = false;
    double Seconds = 0.0;
    std::uint64_t Steps = 0u;
    std::uint64_t Samples = 0u;
    double Checksum = 0.0;
};

inline bool RunModelWorkload(
    GHGFModelConstructor& model,
    std::span<const float> observations,
    std::uint32_t steps,
    std::uint32_t batch,
    const GHGFLearningConfig& learning,
    bool train,
    double& checksum
)
{
    if (
        steps == 0u ||
        batch == 0u ||
        observations.size() != static_cast<std::size_t>(steps) * batch
    )
    {
        return false;
    }

    std::vector<float> prediction(batch);
    checksum = 0.0;

    for (std::uint32_t time = 0u; time < steps; ++time)
    {
        const std::size_t begin = static_cast<std::size_t>(time) * batch;
        const std::span<const float> observation =
            observations.subspan(begin, batch);

        if (!model.PredictModelNONVectorized(batch, prediction))
            return false;

        checksum += static_cast<double>(prediction.front());
        checksum += static_cast<double>(prediction.back());

        const bool advanced = train
            ? model.TrainModelNONVectorized(batch, observation, learning)
            : model.UpdateModelNONVectorized(batch, observation);

        if (!advanced)
            return false;
    }

    return std::isfinite(checksum) && checksum > 0.0;
}

inline WorkloadMeasurement MeasureOneSequentialWorkload(
    std::span<const float> observations,
    std::uint32_t steps,
    std::uint32_t batch,
    const GHGFLearningConfig& learning,
    bool train
)
{
    static constexpr float BENCHMARK_H_COUPLING = 1.0f;

    ScopedModel model{};
    WorkloadMeasurement measurement{};

    if (!ConstructValueObservationModel(
        model.Model,
        batch,
        BENCHMARK_H_COUPLING
    ))
    {
        return measurement;
    }

    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto begin = BenchmarkClock::now();

    const bool ran = RunModelWorkload(
        model.Model,
        observations,
        steps,
        batch,
        learning,
        train,
        measurement.Checksum
    );

    const auto end = BenchmarkClock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);

    measurement.Seconds = std::chrono::duration<double>(end - begin).count();
    measurement.Steps = steps;
    measurement.Samples = static_cast<std::uint64_t>(steps) * batch;
    measurement.Valid =
        ran &&
        measurement.Seconds > 0.0 &&
        std::isfinite(measurement.Seconds);

    return measurement;
}

inline WorkloadMeasurement MedianSequentialWorkload(
    std::span<const float> observations,
    std::uint32_t steps,
    std::uint32_t batch,
    const GHGFLearningConfig& learning,
    bool train,
    std::uint32_t repetitions
)
{
    WorkloadMeasurement invalid{};
    if (repetitions == 0u)
        return invalid;

    std::vector<WorkloadMeasurement> samples;
    samples.reserve(repetitions);

    for (std::uint32_t repetition = 0u; repetition < repetitions; ++repetition)
    {
        WorkloadMeasurement sample = MeasureOneSequentialWorkload(
            observations,
            steps,
            batch,
            learning,
            train
        );

        if (!sample.Valid)
            return invalid;

        samples.push_back(sample);
    }

    std::sort(
        samples.begin(),
        samples.end(),
        [](const WorkloadMeasurement& left, const WorkloadMeasurement& right)
        {
            return left.Seconds < right.Seconds;
        }
    );

    return samples[samples.size() / 2u];
}

inline WorkloadMeasurement MeasureConstruction(
    std::uint32_t constructions,
    std::uint32_t batch
)
{
    static constexpr float CONSTRUCTION_H_COUPLING = 1.0f;

    WorkloadMeasurement measurement{};
    if (constructions == 0u || batch == 0u)
        return measurement;

    const auto begin = BenchmarkClock::now();

    for (std::uint32_t index = 0u; index < constructions; ++index)
    {
        ScopedModel model{};
        if (!ConstructValueObservationModel(
            model.Model,
            batch,
            CONSTRUCTION_H_COUPLING
        ))
        {
            return {};
        }

        measurement.Checksum += model.Model.IsFabricActive() ? 1.0 : 0.0;
    }

    const auto end = BenchmarkClock::now();
    measurement.Seconds = std::chrono::duration<double>(end - begin).count();
    measurement.Steps = constructions;
    measurement.Samples = constructions;
    measurement.Valid =
        measurement.Seconds > 0.0 &&
        measurement.Checksum == static_cast<double>(constructions);
    return measurement;
}

inline WorkloadMeasurement MeasureParallelTrial(
    std::uint32_t thread_count,
    std::span<const float> observations,
    std::uint32_t steps,
    std::uint32_t batch,
    const GHGFLearningConfig& learning
)
{
    static constexpr float PARALLEL_H_COUPLING = 1.0f;

    WorkloadMeasurement measurement{};
    if (thread_count == 0u)
        return measurement;

    std::vector<unsigned char> worker_ok(thread_count, 0u);
    std::vector<double> worker_checksum(thread_count, 0.0);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    BenchmarkClock::time_point begin{};
    BenchmarkClock::time_point end{};

    std::barrier start_gate(
        static_cast<std::ptrdiff_t>(thread_count + 1u),
        [&begin]() noexcept { begin = BenchmarkClock::now(); }
    );

    std::barrier finish_gate(
        static_cast<std::ptrdiff_t>(thread_count + 1u),
        [&end]() noexcept { end = BenchmarkClock::now(); }
    );

    for (std::uint32_t worker = 0u; worker < thread_count; ++worker)
    {
        workers.emplace_back([&, worker]()
        {
            ScopedModel model{};
            const bool constructed = ConstructValueObservationModel(
                model.Model,
                batch,
                PARALLEL_H_COUPLING
            );

            start_gate.arrive_and_wait();

            const bool ran =
                constructed &&
                RunModelWorkload(
                    model.Model,
                    observations,
                    steps,
                    batch,
                    learning,
                    true,
                    worker_checksum[worker]
                );

            worker_ok[worker] = ran ? 1u : 0u;
            finish_gate.arrive_and_wait();
        });
    }

    start_gate.arrive_and_wait();
    finish_gate.arrive_and_wait();

    for (std::thread& worker : workers)
        worker.join();

    const bool all_workers_ok = std::all_of(
        worker_ok.begin(),
        worker_ok.end(),
        [](unsigned char value) noexcept { return value != 0u; }
    );

    for (double checksum : worker_checksum)
        measurement.Checksum += checksum;

    measurement.Seconds = std::chrono::duration<double>(end - begin).count();
    measurement.Steps =
        static_cast<std::uint64_t>(steps) * thread_count;
    measurement.Samples = measurement.Steps * batch;
    measurement.Valid =
        all_workers_ok &&
        measurement.Seconds > 0.0 &&
        std::isfinite(measurement.Seconds) &&
        std::isfinite(measurement.Checksum) &&
        measurement.Checksum > 0.0;

    return measurement;
}

inline WorkloadMeasurement MedianParallelWorkload(
    std::uint32_t thread_count,
    std::span<const float> observations,
    std::uint32_t steps,
    std::uint32_t batch,
    const GHGFLearningConfig& learning,
    std::uint32_t repetitions
)
{
    WorkloadMeasurement invalid{};
    if (repetitions == 0u)
        return invalid;

    std::vector<WorkloadMeasurement> samples;
    samples.reserve(repetitions);

    for (std::uint32_t repetition = 0u; repetition < repetitions; ++repetition)
    {
        WorkloadMeasurement sample = MeasureParallelTrial(
            thread_count,
            observations,
            steps,
            batch,
            learning
        );

        if (!sample.Valid)
            return invalid;

        samples.push_back(sample);
    }

    std::sort(
        samples.begin(),
        samples.end(),
        [](const WorkloadMeasurement& left, const WorkloadMeasurement& right)
        {
            return left.Seconds < right.Seconds;
        }
    );

    return samples[samples.size() / 2u];
}

inline double NanosecondsPerStep(const WorkloadMeasurement& value) noexcept
{
    static constexpr double NANOSECONDS_PER_SECOND = 1.0e9;
    return value.Valid
        ? value.Seconds * NANOSECONDS_PER_SECOND /
            static_cast<double>(value.Steps)
        : std::numeric_limits<double>::infinity();
}

inline double MillionSamplesPerSecond(const WorkloadMeasurement& value) noexcept
{
    static constexpr double SAMPLES_PER_MILLION = 1.0e6;
    return value.Valid
        ? static_cast<double>(value.Samples) /
            value.Seconds /
            SAMPLES_PER_MILLION
        : 0.0;
}

inline void ReportBuildEnvironment()
{
    std::cout << "  compiler                                       ";

#if defined(__clang__)
    std::cout << "Clang " << __clang_version__;
#elif defined(_MSC_VER)
    std::cout << "MSVC " << _MSC_VER;
#elif defined(__GNUC__)
    std::cout
        << "GCC "
        << __GNUC__ << '.' << __GNUC_MINOR__ << '.' << __GNUC_PATCHLEVEL__;
#else
    std::cout << "unknown";
#endif

    std::cout << "\n  __cplusplus                                    "
              << __cplusplus << '\n';
}

inline bool TestJ_TimingAndIndependentParallelism()
{
    Banner("TEST J - TIMING AND INDEPENDENT-MODEL PARALLELISM");

    static constexpr std::uint32_t BATCH = 16u;
    static constexpr std::uint32_t STEPS = 32768u;
    static constexpr std::uint32_t BLOCK_LENGTH = 16u;
    static constexpr std::uint32_t SEQUENTIAL_REPETITIONS = 9u;
    static constexpr std::uint32_t PARALLEL_REPETITIONS = 7u;
    static constexpr std::uint32_t CONSTRUCTION_REPETITIONS = 8192u;
    static constexpr std::uint32_t UNKNOWN_HARDWARE_THREADS_FALLBACK = 1u;
    static constexpr std::size_t REQUESTED_THREAD_COUNT = 4u;
    static constexpr std::array<std::uint32_t, REQUESTED_THREAD_COUNT>
        REQUESTED_THREADS{
        1u, 2u, 4u, 8u
    };
    static constexpr float H_LEARNING_RATE = 0.01f;
    static constexpr float DRIFT_LEARNING_RATE = 0.01f;
    static constexpr float GRADIENT_CLIP = 5.0f;

    const std::vector<float> observations = BlockDataset(
        STEPS,
        BATCH,
        BLOCK_LENGTH
    );

    GHGFLearningConfig learning = ZeroLearning();
    learning.HCouplingLearningRate = H_LEARNING_RATE;
    learning.DriftLearningRate = DRIFT_LEARNING_RATE;
    learning.GradientClip = GRADIENT_CLIP;

    // One untimed pass warms code and allocator paths without contaminating
    // any recorded model state.
    const WorkloadMeasurement warmup = MeasureOneSequentialWorkload(
        observations,
        STEPS,
        BATCH,
        learning,
        false
    );

    const WorkloadMeasurement construction = MeasureConstruction(
        CONSTRUCTION_REPETITIONS,
        BATCH
    );

    const WorkloadMeasurement update = MedianSequentialWorkload(
        observations,
        STEPS,
        BATCH,
        learning,
        false,
        SEQUENTIAL_REPETITIONS
    );

    const WorkloadMeasurement train = MedianSequentialWorkload(
        observations,
        STEPS,
        BATCH,
        learning,
        true,
        SEQUENTIAL_REPETITIONS
    );

    bool parallel_valid = true;
    const std::uint32_t reported_hardware_threads =
        std::thread::hardware_concurrency();
    const std::uint32_t usable_hardware_threads =
        reported_hardware_threads == 0u
            ? UNKNOWN_HARDWARE_THREADS_FALLBACK
            : reported_hardware_threads;

    std::optional<double> one_thread_throughput{};

    ReportBuildEnvironment();

    std::cout
        << std::fixed
        << std::setprecision(TestConst::SHORT_PRECISION)
        << "  timed steps per worker                         "
        << STEPS << '\n'
        << "  batch samples per time-step                    "
        << BATCH << '\n'
        << "  sequential median repetitions                 "
        << SEQUENTIAL_REPETITIONS << '\n'
        << "  parallel median repetitions                   "
        << PARALLEL_REPETITIONS << '\n'
        << "  hardware_concurrency reported                 "
        << reported_hardware_threads << '\n'
        << "  construction amortized ns/model               "
        << NanosecondsPerStep(construction) << '\n'
        << "  predict+update median ns/time-step             "
        << NanosecondsPerStep(update) << '\n'
        << "  predict+update median M samples/s              "
        << MillionSamplesPerSecond(update) << '\n'
        << "  predict+train median ns/time-step              "
        << NanosecondsPerStep(train) << '\n'
        << "  predict+train median M samples/s               "
        << MillionSamplesPerSecond(train) << '\n'
        << "\n  Independent models; predict+train aggregate throughput:\n";

    for (const std::uint32_t thread_count : REQUESTED_THREADS)
    {
        if (thread_count > usable_hardware_threads)
            continue;

        const WorkloadMeasurement parallel = MedianParallelWorkload(
            thread_count,
            observations,
            STEPS,
            BATCH,
            learning,
            PARALLEL_REPETITIONS
        );

        parallel_valid = parallel_valid && parallel.Valid;
        const double throughput = MillionSamplesPerSecond(parallel);

        if (thread_count == REQUESTED_THREADS.front())
            one_thread_throughput = throughput;

        const double speedup =
            one_thread_throughput.has_value() &&
            one_thread_throughput.value() > 0.0
                ? throughput / one_thread_throughput.value()
                : 0.0;

        const double efficiency = speedup / thread_count;

        std::cout
            << "    threads="
            << std::setw(TestConst::THREAD_COLUMN_WIDTH) << thread_count
            << "  M samples/s="
            << std::setw(TestConst::THROUGHPUT_COLUMN_WIDTH) << throughput
            << "  speedup="
            << std::setw(TestConst::SPEEDUP_COLUMN_WIDTH) << speedup
            << "  efficiency=" << efficiency << '\n';
    }

    const bool sequential_valid =
        warmup.Valid &&
        construction.Valid &&
        update.Valid &&
        train.Valid;

    Report("sequential timing samples are valid", sequential_valid);
    Report("parallel workers completed valid workloads", parallel_valid);

    std::cout
        << "  NOTE: parallel rows use separate model instances; no same-model\n"
        << "        thread-safety or latency claim is implied.\n";

    return sequential_valid && parallel_valid;
}

struct TimedTest final
{
    bool Passed = false;
    double Milliseconds = 0.0;
};

template<class TestFunction>
inline TimedTest RunTimedTest(TestFunction&& test)
{
    const auto begin = BenchmarkClock::now();
    const bool passed = std::forward<TestFunction>(test)();
    const auto end = BenchmarkClock::now();

    return {
        passed,
        std::chrono::duration<double, std::milli>(end - begin).count()
    };
}

// ----------------------------------------------------------------------------
// Run all A-J
// ----------------------------------------------------------------------------

inline int RunAll()
{
    std::cout
        << "\n================================================================================\n"
        << "GHGF PAPER VALIDATION TEST KIT - TESTS A-J\n"
        << "================================================================================\n";

    const TimedTest a = RunTimedTest(TestA_ZeroRateIdentity);
    const TimedTest b = RunTimedTest(TestB_ObservationBias);
    const TimedTest c = RunTimedTest(TestC_HCouplingLearning);
    const TimedTest d = RunTimedTest(TestD_HCouplingFiniteDifference);
    const TimedTest e = RunTimedTest(TestE_DriftOracleAgreement);
    const TimedTest f = RunTimedTest(TestF_TemporalParameter);
    const TimedTest g = RunTimedTest(TestG_TonicVolatility);
    const TimedTest h = RunTimedTest(TestH_VCoupling);
    const TimedTest i = RunTimedTest(TestI_OnlineConcurrentStructuralLearning);
    const TimedTest j = RunTimedTest(TestJ_TimingAndIndependentParallelism);

    const int failures =
        static_cast<int>(!a.Passed) +
        static_cast<int>(!b.Passed) +
        static_cast<int>(!c.Passed) +
        static_cast<int>(!d.Passed) +
        static_cast<int>(!e.Passed) +
        static_cast<int>(!f.Passed) +
        static_cast<int>(!g.Passed) +
        static_cast<int>(!h.Passed) +
        static_cast<int>(!i.Passed) +
        static_cast<int>(!j.Passed);

    const auto SummaryLine___ = [](
        const char* label,
        const TimedTest& result
    )
    {
        std::cout
            << "  " << std::left << std::setw(TestConst::REPORT_WIDTH)
            << label
            << (result.Passed ? "PASS" : "FAIL")
            << "  " << std::right << std::fixed
            << std::setprecision(TestConst::SUMMARY_PRECISION)
            << result.Milliseconds << " ms\n";
    };

    std::cout
        << "\n================================================================================\n"
        << "GHGF PAPER VALIDATION TEST SUMMARY\n"
        << "================================================================================\n";

    SummaryLine___("Test A - zero-rate identity", a);
    SummaryLine___("Test B - observation bias", b);
    SummaryLine___("Test C - H coupling learning", c);
    SummaryLine___("Test D - H finite-difference direction", d);
    SummaryLine___("Test E - drift/global-oracle agreement", e);
    SummaryLine___("Test F - temporal parameter", f);
    SummaryLine___("Test G - tonic-volatility efficacy", g);
    SummaryLine___("Test H - V-coupling efficacy", h);
    SummaryLine___("Test I - online concurrent structural learning", i);
    SummaryLine___("Test J - timing/independent parallelism", j);

    std::cout
        << "\n  failures: " << failures << '\n'
        << "================================================================================\n";

    return failures == 0 ? 0 : 1;
}

} // namespace GHGFTestKit
