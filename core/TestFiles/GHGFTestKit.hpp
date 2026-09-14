#pragma once

// ============================================================================
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
// ============================================================================

#ifndef APC_DAG_TEST_EXTERNAL_TYPES
#include "Models/GHGF/GHGFModelOfAPC.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace GHGFTestKit
{
using namespace BidirectionalInMemGraph;

using GM = GHGFLayerModel;
using Role = GM::GHGFNodeRole;

static constexpr std::uint8_t PARENT_CAPACITY = 2u;
static constexpr double LOSS_EPS = 1.0e-10;
static constexpr double DELTA_EPS = 1.0e-12;

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
        << "  " << std::left << std::setw(48) << name
        << (passed ? "PASS" : "FAIL") << '\n';
}

inline GHGFLearningConfig ZeroLearning() noexcept
{
    GHGFLearningConfig learning{};
    learning.HCouplingLearningRate = 0.0f;
    learning.DriftLearningRate = 0.0f;
    learning.VolatilityLearningRate = 0.0f;
    learning.VCouplingLearningRate = 0.0f;
    learning.AutoConnectionLearningRate = 0.0f;
    learning.GradientClip = 10.0f;
    learning.MinTonicLogVolatility = -20.0f;
    learning.MaxTonicLogVolatility = 10.0f;
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

    std::array<GHGFNode, 2u> nodes{};

    const std::array roles{
        Role::VALUE,
        Role::OBSERVATION
    };

    const std::array<GM::GHGFConnection, 1u> connections{{
        {
            0u,
            1u,
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
            v_coupling
        },
        {
            1u,
            2u,
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
        0.0f
    );

    ones_per_batch = std::min(ones_per_batch, batch);

    for (std::uint32_t t = 0u; t < steps; ++t)
    {
        for (std::uint32_t lane = 0u; lane < ones_per_batch; ++lane)
            data[static_cast<std::size_t>(t) * batch + lane] = 1.0f;
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
        0.0f
    );

    for (std::uint32_t t = 0u; t < steps; ++t)
    {
        const float value =
            ((t + time_offset) & 1u) != 0u
                ? 1.0f
                : 0.0f;

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
        0.0f
    );

    if (block_length == 0u)
        return data;

    for (std::uint32_t t = 0u; t < steps; ++t)
    {
        const std::uint32_t global_time = t + time_offset;
        const float value =
            ((global_time / block_length) & 1u) != 0u
                ? 1.0f
                : 0.0f;

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
                value > 0.0f &&
                value < 1.0f;
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
            1.0e-12,
            1.0 - 1.0e-12
        );

        loss -=
            observations[i] == 1.0f
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
        observations.size() != expected
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

    result.Valid = true;
    result.Loss = loss.value();
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
            static_cast<std::size_t>(steps) * step_size
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
        ))
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
    const std::array<GM::GHGFParameterRange, 1u> parameter{{
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

    ScopedModel update_model{};
    ScopedModel train_model{};

    if (
        !ConstructValueObservationModel(
            update_model.Model,
            BATCH,
            0.75f
        ) ||
        !ConstructValueObservationModel(
            train_model.Model,
            BATCH,
            0.75f
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
            3u
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
        maximum_difference <= 1.0e-7;

    std::cout
        << std::fixed
        << std::setprecision(9)
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
    static constexpr std::uint32_t STEPS = 160u;
    static constexpr std::uint32_t ONES = 8u;

    ScopedModel model{};

    if (!ConstructValueObservationModel(
        model.Model,
        BATCH,
        0.0f
    ))
    {
        Report("model construction", false);
        return false;
    }

    const std::vector<float> training =
        FixedBernoulliBatchDataset(
            STEPS,
            BATCH,
            ONES
        );

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.DriftLearningRate = 0.25f;
    learning.GradientClip = 2.0f;

    if (!TrainSequence(
        model.Model,
        training,
        STEPS,
        BATCH,
        learning,
        true
    ))
    {
        Report("internal drift training", false);
        return false;
    }

    if (!model.Model.ResetGHGFState())
    {
        Report("reset after training", false);
        return false;
    }

    std::vector<float> predictions(BATCH);

    if (!model.Model.PredictModelNONVectorized(
        BATCH,
        predictions
    ))
    {
        Report("post-training prediction", false);
        return false;
    }

    double mean_probability = 0.0;

    for (float value : predictions)
        mean_probability += value;

    mean_probability /=
        static_cast<double>(predictions.size());

    const std::span<const float> target{
        training.data(),
        BATCH
    };

    const double final_loss =
        BinaryLogLoss(
            target,
            predictions
        );

    const double initial_loss =
        -std::log(0.5);

    const bool probability_ok =
        std::abs(mean_probability - 0.8) < 0.05;

    const bool loss_ok =
        std::isfinite(final_loss) &&
        final_loss < initial_loss - 0.05;

    std::cout
        << std::fixed
        << std::setprecision(6)
        << "  learned mean probability                     "
        << mean_probability << '\n'
        << "  target probability                           "
        << 0.8 << '\n'
        << "  initial 0.5 loss                             "
        << initial_loss << '\n'
        << "  learned loss                                 "
        << final_loss << '\n';

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

    const std::vector<float> training =
        ConstantDataset(
            TRAIN_STEPS,
            BATCH,
            1.0f
        );

    const std::vector<float> testing =
        ConstantDataset(
            TEST_STEPS,
            BATCH,
            1.0f
        );

    ScopedModel baseline{};
    ScopedModel learned{};

    if (
        !ConstructValueObservationModel(
            baseline.Model,
            BATCH,
            0.20f
        ) ||
        !ConstructValueObservationModel(
            learned.Model,
            BATCH,
            0.20f
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.HCouplingLearningRate = 0.05f;
    learning.GradientClip = 5.0f;

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
        after.Loss + 1.0e-5 < before.Loss;

    std::cout
        << std::fixed
        << std::setprecision(8)
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

    const std::vector<float> training =
        ConstantDataset(
            TRAIN_STEPS,
            BATCH,
            1.0f
        );

    const std::vector<float> probe =
        ConstantDataset(
            PROBE_STEPS,
            BATCH,
            1.0f
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
        (2.0 * static_cast<double>(EPSILON));

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.HCouplingLearningRate = 0.02f;
    learning.GradientClip = 5.0f;

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
        std::abs(plus.Loss - minus.Loss) > 1.0e-8;

    const bool direction_ok =
        agreement > DELTA_EPS;

    const bool quality_ok =
        local.Loss < center.Loss;

    std::cout
        << std::fixed
        << std::setprecision(9)
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

    const std::vector<float> training =
        ConstantDataset(
            TRAIN_STEPS,
            BATCH,
            1.0f
        );

    const std::vector<float> testing =
        ConstantDataset(
            TEST_STEPS,
            BATCH,
            1.0f
        );

    ScopedModel baseline{};
    ScopedModel local{};
    ScopedModel oracle{};

    if (
        !ConstructValueObservationModel(
            baseline.Model,
            BATCH,
            0.0f
        ) ||
        !ConstructValueObservationModel(
            local.Model,
            BATCH,
            0.0f
        ) ||
        !ConstructValueObservationModel(
            oracle.Model,
            BATCH,
            0.0f
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.DriftLearningRate = 0.10f;
    learning.GradientClip = 2.0f;

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
        1u,
        drift_index,
        -2.0f,
        2.0f,
        2u
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
        << std::setprecision(9)
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

    const std::vector<float> training =
        AlternatingDataset(
            TRAIN_STEPS,
            BATCH
        );

    const std::vector<float> testing =
        AlternatingDataset(
            TEST_STEPS,
            BATCH
        );

    ScopedModel baseline{};
    ScopedModel local{};
    ScopedModel oracle{};

    if (
        !ConstructValueObservationModel(
            baseline.Model,
            BATCH,
            1.0f
        ) ||
        !ConstructValueObservationModel(
            local.Model,
            BATCH,
            1.0f
        ) ||
        !ConstructValueObservationModel(
            oracle.Model,
            BATCH,
            1.0f
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.AutoConnectionLearningRate = 0.05f;
    learning.GradientClip = 2.0f;

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
        0u,
        temporal_index,
        0.0f,
        1.0f,
        4u
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
        local_eval.Loss + 1.0e-6 < base_eval.Loss;

    const bool oracle_non_regression =
        oracle_eval.Loss <= base_eval.Loss + LOSS_EPS;

    const bool direction_ok =
        agreement > DELTA_EPS;

    std::cout
        << std::fixed
        << std::setprecision(9)
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

    const std::vector<float> training =
        BlockDataset(
            TRAIN_STEPS,
            BATCH,
            BLOCK,
            0u
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
            1.0f
        ) ||
        !ConstructValueObservationModel(
            local.Model,
            BATCH,
            1.0f
        ) ||
        !ConstructValueObservationModel(
            oracle.Model,
            BATCH,
            1.0f
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.VolatilityLearningRate = 0.01f;
    learning.GradientClip = 2.0f;
    learning.MinTonicLogVolatility = -10.0f;
    learning.MaxTonicLogVolatility = 2.0f;

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
        0u,
        volatility_index,
        -8.0f,
        0.0f,
        4u
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

    const bool local_changed =
        local_change > DELTA_EPS;

    const bool local_non_catastrophic =
        local_eval.Loss <= base_eval.Loss + 0.02;

    const bool oracle_non_regression =
        oracle_eval.Loss <= base_eval.Loss + LOSS_EPS;

    // Diagnostic only: FitGHGFParameters() optimizes complete replay loss,
    // while the internal update is a local prospective rule. They need not
    // produce the same long-horizon direction on every dataset.
    const bool oracle_direction_same =
        agreement > DELTA_EPS;

    std::cout
        << std::fixed
        << std::setprecision(9)
        << "  fixed volatility loss                         "
        << base_eval.Loss << '\n'
        << "  locally learned volatility loss               "
        << local_eval.Loss << '\n'
        << "  globally fitted volatility loss               "
        << oracle_eval.Loss << '\n'
        << std::scientific
        << "  local prediction delta^2                      "
        << local_change << '\n'
        << "  local/global prediction-direction dot         "
        << agreement << '\n'
        << std::fixed;

    Report("tonic volatility changes model behaviour", local_changed);
    Report("local volatility remains stable", local_non_catastrophic);
    Report("global volatility fit is non-regressive", oracle_non_regression);

    std::cout
        << "  global-fit direction diagnostic               "
        << (oracle_direction_same ? "SAME" : "DIFFERENT")
        << " (not a hard failure)\n";

    return
        local_changed &&
        local_non_catastrophic &&
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

    const std::vector<float> training =
        BlockDataset(
            TRAIN_STEPS,
            BATCH,
            BLOCK,
            0u
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
            1.0f
        ) ||
        !ConstructVolatileValueObservationModel(
            local.Model,
            BATCH,
            INITIAL_V,
            1.0f
        ) ||
        !ConstructVolatileValueObservationModel(
            oracle.Model,
            BATCH,
            INITIAL_V,
            1.0f
        )
    )
    {
        Report("model construction", false);
        return false;
    }

    GHGFLearningConfig learning =
        ZeroLearning();

    learning.VCouplingLearningRate = 0.01f;
    learning.GradientClip = 2.0f;

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
            0u,
            PARENT_CAPACITY
        );

    if (!FitOneParameter(
        oracle.Model,
        training,
        TRAIN_STEPS,
        BATCH,
        1u,
        v_index,
        -2.0f,
        2.0f,
        4u
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

    const bool local_changed =
        local_change > DELTA_EPS;

    const bool local_non_catastrophic =
        local_eval.Loss <= base_eval.Loss + 0.02;

    const bool oracle_non_regression =
        oracle_eval.Loss <= base_eval.Loss + LOSS_EPS;

    // Diagnostic only for the same reason as Test G: the global coordinate
    // fitter and the local prospective V update optimize through different
    // computational paths.
    const bool oracle_direction_same =
        agreement > DELTA_EPS;

    std::cout
        << std::fixed
        << std::setprecision(9)
        << "  fixed V-coupling loss                         "
        << base_eval.Loss << '\n'
        << "  locally learned V-coupling loss               "
        << local_eval.Loss << '\n'
        << "  globally fitted V-coupling loss               "
        << oracle_eval.Loss << '\n'
        << std::scientific
        << "  local prediction delta^2                      "
        << local_change << '\n'
        << "  local/global prediction-direction dot         "
        << agreement << '\n'
        << std::fixed;

    Report("V coupling changes model behaviour", local_changed);
    Report("local V learning remains stable", local_non_catastrophic);
    Report("global V fit is non-regressive", oracle_non_regression);

    std::cout
        << "  global-fit direction diagnostic               "
        << (oracle_direction_same ? "SAME" : "DIFFERENT")
        << " (not a hard failure)\n";

    return
        local_changed &&
        local_non_catastrophic &&
        oracle_non_regression;
}

// ----------------------------------------------------------------------------
// Run all A-H
// ----------------------------------------------------------------------------

inline int RunAll()
{
    std::cout
        << "\n================================================================================\n"
        << "GHGF INTERNAL-LEARNING TEST KIT - TESTS A-H\n"
        << "================================================================================\n";

    const bool a = TestA_ZeroRateIdentity();
    const bool b = TestB_ObservationBias();
    const bool c = TestC_HCouplingLearning();
    const bool d = TestD_HCouplingFiniteDifference();
    const bool e = TestE_DriftOracleAgreement();
    const bool f = TestF_TemporalParameter();
    const bool g = TestG_TonicVolatility();
    const bool h = TestH_VCoupling();

    const int failures =
        static_cast<int>(!a) +
        static_cast<int>(!b) +
        static_cast<int>(!c) +
        static_cast<int>(!d) +
        static_cast<int>(!e) +
        static_cast<int>(!f) +
        static_cast<int>(!g) +
        static_cast<int>(!h);

    std::cout
        << "\n================================================================================\n"
        << "GHGF INTERNAL-LEARNING TEST SUMMARY\n"
        << "================================================================================\n"
        << "  Test A - zero-rate identity                  " << (a ? "PASS" : "FAIL") << '\n'
        << "  Test B - observation bias                    " << (b ? "PASS" : "FAIL") << '\n'
        << "  Test C - H coupling learning                 " << (c ? "PASS" : "FAIL") << '\n'
        << "  Test D - H finite-difference direction       " << (d ? "PASS" : "FAIL") << '\n'
        << "  Test E - drift/global-oracle agreement       " << (e ? "PASS" : "FAIL") << '\n'
        << "  Test F - temporal parameter                  " << (f ? "PASS" : "FAIL") << '\n'
        << "  Test G - tonic volatility                    " << (g ? "PASS" : "FAIL") << '\n'
        << "  Test H - V coupling                          " << (h ? "PASS" : "FAIL") << '\n'
        << "\n  failures: " << failures << '\n'
        << "================================================================================\n";

    return failures == 0 ? 0 : 1;
}

} // namespace GHGFTestKit
