#pragma once

// ============================================================================
// SuperNova public umbrella
// ============================================================================
//
// Stable public entry point for the SuperNova C++ architecture.
//
// Prefer:
//     #include <SuperNova>
//
// or:
//     #include <SuperNova.hpp>
//
// Internal implementation/orchestrator headers remain reachable through their
// own paths, but are intentionally not promoted as part of this public facade.
// ============================================================================

// APC / schema-backed node substrate.
#include "AdaptivePackedCellContainer/AdaptivePackedCellContainer.hpp"

// Fabric / slab / DAG public layer.
#include "NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"

// GHGF public model layer.
#include "Models/GHGF/GHGFLayer.hpp"
#include "Models/GHGF/GHGFNode.hpp"
#include "Models/GHGF/GHGFModelOfAPC.hpp"

// Transitional public namespace facade.
//
// Existing implementation code remains in BidirectionalInMemGraph, so this is
// source-compatible with the current architecture while allowing public code to
// use:
//
//     SuperNova::GHGFModelConstructor
//     SuperNova::GHGFNode
//     SuperNova::AdaptivePackedCellContainer
//     SuperNova::APCFinilizer
//
// Do not open `namespace SuperNova { ... }` elsewhere while this alias exists.
namespace SuperNova = BidirectionalInMemGraph;
