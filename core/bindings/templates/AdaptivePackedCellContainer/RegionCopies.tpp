#pragma once
#include <pybind11/numpy.h>

#include "Bindings.hpp"
#include <SuperNova>


namespace SuperNovaBindings
{
    // Route by the stored schema type. CHAR uses a one-byte integer array;
    // float16 copies use the native SYCL half representation when available.
    template<class Copy>
    decltype(auto) VisitRegionType(BidirectionalInMemGraph::SchemaDefinition::DataTypeOfMacroColumn dtype, Copy copy)
    {
        using DType = BidirectionalInMemGraph::SchemaDefinition::DataTypeOfMacroColumn;
        switch (dtype)
        {
        case DType::UINT8_T:  return copy.template operator()<uint8_t>();
        case DType::UINT16_T: return copy.template operator()<uint16_t>();
        case DType::UINT32_T: return copy.template operator()<uint32_t>();
        case DType::UINT64_T: return copy.template operator()<uint64_t>();
        case DType::INT8_T:   return copy.template operator()<int8_t>();
        case DType::INT16_T:  return copy.template operator()<int16_t>();
        case DType::INT32_T:  return copy.template operator()<int32_t>();
        case DType::INT64_T:  return copy.template operator()<int64_t>();
        case DType::FLOAT32_T: return copy.template operator()<float>();
        case DType::FLOAT64_T: return copy.template operator()<double>();
        case DType::CHAR: return copy.template operator()<char>();
#ifdef SN_ACPP_ENABLED
        case DType::FLOAT16_T: return copy.template operator()<sycl::half>();
#endif
        default: throw py::value_error("This build cannot copy that region dtype.");
        }
    }

    template<class T>
    py::dtype RegionDType()
    {
#ifdef SN_ACPP_ENABLED
        if constexpr (std::is_same_v<T, sycl::half>)
            return py::dtype("float16");
        else
#endif
        if constexpr (std::is_same_v<T, char>)
            return py::dtype::of<uint8_t>();
        else
            return py::dtype::of<T>();
    }

    template<class T>
    py::array ReadRegion(BidirectionalInMemGraph::AdaptivePackedCellContainer& apc,
                         BidirectionalInMemGraph::MacroColumnOfAPC column)
    {
        const auto bridge = apc.BuildAViewOverRegion<T>(column);
        const auto cells = bridge.RawSpan();
        if (!cells)
            throw std::runtime_error("Expected a live private or immutable region with the requested dtype.");

        py::array result(RegionDType<T>(), {static_cast<py::ssize_t>(bridge.Height()),
                                           static_cast<py::ssize_t>(bridge.ActiveWidth())});
        std::memcpy(result.mutable_data(), cells->data(), cells->size_bytes());
        return result;
    }

    template<class T>
    void WriteRegion(BidirectionalInMemGraph::AdaptivePackedCellContainer& apc,
                     BidirectionalInMemGraph::MacroColumnOfAPC column, const py::array& values,
                     BidirectionalInMemGraph::FabricExecutionGate& gate)
    {
        auto bridge = apc.BuildAViewOverRegion<T>(column);
        auto cells = bridge.RawSpan();
        if (!cells)
            throw std::runtime_error("Expected a live private region with the requested dtype.");

        if (!values.dtype().equal(RegionDType<T>()) || !(values.flags() & py::array::c_style) ||
            values.ndim() != 2 || values.shape(0) != bridge.Height() ||
            values.shape(1) != bridge.ActiveWidth())
            throw py::value_error("Expected a C-contiguous array with the exact region dtype and shape.");

        std::memcpy(cells->data(), values.data(), cells->size_bytes());
        gate.NotifyHostChange();
    }
}
