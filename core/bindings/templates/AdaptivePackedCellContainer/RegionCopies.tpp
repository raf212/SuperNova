#pragma once
#include <pybind11/numpy.h>
#include "AdaptivePackedCellContainer/APCHandle.hpp"

#include "Bindings.hpp"
#include <SuperNova>


namespace SuperNovaBindings
{
    using SD = BidirectionalInMemGraph::SchemaDefinition;
    using MacroColumn = BidirectionalInMemGraph::MacroColumnOfAPC;
    
    // Route by the stored schema type. CHAR uses a one-byte integer array;
    // float16 copies use the native SYCL half representation when available.
    template<class Visitor>
    decltype(auto) VisitRegionType(SD::DataTypeOfMacroColumn dtype, Visitor visit)
    {
        using DType = SD::DataTypeOfMacroColumn;
        switch (dtype)
        {
        case DType::UINT8_T:  return visit.template operator()<uint8_t>();
        case DType::UINT16_T: return visit.template operator()<uint16_t>();
        case DType::UINT32_T: return visit.template operator()<uint32_t>();
        case DType::UINT64_T: return visit.template operator()<uint64_t>();
        case DType::INT8_T:   return visit.template operator()<int8_t>();
        case DType::INT16_T:  return visit.template operator()<int16_t>();
        case DType::INT32_T:  return visit.template operator()<int32_t>();
        case DType::INT64_T:  return visit.template operator()<int64_t>();
        case DType::FLOAT32_T: return visit.template operator()<float>();
        case DType::FLOAT64_T: return visit.template operator()<double>();
        case DType::CHAR: return visit.template operator()<char>();
#ifdef SN_ACPP_ENABLED
        case DType::FLOAT16_T: return visit.template operator()<sycl::half>();
#endif
        default: throw py::value_error("This build cannot visit that region dtype.");
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
    struct RegionArrayLayout
    {
        py::ssize_t Rows;
        py::ssize_t Columns;
        py::ssize_t RowStride;
        explicit RegionArrayLayout(const BidirectionalInMemGraph::SNTOComputerBridge<T>& bridge)
        {
            constexpr uint64_t max_size = std::numeric_limits<py::ssize_t>::max();
            const uint64_t row_stride = static_cast<uint64_t>(bridge.CapacityWidth()) * sizeof(T);
            if (
                row_stride == 0 ||
                bridge.Height() >(max_size / row_stride)
            ) 
            {
                throw std::overflow_error("Region Dimensions exceeds Numpy's index range");
            }
            Rows = bridge.Height();
            Columns = bridge.ActiveWidth();
            RowStride = static_cast<py::ssize_t>(row_stride);
        }

        py::ssize_t RowBytes() const noexcept
        {
            return Columns * static_cast<py::ssize_t>(sizeof(T));
        }
    };

    template<class T>
    struct RegionArrayOwner
    {
        std::shared_ptr<BidirectionalInMemGraph::APCFinilizer> Fabric;
        BidirectionalInMemGraph::SNTOComputerBridge<T> Bridge;
    };

    template<class T>
    py::array RegionView(APCHandle& apc, MacroColumn column, bool writable) noexcept;

    template<class T>
    void WriteRegion(
        APCHandle& apc, MacroColumn column,
        const py::array& values
    ) noexcept
    {
        using namespace BidirectionalInMemGraph;
        SNTOComputerBridge<T> bridge = apc.BuildAViewOverRegion<T>(column);
        std::optional<std::span<T>> cells = bridge.RawSpan();
        if (!cells)
        {
            throw std::runtime_error("Expected an accessible private region with the requested dtype.");
        }
        
    }

}
