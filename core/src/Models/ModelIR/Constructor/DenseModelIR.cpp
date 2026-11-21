#include <SuperNova>

namespace BidirectionalInMemGraph::GMI::Dense 
{
    using SD = SchemaDefinition;
// INTERNAL - no header necessery
namespace 
{
    constexpr uint32_t MAGIC = 0x52494e44u; // DNIR
    constexpr uint32_t VERSION = 2u;
    constexpr uint32_t FIRST_VERSION = 1u; 
    constexpr size_t FINGERPRINT_BYTES = 64u;
    constexpr size_t OPERATOR_BYTES = 13u;
    constexpr size_t TENSOR_HEADER_BYTES = 15u;
    constexpr size_t BINDING_BYTES = 21u;
    constexpr uint8_t DIM_OF_WEIGHT_TENSOR = 2u;
    constexpr uint8_t DIM_OF_BIAS_TENSOR = 1u;
    
    

    constexpr void Require(bool valid)
    {
        if (!valid) throw std::invalid_argument("Invalid Dense ModelIR configuration.");
    }

    /// little-endian serialization
    template<class UInt>
    void Put(std::vector<std::byte>& bytes, UInt value)
    {
        static_assert(sizeof(UInt) <= sizeof(uint64_t));
        for (size_t i = 0; i < sizeof(UInt); i ++) 
        {
            bytes.push_back(static_cast<std::byte>(
                (uint64_t{value} >> (LEN_OF_BYTE_IN_BITS * i)) & 0xffu
            )); // oxffu means 255 max of a byte and this operation shifts 
        }
    }

    void PutSize(std::vector<std::byte>& bytes, size_t value) 
    {
        Require(value <= ADS::APC_INDEX_BOUND_SENTINAL);
        Put(bytes, static_cast<uint32_t>(value));
    }

    class Reader final
    {
        std::span<const std::byte> Bytes_;
    public:
        explicit Reader(std::span<const std::byte> bytes) :
            Bytes_(bytes)
        {}

        constexpr std::span<const std::byte> Take(size_t count)
        {
            Require(count <= Bytes_.size());
            const auto result = Bytes_.first(count);
            Bytes_ = Bytes_.subspan(count);
            return result;
        }

        template<class UInt>
        UInt Get()
        {
            const std::span<const std::byte> bytes = Take(sizeof(UInt));
            uint64_t value = UNSIGNED_ZERO;
            for (size_t i = 0; i < bytes.size(); i++) 
            {
                value |= std::to_integer<uint64_t>(bytes[i]) << (LEN_OF_BYTE_IN_BITS * i);
            }
            return static_cast<UInt>(value);
        }

        constexpr void  CheckCount(uint32_t count, size_t record_bytes) const
        {
            Require(count <= (Bytes_.size() / record_bytes));
        }

        bool Empty()const noexcept {return Bytes_.empty();}
    };
}
    std::vector<std::byte> DenseModelDynamics::Encode() const
    {
        std::vector<std::byte> bytes;
        Put(bytes, MAGIC);
        Put(bytes, VERSION);
        PutSize(bytes, Tensors.size());

        for (const TensorDescription& tensor : Tensors) 
        {
            Require(tensor.Shape.size() <= UINT8_MAX);
            PutSize(bytes, tensor.Name.size());
            Put(bytes, static_cast<uint8_t>(tensor.Role));
            Put(bytes, static_cast<uint8_t>(tensor.DType));
            Put(bytes, tensor.LayerIndex);
            Put(bytes, static_cast<uint8_t>(tensor.Shape.size()));
            PutSize(bytes, tensor.Bindings.size());
            const std::span name = std::as_bytes(std::span{
                tensor.Name.data(), tensor.Name.size()
            });
            bytes.insert(bytes.end(), name.begin(), name.end());
            for (const uint32_t dim : tensor.Shape) 
            {
                Put(bytes, dim);
            }
            for (const TensorBindings& binding : tensor.Bindings) 
            {
                Put(bytes, binding.APCIndex);
                Put(bytes, static_cast<uint8_t>(binding.Column));
                Put(bytes, binding.TensorOffset);
                Put(bytes, binding.APCOffset);
                Put(bytes, binding.ElementCount);
            }
        }

        Require(
            TopologyFingerPrint.empty() || 
            TopologyFingerPrint.size() == FINGERPRINT_BYTES
        );
        Put(bytes, static_cast<uint8_t>(!TopologyFingerPrint.empty()));
        for (const unsigned char digit : TopologyFingerPrint)
        {
            Put(bytes, digit);
        }
        Put(bytes, static_cast<uint8_t>(Graph.has_value()));
        if (Graph) 
        {
            PutSize(bytes, Graph->Operators.size());
            Put(bytes, Graph->Output);
            for (const Operator& op : Graph->Operators) 
            {
                Put(bytes, static_cast<uint8_t>(op.Kind));
                Put(bytes, op.Input);
                Put(bytes, op.Weight);
                Put(bytes, op.Bias);
            }
        }
        return bytes;
    }


    DenseModelDynamics DenseModelDynamics::Decode(std::span<const std::byte> bytes)
    {
        Reader reader(bytes);
        Require(reader.Get<uint32_t>() == MAGIC);
        const uint32_t version = reader.Get<uint32_t>();
        Require(version == FIRST_VERSION || version == VERSION);
        const uint32_t count = reader.Get<uint32_t>();
        reader.CheckCount(count, TENSOR_HEADER_BYTES);

        DenseModelDynamics dense_ir;
        dense_ir.Tensors.reserve(count);
        for (uint32_t i = 0; i < count; i++) 
        {
            TensorDescription tensor;
            const uint32_t name_size = reader.Get<uint32_t>();
            tensor.Role = static_cast<TensorRole>(reader.Get<uint8_t>());
            tensor.DType = static_cast<SD::DataTypeOfMacroColumn>(reader.Get<uint8_t>());
            tensor.LayerIndex = reader.Get<uint32_t>();
            const  uint8_t rank = reader.Get<uint8_t>();
            const uint32_t bindings = reader.Get<uint32_t>();
            const std::span name = reader.Take(name_size);
            tensor.Name.assign(reinterpret_cast<const char*>(name.data()), name.size());
            for (uint16_t d = 0; d < rank; d++) 
            {
                tensor.Shape.push_back(reader.Get<uint32_t>());
            }
            reader.CheckCount(bindings, BINDING_BYTES);
            tensor.Bindings.reserve(bindings);
            for (uint32_t b = 0; b < bindings; b++) 
            {
                tensor.Bindings.push_back({
                    reader.Get<uint32_t>(),
                    static_cast<MacroColumnOfAPC>(reader.Get<uint8_t>()),
                    reader.Get<uint64_t>(),
                    reader.Get<uint32_t>(),
                    reader.Get<uint32_t>()
                });
            }
            dense_ir.Tensors.push_back(std::move(tensor));
        }
        if (version >= VERSION) 
        {
            const uint8_t has_fingerprint = reader.Get<uint8_t>();
            Require(has_fingerprint <= 1u);
            if (has_fingerprint) 
            {
                const std::span fingerprint = reader.Take(FINGERPRINT_BYTES);
                dense_ir.TopologyFingerPrint.assign(
                    reinterpret_cast<const char*>(fingerprint.data()),
                    fingerprint.size()
                );
            }

            const uint8_t has_graph = reader.Get<uint8_t>();
            Require(has_graph <= 1u);
            if (has_graph) 
            {
                OperatorGraph& graph = dense_ir.Graph.emplace();
                const uint32_t count = reader.Get<uint32_t>();
                graph.Output = reader.Get<uint32_t>();
                reader.CheckCount(count, OPERATOR_BYTES);
                graph.Operators.reserve(count);
                for (uint32_t i = 0; i < count; i++) 
                {
                    graph.Operators.push_back({
                        static_cast<OperatorKind>(reader.Get<uint8_t>()),
                        reader.Get<uint32_t>(),
                        reader.Get<uint32_t>(),
                        reader.Get<uint32_t>()
                    });
                }
            }
        }
        Require(reader.Empty());
        return dense_ir;
    }

    void DenseModelDynamics::Validate(std::span<const LayerDescription> layers) const
    {
        using __Region__ = std::tuple<uint32_t, uint32_t, MacroColumnOfAPC>;
        struct __Range__ final
        {
            __Region__ Region{};
            uint32_t Offset{};
            uint64_t End{};
            SD::DataTypeOfMacroColumn Dtype{};
        };

        struct __ValueGeometry__ final
        {
            uint32_t Width{};
            SD::DataTypeOfMacroColumn DType{};
        };

        std::vector<__Range__> ranges;
        std::unordered_set<std::string_view> names;
        for (const TensorDescription& tensor : Tensors) 
        {
            Require(
                tensor.LayerIndex < layers.size() &&
                (
                    tensor.Role == TensorRole::Parameter ||
                    tensor.Role == TensorRole::Buffer
                ) &&
                SD::DTypeByteCount(tensor.DType).has_value() &&
                !tensor.Name.empty() &&
                tensor.Name.find('\0') == std::string::npos &&
                names.insert(tensor.Name).second
            );

            uint64_t count = UNSIGNED_ZERO;
            if (std::ranges::find(tensor.Shape, UNSIGNED_ZERO) == tensor.Shape.end()) 
            {
                count = 1;
                for (const uint32_t dim : tensor.Shape) 
                {
                    Require(count <= (std::numeric_limits<uint64_t>::max() / dim));
                    count *= dim;
                }
            }

            uint64_t cursor = UNSIGNED_ZERO;
            for (const TensorBindings& binding : tensor.Bindings) 
            {
                const uint64_t end = uint64_t{binding.APCOffset} + binding.ElementCount;
                Require(
                    binding.APCIndex < layers[tensor.LayerIndex].APCCount &&
                    static_cast<uint8_t>(binding.Column) < ADS::CountOfMacroColumn() &&
                    binding.ElementCount != UNSIGNED_ZERO &&
                    binding.TensorOffset == cursor &&
                    binding.ElementCount <= (count - cursor) &&
                    end <= ADS::APC_INDEX_BOUND_SENTINAL
                );
                cursor += binding.ElementCount;
                ranges.emplace_back(
                    __Region__{tensor.LayerIndex, binding.APCIndex, binding.Column},
                    binding.APCOffset, 
                    end,
                    tensor.DType
                );
            }
            Require(cursor == count);
        }
        std::sort(
            ranges.begin(),
            ranges.end(),
            [](const __Range__& a, const __Range__& b)
            {
                return std::tie(a.Region, a.Offset)
                    < std::tie(b.Region, b.Offset);
            }
        );

        for (size_t i = 1; i < ranges.size(); i++) 
        {
            const __Range__& range = ranges[i];
            const __Range__& previous_range = ranges[i - 1];            
            if (range.Region == previous_range.Region) 
            {
                Require(
                    range.Offset >= previous_range.End &&
                    range.Dtype == previous_range.Dtype
                );
            }
        }

        Require(
            TopologyFingerPrint.empty() ||
            (
                TopologyFingerPrint.size() == FINGERPRINT_BYTES &&
                TopologyFingerPrint.find_first_not_of("0123456789abcdef") == std::string::npos

            )
        );
        if (!Graph) return;

        Require(!TopologyFingerPrint.empty() && Graph->Output <= Graph->Operators.size());

        std::vector<size_t> origins{UNSIGNED_ZERO};
        std::vector<std::optional<__ValueGeometry__>> geometry{1};
        for (const Operator& op : Graph->Operators) 
        {
            Require(op.Input < origins.size());
            if (op.Kind == OperatorKind::ReLU) 
            {
                Require(op.Weight == NO_TENSOR && op.Bias == NO_TENSOR);
                const size_t origin = origins[op.Input];
                origins.push_back(origin);
                geometry.emplace_back();
                continue;
            }

            Require(op.Kind == OperatorKind::Linear && op.Weight < Tensors.size());
            const TensorDescription& weight_tensor = Tensors[op.Weight];
            Require(weight_tensor.Role == TensorRole::Parameter && weight_tensor.Shape.size() == DIM_OF_WEIGHT_TENSOR);
            if (op.Bias != NO_TENSOR) 
            {
                Require(op.Bias < Tensors.size());
                const TensorDescription& bias_tensor = Tensors[op.Bias];
                Require(
                    bias_tensor.Role == TensorRole::Parameter &&
                    bias_tensor.Shape.size() == DIM_OF_BIAS_TENSOR &&
                    bias_tensor.DType == weight_tensor.DType &&
                    bias_tensor.Shape[0] == weight_tensor.Shape[0]
                );
            }

            std::optional<__ValueGeometry__>& input = geometry[origins[op.Input]];
            Require(!input.has_value() || (
                input->Width == weight_tensor.Shape[1] &&
                input->DType == weight_tensor.DType
            ));

            input = __ValueGeometry__{
                weight_tensor.Shape[1],
                weight_tensor.DType
            };
            origins.push_back(geometry.size());
            geometry.emplace_back(__ValueGeometry__(
                weight_tensor.Shape[0],
                weight_tensor.DType
            ));
        }

    }


}