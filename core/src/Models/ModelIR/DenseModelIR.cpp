#include <SuperNova>

namespace BidirectionalInMemGraph::GMI::Dense 
{
// INTERNAL - no header necessery
namespace 
{
    constexpr uint32_t MAGIC = 0x52494e44u; // DNIR
    constexpr uint32_t Version = 1u;
    constexpr size_t FINGERPRINT_BYTES = 64u;
    constexpr size_t OPERATOR_BYTES = 13u;
    constexpr size_t TENSOR_HEADER_BYTES = 15u;
    constexpr size_t BINDING_BYTES = 21u;
    

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
        Put(bytes, Tensors.size());

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

}