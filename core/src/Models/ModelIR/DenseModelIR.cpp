#include <SuperNova>

namespace BidirectionalInMemGraph::GMI::Dense 
{
// INTERNAL - no header necessery
namespace 
{
    constexpr uint32_t MAGIC = 0x52494e44u; // DNIR
    constexpr uint32_t Version = 01u;
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


    };
}

}