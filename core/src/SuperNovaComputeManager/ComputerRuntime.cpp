#include "SuperNovaComputeManager/ComputerRuntime.hpp"

namespace BidirectionalInMemGraph::ComputerDetail
{
    bool Device::ConfigureGPU() noexcept
    {
        Reset();
    #if defined (SN_ACPP_ENABLED)
        try 
        {
            sycl::device gpu = sycl::device{sycl::gpu_selector_v};
            std::shared_ptr<Runtime> runtime = std::make_shared<Runtime>(gpu);
            FP16_ = gpu.has(sycl::aspect::fp16);
            FP64_ = gpu.has(sycl::aspect::fp64);
            Name_ = gpu.get_info<sycl::info::device::name>();
            Runtime_ = std::move(runtime);
            return true;
        }
        catch(...)
        {
            Reset();
        }
    #endif
        return false;
    }

    bool Device::StagingAvailable() const noexcept
    {
    #if defined (SN_ACPP_ENABLED)
        try 
        {
            return Runtime_ && Runtime_->Gpu.has(sycl::aspect::usm_device_allocations);
        }
        catch(...)
        {
            return false;
        }
    #else
        return false;
    #endif
    }
}

namespace BidirectionalInMemGraph
{
    bool FabricCheckpoint::ReadBytes_(
        uint64_t offset,
        std::span<std::byte> output
    ) const noexcept
    {
        if (Cells_  > FABRIC_CELL_SENTINAL / BYTES_PER_WORD) 
        {
            return false;
        }

        const uint64_t slab_bytes = Cells_ * BYTES_PER_WORD;
        if (
            offset > slab_bytes ||
            output.size() > slab_bytes - offset ||
            Hashes_.size() != PageCount(slab_bytes)
        ) 
        {
            return false;
        }

        std::ifstream input(Path_, std::ios::binary);
        if (!input) return false;

        for (uint64_t expected : Header_) 
        {
            uint64_t actual{};
            if (
                ReadWord(input, actual) ||
                actual != expected
            ) 
            {
                return false;
            }
        }

        if (output.empty()) 
        {
            return true;
        }

        std::vector<std::byte> page_buffer(
            static_cast<size_t>(std::min(PAGE_BYTES, slab_bytes))
        );

        uint64_t copied = UNSIGNED_ZERO;
        while (copied < output.size()) 
        {
            const uint64_t absolute = offset + copied;
            const uint64_t page = absolute / PAGE_BYTES;
            const uint64_t page_begin = page * PAGE_BYTES;

            const size_t page_length = static_cast<size_t>(
                std::min(PAGE_BYTES, slab_bytes - page_begin)
            );

            input.seekg(static_cast<std::streamoff>(Payload_ + page_begin));
            if (
                !input.read(
                    reinterpret_cast<char*>(page_buffer.data()), 
                    static_cast<std::streamsize>(page_length)
                )
            ) 
            {
                return false;
            }

            if (
                Hash(std::span<const std::byte>(page_buffer.data(), page_length)) != 
                Hashes_[static_cast<size_t>(page)]
            ) 
            {
                return false;
            }

            const size_t page_offset = static_cast<size_t>(absolute - page_begin);
            const size_t count = std::min(page_length - page_offset, output.size() - copied);
            std::copy_n(
                page_buffer.data() + page_offset, 
                count, 
                output.data() + copied
            );

            copied += count;
        }
        return true;
    }
}