#include "MemoryDefinition.hpp"

/// DISK - MEMORY - Checkpoint
namespace BidirectionalInMemGraph::CheckpointFormat
{

    template<class Reader>
    static bool WriteStremed(
        const std::filesystem::path path,
        FabricId id,
        uint64_t revision,
        const CFC::FabricCache& cache,
        Reader reader,
        std::span<const std::byte> meta_data = {}
    ) noexcept
    {
        std::filesystem::path temp_path;
        try 
        {
            if (
                !id || meta_data.size() > IV::MAX_META_BYTES ||
                !StorageAllocationDef::ValidateFabricCache(cache)
            ) 
            {
                return false;
            }
            const uint64_t total_cells = cache.SlabCellCount_;
            const uint64_t total_bytes = total_cells * IV::BYTES_PER_CELL;
            const uint64_t pages = PageCount(total_bytes);

            std::array<std::byte, IV::PAGE_BYTES> buffer;
            if (
                total_cells > LIMIT_OF_CELL_COUNT ||
                pages > IV::MAX_PAGE_COUNT
            ) 
            {
                return false;
            }

            //PAGE HASHES
            std::unique_ptr hashes = std::make_unique<std::array<uint64_t, IV::MAX_PAGE_COUNT>>();
            for (uint64_t p = 0; p < pages; p++) 
            {
                const uint64_t begin = p * IV::PAGE_BYTES;
                const size_t len = static_cast<size_t>(
                    std::min<uint64_t>(IV::PAGE_BYTES, (total_bytes - begin))
                );

                std::span bytes = std::span<std::byte>(buffer.data(), len);
                if (!reader(begin, bytes)) return false;
                (*hashes)[p] = PageHash(bytes);
            }

            if (pages > (UINT64_MAX - IV::HEADER_BYTES - meta_data.size())) 
            {
                return false;
            }

            const uint64_t directory_bytes = pages * IV::BYTES_PER_CELL + meta_data.size();
            const uint64_t payload_offset = IV::HEADER_BYTES * directory_bytes;
            if (total_bytes > (UINT64_MAX - payload_offset)) return false;

            uint64_t directory_hash = IV::FnvOffsetBasis;

            FileHeader header
            {
                IV::FILE_MAGIC,
                IV::FILE_VERSION,
                IV::PAGE_BYTES,
                id.High,
                id.Low,
                revision,
                static_cast<uint64_t>(total_cells),
                static_cast<uint64_t>(meta_data.size()),
                payload_offset,
                directory_hash,
                (payload_offset + total_bytes),
                UNSIGNED_ZERO
            };

            header[PageHeaderIndex(PageHeader::HEADER_HASH)] = HedaderHash(header);

            const FabricId fab_id = FabricId::New();

            temp_path = path;
            temp_path += std::string(".tmp-") + std::to_string(fab_id.High) +
                "-" + std::to_string(fab_id.Low);
            {
                std::ofstream output(temp_path, (std::ios::binary | std::ios::trunc));
                if (!output) return false;
                for (uint64_t val : header) 
                {
                    if (!WritePageHeaderCells(output, val)) 
                    {
                        throw std::runtime_error("header write");
                    }
                }
                for (uint64_t p = 0; p < pages; p++) 
                {
                    if (!WritePageHeaderCells(output, (*hashes)[p])) 
                    {
                        throw std::runtime_error("directory write");
                    }
                }

                output.write(
                    reinterpret_cast<const char*>(meta_data.data()), 
                    static_cast<std::streamsize>(meta_data.size())  
                );

                for (uint64_t p = 0; p < pages; p++) 
                {
                    const uint64_t begin = p * IV::PAGE_BYTES;
                    const size_t len = static_cast<size_t>(
                        std::min<uint64_t>(IV::PAGE_BYTES, (total_bytes - begin))
                    );

                    std::span bytes = std::span<std::byte>(buffer.data(), len);
                    if (
                        !reader(begin, bytes) ||
                        PageHash(bytes) != (*hashes)[p]
                    )
                    {
                        throw std::runtime_error("payload changed during checkpoint");
                    }
                }
                output.flush();
                if (!output) throw std::runtime_error("payload write");
            }
        
        #if !defined (_WIN32)
            const int fd = ::open(temp_path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("open flush");

            const int synced = ::fsync(fd);
            ::close(fd);
            if (synced != 0) throw std::runtime_error("file flush");
        #endif
        
            std::filesystem::rename(temp_path, path);
            temp_path.clear();

        #if !defined(_WIN32)
            auto parent = path.parent_path();
            if (parent.empty()) parent = ".";

            const int directory_fd = ::open(parent.c_str(), (O_RDONLY | O_DIRECTORY));
            if (directory_fd < 0) return false;

            const int directory_synced = ::fsync(directory_fd);
            ::close(directory_fd);
            if (directory_synced != 0) return false;
        #endif
            return true;
        }
        catch (...)
        {
            if (!temp_path.empty()) 
            {
                std::error_code ignored;
                std::filesystem::remove(temp_path, ignored);
            }
            return false;
        }
    }

    static bool GenericPayloadReader(
        uint64_t offset,
        std::span<std::byte> output,
        std::span<const std::byte> hot_bytes,
        std::span<const std::byte> numerical_bytes
    ) noexcept
    {
        if (hot_bytes.size() > (SIZE_MAX - numerical_bytes.size())) 
        {
            return false;
        }

        const size_t total_bytes = hot_bytes.size() + numerical_bytes.size();
        if (offset > total_bytes) 
        {
            return false;
        }

        const size_t begin = static_cast<size_t>(offset);
        if (output.size() > (total_bytes - begin)) 
        {
            return false;
        }

        size_t copied_hot = UNSIGNED_ZERO;

        if (begin < hot_bytes.size()) 
        {
            copied_hot = std::min(output.size(), (hot_bytes.size() - begin));
            if (copied_hot != UNSIGNED_ZERO) 
            {
                std::memcpy(output.data(), (hot_bytes.data() + begin), copied_hot);
            }
        }

        const size_t remaining = output.size() - copied_hot;
        if (remaining != UNSIGNED_ZERO) 
        {
            const size_t numerical_offset = begin + copied_hot - hot_bytes.size();
            std::memcpy(
                output.data() + copied_hot,
                numerical_bytes.data() + numerical_offset,
                remaining
            );
        }
        return true;
    }

    static bool WriteCheckpointToDisk(
        const std::filesystem::path& path,
        FabricId id,
        uint64_t revision,
        std::span<const uint64_t> hot_slab,
        std::span<const std::byte> meta_data = {},
        std::span <const uint64_t> numerical = {}
    ) noexcept
    {
        if (
            hot_slab.size() > LIMIT_OF_CELL_COUNT ||
            numerical.size() > LIMIT_OF_CELL_COUNT ||
            hot_slab.size_bytes() < sizeof(CFC::FabricCache)
        ) 
        {
            return false;
        }

        CFC::FabricCache cache{};
        std::memcpy(&cache, hot_slab.data(), sizeof(CFC::FabricCache));

        const uint64_t total_cells = hot_slab.size() + numerical.size();
        if (total_cells != cache.SlabCellCount_) return false;

        if (
            !numerical.empty() &&
            hot_slab.size() != cache.SegmentPoolBegin_
        ) 
        {
            return false;
        }

        const std::span<const std::byte> hot_bytes = std::as_bytes(hot_slab);
        const std::span<const std::byte> numerical_bytes = std::as_bytes(numerical);
        
        auto reader = [&](uint64_t offset, std::span<std::byte> output) noexcept -> bool
        {
            return GenericPayloadReader(offset, output, hot_bytes, numerical_bytes);
        };
        
        return WriteStremed(
            path,
            id,
            revision,
            cache,
            reader,
            meta_data
        );
    }
}