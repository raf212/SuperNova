#pragma once
#include "AdaptivePackedCellContainer/APCOrchestrators/SharedConf.hpp"
#include "SuperNova"
#include <cstdint>
#include <limits>
#include <sys/types.h>

namespace BidirectionalInMemGraph
{
    void NumericalBacking::Swap_(NumericalBacking& other) noexcept
    {
        std::swap(Data_, other.Data_);
        std::swap(Bytes_, other.Bytes_);
        PendingPath_.swap(other.PendingPath_);
        std::swap(File_, other.File_);
    #if defined (_WIN32)
        std::swap(Mapping_, other.Mapping_);
    #endif
    }

    void NumericalBacking::Close() noexcept
    {
    #if defined (_WIN32)
        if (Data_) UnmapViewOfFile(Data_);

        if (Mapping_) CloseHandle(Mapping_);

        if (File_ != INVALID_HANDLE_VALUE) CloseHandle(File_);

    #else
        if (Data_) 
        {
            ::munmap(Data_, Bytes_);
        }
        if (File_ >= UNSIGNED_ZERO) 
        {
            ::close(File_);
        }
        File_ = -1 
    #endif
        Data_ = nullptr;
        Bytes_ = UNSIGNED_ZERO;
        if (!PendingPath_.empty()) 
        {
            std::error_code ignored;
            std::filesystem::remove(PendingPath_, ignored);
            PendingPath_.clear();
        }
    }

#if defined (_WIN32)
    bool NumericalBacking::CreateWinMap_() noexcept
    {
        File_ = CreateFileW(
            PendingPath_.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            UNSIGNED_ZERO,
            nullptr,
            CREATE_NEW,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (File_ == INVALID_HANDLE_VALUE) 
        {
            PendingPath_.clear();
            Bytes_ = UNSIGNED_ZERO;
            return false;
        }

        LARGE_INTEGER size;
        size.QuadPart = static_cast<LONGLONG>(Bytes_);

        if (
            !SetFilePointerEx(File_, size, nullptr, FILE_BEGIN) ||
            !SetEndOfFile(File_)
        ) 
        {
            Close();
            return false;
        }

        Mapping_ = CreateFileMappingW(
            File_,
            nullptr,
            PAGE_READWRITE,
            UNSIGNED_ZERO,
            UNSIGNED_ZERO,
            nullptr
        );

        if (!Mapping_) {Close(); return false;}

        Data_ = static_cast<uint64_t*>(MapViewOfFile(
            Mapping_,
            FILE_MAP_ALL_ACCESS,
            UNSIGNED_ZERO,
            UNSIGNED_ZERO,
            Bytes_
        ));

        if (!Data_) {Close(); return false;}
        return true;
    }

    bool NumericalBacking::WinPrefetch_(uint64_t first_cell, uint64_t cells) noexcept
    {
        struct RangePF__
        {
            void* VirtualAddress;
            SIZE_T NumberOfBytes;
        };

        using FN__ = BOOL (WINAPI*) (HANDLE, ULONG_PTR, RangePF__*, ULONG);
        auto function = reinterpret_cast<FN__>(GetProcAddress(
            GetModuleHandleW(L"kernel32.dll"),
            "PrefetchVirtualMemory"
        ));
        RangePF__ range{
            Data_ + first_cell,
            static_cast<SIZE_T>(cells * sizeof(uint64_t))
        };
        return function && function(GetCurrentProcess(), 1, &range, 0);
    }

#else

    bool NumericalBacking::CreateLinMap_(uint64_t cells) noexcept
    {
        File_ = ::open(PendingPath_.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        if (File_ < 0) 
        {
            PendingPath_.clear();
            Bytes_ = UNSIGNED_ZERO;
            return false;
        }

        if (
            cells > static_cast<uint64_t>(std::numeric_limits<off_t>::max() / sizeof(uint64_t)) ||
            ::ftruncate(File_, static_cast<off_t>(Bytes_) != UNSIGNED_ZERO
        )
        ) 
        {
            Close();
            return false;   
        }

        void* address = ::mmap(
            nullptr, 
            Bytes_, 
            PORT_READ | PORT_WRITE, 
            MAP_SHARED, 
            FILE, 
            UNSIGNED_ZERO
        );

        if (address != MAP_FAILED) 
        {
            Data_ = static_cast<uint64_t*>(address);
        }
    }

    bool NumericalBacking::LinPrefetch_(uint64_t first_cell, uint64_t cells) noexcept
    {
        const size_t page_size = ::sysconf(_SC_PAGE_SIZE);
        if (page_size < UNSIGNED_ZERO) return false;

        const size_t begin = static_cast<size_t>((first_cell * sizeof(uint64_t)) / page_size) * page_size;
        const size_t end = static_cast<size_t>((first_cell + cells) * sizeof(uint64_t));
        return ::madvise(
            reinterpret_cast<std::byte*>(Data_) + begin,
            end - begin,
            MADV_WILLNEED
        ) == UNSIGNED_ZERO;
    }

#endif

    bool NumericalBacking::Create(const std::filesystem::path& path, uint64_t cells) noexcept
    {
        if (
            Data_ ||
            !PendingPath_.empty() ||
            !cells ||
            cells > LIMIT_OF_CELL_COUNT
        ) 
        {
            return false;
        }

        try 
        {
            PendingPath_ = path;
        }
        catch (...)
        {
            PendingPath_.clear();
            return false;
        }
        Bytes_ = static_cast<size_t>(cells * sizeof(uint64_t));
    #if defined (_WIN32)
        return CreateWinMap_();
    #else
        return CreateLinMap_(cells);
    #endif
    }

    bool NumericalBacking::Flush() noexcept
    {
        if (!Data_) return false;

    #if defined (_WIN32)
        return FlushViewOfFile(Data_, Bytes_) &&
            FlushFileBuffers(File_);
    #else
        return ::msync(Data_, Bytes_, MS_SYNC) == UNSIGNED_ZERO && 
            ::fsync(File_);
    #endif
    }

    bool NumericalBacking::Prefetch(uint64_t first_cell, uint64_t cells) noexcept
    {
        if (
            !Data_ || first_cell > Bytes_ / sizeof(uint64_t) || 
            cells > (Bytes_ / sizeof(uint64_t)) - first_cell
        ) 
        {
            return false;
        }

        if (!cells) return false;
        
    #if defined (_WIN32)
        return WinPrefetch_(first_cell, cells);
    #else
        return LinPrefetch_(first_cell, cells);
    #endif
    }

    void NumericalBacking::WarmAll() noexcept
    {
        if (!Data_) return;
    #if defined (_WIN32)
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        const size_t page = info.dwPageSize;
    #else
        const long size = ::sysconf(_SC_PAGESIZE);
        const size_t page = size > UNSIGNED_ZERO ? static_cast<size_t>(size) : 4096;
    #endif
        Data_ = nullptr;
        Bytes_ = UNSIGNED_ZERO;
        if (!PendingPath_.empty()) 
        {
            std::error_code ignored;
            std::filesystem::remove(PendingPath_, ignored);
            PendingPath_.clear();
        }
    }







}
