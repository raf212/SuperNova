#pragma once
#include "SuperNova"

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

    bool NumericalBacking::Create(const std::filesystem::path& path, uint64_t cells) noexcept
    {
        if (
            Data_ ||
            !PendingPath_.empty() ||
            !cells ||
            cells > SIZE_MAX / sizeof(uint64_t) ||
            cells > uint64_t{INT64_MAX} / sizeof(uint64_t)
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

    #if defined (_WIN32)
        CreateWinMap_(cells);
    #else
        CreateLinMap_(cells);
    #endif
        
        if (!Data_) 
        {
            Close();
            return false;
        }
        return true;
    }



}
