#pragma once
#include <atomic>
#include <cstdint>
#include <exception>
#include "SharedConf.hpp"

namespace BidirectionalInMemGraph 
{
    class FabricExecutionGate;

    class ReadLease final
    {
    private:
        friend class FabricExecutionGate;
        FabricExecutionGate* Gate_{nullptr};
        bool Counted_{false};
        ReadLease(FabricExecutionGate* gate, bool counted) noexcept :
            Gate_(gate), Counted_(counted)
        {}
    public:
        ReadLease() noexcept = default;
        ReadLease(const ReadLease&) = delete;
        ReadLease& operator = (const ReadLease&) = delete;
        ReadLease(ReadLease&& x) noexcept:
            Gate_(std::exchange(x.Gate_, nullptr)), Counted_(x.Counted_)
        {}
        ReadLease& operator = (ReadLease&& x) noexcept 
        {
            if (this != &x) 
            {
                Reset(Gate_);
                Gate_ = std::exchange(x.Gate_, nullptr);
                Counted_ = x.Counted_;
            }
            return *this;
        }
        ~ReadLease()
        {
            Reset(Gate_);
        }

        explicit operator bool() const noexcept 
        {
            return Gate_ != nullptr;
        }

        void Reset(FabricExecutionGate* owner) noexcept;
        void Reset() noexcept
        {
            Reset(Gate_);
        }
    };

    class ExclusiveLease final
    {
    private:
        friend class FabricExecutionGate;
        FabricExecutionGate* Gate_{nullptr};
        bool Counted_{false};
        ExclusiveLease (FabricExecutionGate* gate, bool counted) noexcept :
            Gate_(gate), Counted_(counted)
        {}
    public:
        ExclusiveLease() noexcept = default;
        ExclusiveLease(const ExclusiveLease&) = delete;
        ExclusiveLease& operator = (const ExclusiveLease&) = delete;
        ExclusiveLease(ExclusiveLease&& x) noexcept 
        {
            if (this != &x) 
            {
                Reset(Gate_);
                Gate_ = std::exchange(x.Gate_, nullptr);
                Counted_ = x.Counted_;
            }
        }

        ~ExclusiveLease()
        {
            Reset(Gate_);
        }

        explicit operator bool() const noexcept 
        {
            return Gate_ != nullptr;
        }

        bool Owns(const FabricExecutionGate& gate) const noexcept
        {
            return Gate_ == &gate && Counted_;
        }

        void Reset(FabricExecutionGate* owner) noexcept;
        void Reset() noexcept
        {
            Reset(Gate_);
        }
    };

    class FabricExecutionGate final
    {
    private:
        friend class ReadLease;
        friend class ExclusiveLease;
        static constexpr uint64_t WRITER = uint64_t{1} << (BIT_COUNT_OF_UINT64_T - 1);
        std::atomic<uint64_t> State_{UNSIGNED_ZERO};
        std::atomic<bool> Poisoned_{false};
        std::atomic<uint64_t> HotStamp_{1};
        inline static thread_local FabricExecutionGate* Owner_{nullptr};
        inline static thread_local bool Maintenance_{false};
    
    public:
        void NotifyHostChange() noexcept
        {
            HotStamp_.fetch_add(1, std::memory_order_release);
        }

        uint64_t HostStamp()const noexcept
        {
            return  HotStamp_.load(std::memory_order_acquire);
        }

        bool IsPoisoned() const noexcept
        {
            return Poisoned_.load(std::memory_order_acquire);
        } 

        void Poison() noexcept
        {
            Poisoned_.store(true, std::memory_order_release);
        }

        bool EpochOwnedByCaller() const noexcept
        {
            return Owner_ == this && !Maintenance_;
        }

        bool ConfirmedRestored() noexcept;

        ReadLease TryRead(bool allow_epoch_pin = false) noexcept;

        ExclusiveLease TryEpoch() noexcept;

        ExclusiveLease TryMaintenance() noexcept;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    };

}