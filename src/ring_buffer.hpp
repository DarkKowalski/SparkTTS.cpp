#pragma once

#include <vector>
#include <stdexcept>

namespace spark_tts
{
    template <typename T>
    class RingBuffer
    {
    public:
        RingBuffer(size_t capacity) : capacity_(capacity), data_(capacity) {}

        T &front()
        {
            if (size_ == 0)
            {
                throw std::out_of_range("Ring buffer is empty");
            }
            return data_[first_];
        }

        const T &front() const
        {
            if (size_ == 0)
            {
                throw std::out_of_range("Ring buffer is empty");
            }
            return data_[first_];
        }

        T &back()
        {
            if (size_ == 0)
            {
                throw std::out_of_range("Ring buffer is empty");
            }
            return data_[pos_];
        }

        const T &back() const
        {
            if (size_ == 0)
            {
                throw std::out_of_range("Ring buffer is empty");
            }
            return data_[pos_];
        }

        void push_back(const T &value)
        {
            if (size_ == capacity_)
            {
                first_ = (first_ + 1) % capacity_;
            }
            else
            {
                size_++;
            }
            data_[pos_] = value;
            pos_ = (pos_ + 1) % capacity_;
        }

        T pop_front()
        {
            if (size_ == 0)
            {
                throw std::out_of_range("Ring buffer is empty");
            }
            T value = data_[first_];
            first_ = (first_ + 1) % capacity_;
            size_--;
            return value;
        }

        const T &at(size_t index) const
        {
            if (index >= size_)
            {
                throw std::out_of_range("Ring buffer index out of bounds");
            }
            return data_[(first_ + size_ - index - 1) % capacity_];
        }

        std::vector<T> to_vector() const
        {
            std::vector<T> result;
            result.reserve(size_);
            for (size_t i = 0; i < size_; i++)
            {
                result.push_back(data_[(first_ + i) % capacity_]);
            }
            return result;
        }

        void clear()
        {
            size_ = 0;
            first_ = 0;
            pos_ = 0;
        }

        bool empty() const
        {
            return size_ == 0;
        }

        size_t size() const
        {
            return size_;
        }

    private:
        size_t capacity_ = 0;
        size_t size_ = 0;
        size_t first_ = 0;
        size_t pos_ = 0;
        std::vector<T> data_;
    };
} // namespace spark_tts
