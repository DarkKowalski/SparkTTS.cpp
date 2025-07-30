#pragma once

#include <vector>
#include <array>
#include <cstdint>

namespace spark_tts
{
    class TokenBuffer
    {
    public:
        TokenBuffer(const size_t capacity, const size_t overlapped_tokens);

    public:
        // return true if the buffer is full, false otherwise
        // semantic_tokens size won't exceed capacity
        bool add_tokens(const std::vector<int64_t> &semantic_tokens);

        void flip()
        {
            // clear and swap buffers
            front_buffer().clear();
            front_buffer_index_ = 1 - front_buffer_index_;
        }

        std::vector<int64_t> &front_buffer() { return buffers_[front_buffer_index_]; }

    private:
        std::vector<int64_t> &back_buffer()
        {
            return buffers_[1 - front_buffer_index_];
        }

    private:
        // Front buffer is the working buffer where new tokens are added.
        // Back buffer is the buffer that holds the last overlapped tokens.
        std::array<std::vector<int64_t>, 2> buffers_;
        size_t front_buffer_index_ = 0;

        size_t capacity_;
        size_t overlapped_tokens_;
    };
} // namespace spark_tts
