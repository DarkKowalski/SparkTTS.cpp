#pragma once

#include <vector>
#include <array>
#include <cstdint>

namespace spark_tts
{
    // Overlap semantic tokens to eliminate voice gaps
    // Example: overlapped = 1
    // |---audio decoder input---|
    //      |-----valid-----|trim|
    // | 01 | 02 | ... | 49 | 50 |
    //                 | 01 | 02 | ... | 49 | 50 |
    //                 |trim|-----valid-----|

    class TokenBuffer
    {
    public:
        TokenBuffer(const size_t capacity, const size_t overlapped_tokens);

    public:
        // return true if the buffer is full, false otherwise
        // semantic_tokens size won't exceed capacity
        bool add_tokens(const std::vector<int64_t> &semantic_tokens);

        void clear(); // Clear all buffers and reset the front buffer index

        void flip(); // Clear the front buffer and swap it with the back buffer

        std::vector<int64_t> &front_buffer();

    private:
        std::vector<int64_t> &back_buffer();

    private:
        // Front buffer is the working buffer where new tokens are added.
        // Back buffer is the buffer that holds the last overlapped tokens.
        std::array<std::vector<int64_t>, 2> buffers_;
        size_t front_buffer_index_ = 0;

        size_t capacity_;
        size_t overlapped_tokens_;
    };
} // namespace spark_tts
