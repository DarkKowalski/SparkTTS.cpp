#include "token_buffer.h"

namespace spark_tts
{
    TokenBuffer::TokenBuffer(const size_t size, const size_t overlapped_tokens)
        : capacity_(size), overlapped_tokens_(overlapped_tokens)
    {
        buffers_[0].reserve(capacity_);
        buffers_[1].reserve(capacity_);
    }

    bool TokenBuffer::add_tokens(const std::vector<int64_t> &semantic_tokens)
    {
        // assert semantic_tokens.size() <= capacity_
        const size_t remaining_space = capacity_ - front_buffer().size();
        if (semantic_tokens.size() < remaining_space)
        {
            front_buffer().insert(front_buffer().end(), semantic_tokens.begin(), semantic_tokens.end());
            return false; // Buffer is not full
        }
        else if (semantic_tokens.size() == remaining_space)
        {
            front_buffer().insert(front_buffer().end(), semantic_tokens.begin(), semantic_tokens.end());

            // save last overlapped tokens to back buffer
            back_buffer().insert(back_buffer().end(), front_buffer().end() - overlapped_tokens_ * 2, front_buffer().end());

            return true; // Buffer is full
        }
        else if (semantic_tokens.size() > remaining_space)
        {
            front_buffer().insert(front_buffer().end(), semantic_tokens.begin(), semantic_tokens.begin() + remaining_space);

            // save overlapped tokens to back buffer
            back_buffer().insert(back_buffer().end(), front_buffer().end() - overlapped_tokens_ * 2, front_buffer().end());
            // save remaining tokens to back buffer
            back_buffer().insert(back_buffer().end(), semantic_tokens.begin() + remaining_space, semantic_tokens.end());

            return true; // Buffer is full
        }

        return false; // Should not reach here
    }

    void TokenBuffer::clear()
    {
        front_buffer().clear();
        back_buffer().clear();
        front_buffer_index_ = 0;
    }

    void TokenBuffer::flip()
    {
        front_buffer().clear();
        front_buffer_index_ = 1 - front_buffer_index_;
    }

    std::vector<int64_t> &TokenBuffer::front_buffer() { return buffers_[front_buffer_index_]; }

    std::vector<int64_t> &TokenBuffer::back_buffer()
    {
        return buffers_[1 - front_buffer_index_];
    }

} // namespace spark_tts
