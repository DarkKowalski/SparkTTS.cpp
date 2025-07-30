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
            back_buffer().insert(back_buffer().end(), front_buffer().end() - overlapped_tokens_, front_buffer().end());

            return true; // Buffer is full
        }
        else if (semantic_tokens.size() > remaining_space)
        {
            front_buffer().insert(front_buffer().end(), semantic_tokens.begin(), semantic_tokens.begin() + remaining_space);

            // save overlapped tokens to back buffer
            back_buffer().insert(back_buffer().end(), front_buffer().end() - overlapped_tokens_, front_buffer().end());
            // save remaining tokens to back buffer
            back_buffer().insert(back_buffer().end(), semantic_tokens.begin() + remaining_space, semantic_tokens.end());

            return true; // Buffer is full
        }

        return false; // Should not reach here
    }


} // namespace spark_tts
