#include "llm_streamer.h"

namespace spark_tts
{
    // FIXME: SUCKS, use a proper sliding window/ring buffer to handle this
    constexpr size_t overlapped_token = 2;

    void LLMStreamer::generate_audio()
    {
        if (semantic_token_index_ == 0)
        {
            return; // No tokens to detokenize
        }

        // pad semantic token buffer if needed
        if (semantic_token_index_ < semantic_token_buffer_size_)
        {
            for (size_t i = semantic_token_index_; i < semantic_token_buffer_size_; ++i)
            {
                semantic_token_buffer_[i] = 0; // padding with 0
            }
        }

        auto generated = audio_detokenizer_->detokenize(
            ov::Tensor(ov::element::i64, {1, semantic_token_buffer_size_}, semantic_token_buffer_.data()),
            global_tokens_);

        // Remove padding from generated audio
        // 1 semantic token can generate 320 audio samples at 16kHz
        // std::vector<float> audio_output(generated.begin(), generated.begin() + semantic_token_index_ * 320);
        // std::vector<float> audio_output(generated.begin(), generated.end() + semantic_token_index_ * 320);
        std::vector<float> audio_output(generated.begin() + overlapped_token * 320, generated.end() - overlapped_token * 320);
        on_audio_generated_(audio_output);

        // Reset semantic token buffer and index for next generation
        semantic_token_index_ = 0;
    }

    ov::genai::StreamingStatus LLMStreamer::write(const std::vector<int64_t> &tokens)
    {
        std::cout << "Received tokens: " << tokens.size() << std::endl;
        return ov::genai::StreamingStatus::RUNNING;
        const auto semantic_token_ids = spark_tts::extract_semantic_token_ids(tokenizer_.decode(tokens));

        for (const auto &token_id : semantic_token_ids)
        {
            if (semantic_token_index_ < semantic_token_buffer_size_)
            {
                semantic_token_buffer_[semantic_token_index_] = token_id;
                semantic_token_index_++;
            }
            else
            {
                // Buffer is full, generate audio and reset buffer
                generate_audio();
                // semantic_token_buffer_[0] = token_id; // start new buffer with current token
                // semantic_token_index_ = 1;

                // preserve tokens
                for (size_t i = 0; i < overlapped_token * 2; ++i)
                {
                    // copy trailing tokens to the start of the buffer
                    semantic_token_buffer_[i] = semantic_token_buffer_[semantic_token_buffer_size_ - (overlapped_token * 2) + i];
                }
                semantic_token_index_ = overlapped_token * 2;
                semantic_token_buffer_[semantic_token_index_] = token_id; // add current token
                semantic_token_index_++;
            }
        }

        return ov::genai::StreamingStatus::RUNNING;
    }

    void LLMStreamer::end()
    {
        generate_audio();
    }
} // namespace spark_tts
