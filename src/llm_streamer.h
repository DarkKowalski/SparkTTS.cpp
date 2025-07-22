#include <openvino/openvino.hpp>
#include <openvino/genai/streamer_base.hpp>
#include <openvino/genai/tokenizer.hpp>

#include <array>
#include <memory>
#include <functional>
#include <vector>

#include "audio_detokenizer.h"
#include "prompt.h"

namespace spark_tts
{
    class LLMStreamer : public ov::genai::StreamerBase
    {
    public:
        LLMStreamer(ov::genai::Tokenizer tokenizer,
                    std::shared_ptr<AudioDetokenizer> audio_detokenizer,
                    std::function<void(std::vector<float> &)> on_audio_generated)
            : tokenizer_(std::move(tokenizer)),
              audio_detokenizer_(std::move(audio_detokenizer)),
              on_audio_generated_(std::move(on_audio_generated)),
              global_tokens_({ov::element::i32, {1, 1, 32}}),
              semantic_token_buffer_({}),
              semantic_token_index_(0),
              semantic_token_buffer_size_(50) // 1 second of audio at 16kHz
        {
        }

        virtual ov::genai::StreamingStatus write(const std::vector<int64_t> &tokens) override;

        virtual void end() override;

    public:
        void set_global_tokens(const ov::Tensor &global_tokens)
        {
            global_tokens_ = global_tokens;
        }

        void generate_audio();

    private:
        ov::genai::Tokenizer tokenizer_;
        std::shared_ptr<AudioDetokenizer> audio_detokenizer_;
        std::function<void(std::vector<float> &)> on_audio_generated_;
        ov::Tensor global_tokens_;

        std::array<int64_t, 16000 * 1> semantic_token_buffer_;
        size_t semantic_token_index_;
        const size_t semantic_token_buffer_size_;
    };
} // namespace spark_tts
