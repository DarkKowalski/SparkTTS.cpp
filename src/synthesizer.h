#pragma once

#include <llama-cpp.h>

#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <array>

#include "transformer.h"
#include "prompt.h"
#include "token_buffer.h"

#include "audio_tokenizer.h"
#include "audio_detokenizer.h"

namespace spark_tts
{

    class Synthesizer
    {
    public:
        typedef std::function<bool(std::vector<float> &)> TextToSpeechCallback; // true to continue, false to stop

    public:
        Synthesizer();
        ~Synthesizer();

        // Lazy initialization
    public:
        void init_voice_feature_extraction(const std::string &audio_tokenizer_model_path);

        void init_text_to_speech(const std::string &audio_detokenizer_model_path,
                                 const std::string &transformer_model_path,
                                 const std::string &tokenizer_path,
                                 const uint32_t transformer_n_ctx,
                                 const size_t overlapped_semantic_tokens,
                                 const size_t callback_semantic_tokens);

    public:
        std::array<int32_t, 32> extract_voice_features(const std::vector<float> &audio_data);

        void text_to_speech(
            const std::string &text,
            std::array<int32_t, 32> &voice_features,
            const size_t n_sec, // max number of seconds to generate
            const bool drop_last,
            TextToSpeechCallback &callback);

    private:
        Transformer::DecodeCallbackAction decode_callback(std::string &semantic_tokens,
                                                          std::array<int32_t, 32> &voice_features,
                                                          TextToSpeechCallback &callback);

        std::vector<float> synthesize(std::array<int32_t, 32> &voice_features);

    private:
        std::unique_ptr<IAudioTokenizer> audio_tokenizer_;
        std::unique_ptr<IAudioDetokenizer> audio_detokenizer_;
        std::unique_ptr<Transformer> transformer_;
        std::unique_ptr<TokenBuffer> token_buffer_;

        size_t overlapped_semantic_tokens_; // Number of tokens to overlap between generations
                                            // Tradeoff between quality and throughput
                                            // 0 to 25, 3 to 5 is good for most cases

        size_t callback_semantic_tokens_; // Number of tokens to trigger callback, 0 for immediate callback
                                          // Tradeoff between latency and throughput
                                          // 0 to 50

        size_t synthesized_frames_; // Number of frames synthesized for the current text
    };

} // namespace spark_tts
