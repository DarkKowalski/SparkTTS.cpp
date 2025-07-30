#pragma once

#include <llama-cpp.h>
#include <openvino/openvino.hpp>

#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <array>

#include "audio_tokenizer.h"
#include "audio_detokenizer.h"
#include "transformer.h"
#include "prompt.h"
#include "token_buffer.h"

namespace spark_tts
{

    class Synthesizer
    {
    public:
        typedef std::function<bool(std::vector<float> &)> TextToSpeechCallback; // true to continue, false to stop

    public:
        Synthesizer() : core_(ov::Core())
        {
            llama_backend_init(); // Initialize llama backend
        }

        ~Synthesizer()
        {
            llama_backend_free(); // Free llama backend resources
        }

        // Lazy initialization
    public:
        void init_voice_feature_extraction(const std::string &wav2vec_model_path,
                                           const std::string &mel_spectrogram_model_path,
                                           const std::string &bicodec_tokenizer_model_path,
                                           const std::string &device_name);

        void init_text_to_speech(const std::string &audio_detokenizer_model_path,
                                 const std::string &transformer_model_path,
                                 const std::string &tokenizer_path,
                                 const uint32_t transformer_n_ctx,
                                 const size_t overlapped_semantic_tokens,
                                 const size_t callback_semantic_tokens,
                                 const std::string &device_name);

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

        bool synthesize(const std::array<int32_t, 32> &voice_features, std::vector<float> &generated_audio);

    private:
        ov::Core core_;

        std::unique_ptr<AudioTokenizer> audio_tokenizer_;
        std::unique_ptr<AudioDetokenizer> audio_detokenizer_;
        std::unique_ptr<Transformer> transformer_;
        std::unique_ptr<TokenBuffer> token_buffer_;

        bool first_sample_generated_;       // Flag to indicate if the first sample has been generated
        size_t overlapped_semantic_tokens_; // Number of tokens to overlap between generations
        size_t callback_semantic_tokens_;   // Number of tokens to trigger callback, 0 for immediate callback
    };

} // namespace spark_tts
