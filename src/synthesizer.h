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
                                 const std::string &device_name);

    public:
        std::array<int32_t, 32> extract_voice_features(const std::vector<float> &audio_data);

        void text_to_speech(
            const std::string &text,
            std::array<int32_t, 32> &voice_features,
            const size_t n_sec, // max number of seconds to generate
            TextToSpeechCallback &callback);

    private:
        Transformer::DecodeCallbackAction decode_callback(std::string &semantic_tokens,
                                                          std::array<int32_t, 32> &voice_features,
                                                          TextToSpeechCallback &callback);

    private:
        ov::Core core_;

        std::unique_ptr<AudioTokenizer> audio_tokenizer_;
        std::unique_ptr<AudioDetokenizer> audio_detokenizer_;
        std::unique_ptr<Transformer> transformer_;

        const size_t overlapped_semantic_tokens_ = 2;
        const size_t max_semantic_tokens_per_generation_ = 50;
        std::array<int64_t, 50> semantic_tokens_buffer_;
    };

} // namespace spark_tts
