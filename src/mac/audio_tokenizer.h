#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/cpu_provider_factory.h>

#include <memory>
#include <unordered_map>
#include <string>

#include "../audio_tokenizer.h"

namespace spark_tts
{
    class AudioTokenizer : public IAudioTokenizer
    {
    public:
        AudioTokenizer(const std::string &wav2vec_model_path,
                       const std::string &mel_spectrogram_model_path,
                       const std::string &bicodec_tokenizer_model_path);

    public:
        // global_tokens
        virtual std::array<int32_t, 32> tokenize(const std::vector<float> &mono_audio) override;

    private:
        std::array<float, 16000 * 6> pad_or_trim_audio(const std::vector<float> &mono_audio) const;

    private:
        Ort::Env env_;
        Ort::MemoryInfo memory_info_;
        std::unique_ptr<Ort::Session> wav2vec_session_;
        std::unique_ptr<Ort::Session> mel_spectrogram_session_;
        std::unique_ptr<Ort::Session> bicodec_tokenizer_session_;

        const std::array<const char *, 1> mel_input_names_ = {"mel_input"};
        const std::array<const char *, 1> mel_output_names_ = {"mel_output"};
        const std::array<int64_t, 3> mel_input_shape_ = {1, 1, 96000};
        const std::array<int64_t, 3> mel_output_shape_ = {1, 128, 302};

        const std::array<const char *, 1> wav2vec_input_names_ = {"feat_input"};
        const std::array<const char *, 1> wav2vec_output_names_ = {"feat_output"};
        const std::array<int64_t, 2> wav2vec_input_shape_ = {1, 96000};
        const std::array<int64_t, 3> wav2vec_output_shape_ = {1, 299, 1024};

        const std::array<const char *, 2> bicodec_input_names_ = {"feat", "mel"};
        const std::array<const char *, 2> bicodec_output_names_ = {"semantic_tokens", "global_tokens"};
        // feat: wav2vec_output
        // mel: mel_output
        const std::array<int64_t, 2> bicodec_output_semantic_tokens_shape_ = {1, 299};
        const std::array<int64_t, 3> bicodec_output_global_tokens_shape_ = {1, 1, 32};
    };
}
