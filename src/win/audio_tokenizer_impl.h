#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/cpu_provider_factory.h>

#include "../audio_tokenizer.h"

namespace spark_tts
{
    class AudioTokenizerImpl : public IAudioTokenizer
    {
    public:
        AudioTokenizerImpl(const std::string &model_path);

    public:
        virtual std::array<int32_t, 32> tokenize(const std::vector<float> &mono_audio) override;

    private:
        std::array<float, 16000 * 6> pad_or_trim_audio(const std::vector<float> &mono_audio) const;

    private:
        Ort::Env env_;
        Ort::MemoryInfo memory_info_;
        std::unique_ptr<Ort::Session> audio_tokenizer_session_;

        const std::array<const char *, 1> audio_tokenizer_input_names_ = {"audio_input"};
        const std::array<const char *, 2> audio_tokenizer_output_names_ = {"semantic_tokens", "global_tokens"};
        const std::array<int64_t, 1> audio_tokenizer_input_shape_ = {96000};
        const std::array<int64_t, 2> audio_tokenizer_output_semantic_tokens_shape_ = {1, 299};
        const std::array<int64_t, 3> audio_tokenizer_output_global_tokens_shape_ = {1, 1, 32};

        std::array<float, 16000 * 6> audio_input_data_;
        std::array<int64_t, 299> semantic_tokens_data_;
        std::array<int32_t, 32> global_tokens_data_;

        Ort::Value input_tensor_;
        std::array<Ort::Value, 2> output_tensors_;

    };
}
