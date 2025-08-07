#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/cpu_provider_factory.h>

#include <memory>
#include <unordered_map>
#include <string>

#include "../audio_detokenizer.h"

namespace spark_tts
{
    class AudioDetokenizer : public IAudioDetokenizer
    {
    public:
        AudioDetokenizer(const std::string &model_path);

    public:
        // Detokenize semantic tokens to audio
        virtual std::array<float, 16000 * 1> detokenize(std::array<int64_t, 50> &semantic_tokens,
                                                        std::array<int32_t, 32> &global_tokens) override;

    private:
        Ort::Env env_;
        Ort::MemoryInfo memory_info_;
        std::unique_ptr<Ort::Session> bicodec_detokenizer_session_;

        const std::array<const char *, 2> bicodec_input_names_ = {"semantic_tokens", "global_tokens"};
        const std::array<const char *, 1> bicodec_output_names_ = {"wav_recon"};
        const std::array<int64_t, 2> bicodec_input_semantic_tokens_shape_ = {1, 50};
        const std::array<int64_t, 3> bicodec_input_global_tokens_shape_ = {1, 1, 32};
        const std::array<int64_t, 3> bicodec_output_wav_recon_shape_ = {1, 1, 16000};
    };
} // namespace spark_tts
