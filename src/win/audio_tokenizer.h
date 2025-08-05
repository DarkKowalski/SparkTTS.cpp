#pragma once

#include <openvino/openvino.hpp>

#include "../audio_tokenizer.h"

namespace spark_tts
{
    class AudioTokenizer: public IAudioTokenizer
    {
    public:
        AudioTokenizer(ov::Core &core,
                       const std::string &wav2vec_model_path,
                       const std::string &mel_spectrogram_model_path,
                       const std::string &bicodec_tokenizer_model_path,
                       const std::string &device_name);

    public:
        virtual std::array<int32_t, 32> tokenize(const std::vector<float> &mono_audio) override;

    private:
        std::array<float, 16000 * 6> pad_or_trim_audio(const std::vector<float> &mono_audio) const;

    private:
        ov::CompiledModel wav2vec_;
        ov::CompiledModel mel_spectrogram_;
        ov::CompiledModel bicodec_tokenizer_;
    };
}
