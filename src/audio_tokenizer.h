#pragma once

#include <openvino/openvino.hpp>

#include <array>
#include <vector>
#include <tuple>

namespace spark_tts {
    class AudioTokenizer {
    public:
        AudioTokenizer(ov::Core& core, const std::string& model_path, const std::string& device_name);

    public:
        // semantic_tokens, global_tokens
        std::pair<ov::Tensor, ov::Tensor> tokenize(const std::vector<float>& mono_audio);

    private:
        std::array<float, 16000 * 6> pad_or_trim_audio(const std::vector<float>& mono_audio) const;

    private:
        ov::CompiledModel wav2vec_;
        ov::CompiledModel mel_spectrogram_;
        ov::CompiledModel bicodec_tokenizer_;
    };
}
