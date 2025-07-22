#pragma once

#include <openvino/openvino.hpp>

namespace spark_tts
{
    class AudioDetokenizer {
    public:
        AudioDetokenizer(ov::Core& core, const std::string& model_path, const std::string& device_name);

    public:
        // Detokenize semantic tokens to audio
        std::array<float, 16000 * 1> detokenize(const ov::Tensor& semantic_tokens, const ov::Tensor& global_tokens);

    private:
        ov::CompiledModel bicodec_detokenizer_;
    };
} // namespace spark_tts
