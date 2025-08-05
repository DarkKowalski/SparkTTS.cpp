#pragma once

#include <openvino/openvino.hpp>

#include "../audio_detokenizer.h"

namespace spark_tts
{
    class AudioDetokenizer : public IAudioDetokenizer
    {
    public:
        AudioDetokenizer(ov::Core &core, const std::string &model_path, const std::string &device_name);

    public:
        virtual std::array<float, 16000 * 1> detokenize(std::array<int64_t, 50> &semantic_tokens,
                                                        std::array<int32_t, 32> &global_tokens) override;

    private:
        ov::CompiledModel bicodec_detokenizer_;
    };
} // namespace spark_tts
