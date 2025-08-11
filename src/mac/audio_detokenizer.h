#pragma once

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
        ~AudioDetokenizer() override;

    public:
        // Detokenize semantic tokens to audio
        virtual std::array<float, 16000 * 1> detokenize(std::array<int64_t, 50> &semantic_tokens,
                                                        std::array<int32_t, 32> &global_tokens) override;

    private:
        void *objc_context_ = nullptr; // Pointer to the CoreML model context
    };
} // namespace spark_tts
