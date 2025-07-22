#pragma once

#include <memory>
#include <unordered_map>
#include <string>

#include "../audio_tokenizer.h"

namespace spark_tts
{
    class AudioTokenizerImpl : public IAudioTokenizer
    {
    public:
        AudioTokenizerImpl(const std::string &model_path);
        ~AudioTokenizerImpl() override;

    public:
        // global_tokens
        virtual std::array<int32_t, 32> tokenize(const std::vector<float> &mono_audio) override;

    private:
        std::array<float, 16000 * 6> pad_or_trim_audio(const std::vector<float> &mono_audio) const;

    private:
        void *objc_context_ = nullptr; // Pointer to the CoreML model context for Wav2Vec
    };
}
