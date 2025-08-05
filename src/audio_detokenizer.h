#pragma once

#include <array>
#include <cstdint>

namespace spark_tts
{
    class IAudioDetokenizer
    {
    public:
        virtual ~IAudioDetokenizer() = default;
        // Detokenize semantic tokens to audio
        virtual std::array<float, 16000 * 1> detokenize(std::array<int64_t, 50> &semantic_tokens,
                                                        std::array<int32_t, 32> &global_tokens) = 0;
    };
} // namespace spark_tts
