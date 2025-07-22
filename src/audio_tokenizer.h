#pragma once

#include <array>
#include <vector>
#include <cstdint>

namespace spark_tts
{
    class IAudioTokenizer
    {
    public:
        virtual ~IAudioTokenizer() = default;
        virtual std::array<int32_t, 32> tokenize(const std::vector<float> &mono_audio) = 0;
    };
}
