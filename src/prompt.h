#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <array>

namespace spark_tts
{
    enum class PromptGender : uint8_t
    {
        kFemale = 0,
        kMale = 1,
    };

    enum class PromptLevel : uint8_t
    {
        kVeryLow = 0,
        kLow = 1,
        kModerate = 2,
        kHigh = 3,
        kVeryHigh = 4,
    };

    std::string stringify_global_tokens(const std::array<int32_t, 32> &global_tokens);

    std::string assemble_prompt(const std::string &global_token_input, const std::string &text);

    std::vector<int64_t> extract_semantic_token_ids(const std::string &semantic_tokens_string);
}
