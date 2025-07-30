#include "prompt.h"

#include <sstream>
#include <regex>

namespace spark_tts
{
    std::string stringify_global_tokens(const std::array<int32_t, 32> &global_tokens)
    {
        std::ostringstream oss;
        for (size_t i = 0; i < global_tokens.size(); ++i)
        {
            oss << "<|bicodec_global_" << global_tokens[i] << "|>";
        }
        return oss.str();
    }

    std::string stringify_global_tokens(const ov::Tensor &global_tokens)
    {
        std::ostringstream oss;
        auto global_token_ids = global_tokens.data<int32_t>();
        for (size_t i = 0; i < global_tokens.get_size(); ++i)
        {
            oss << "<|bicodec_global_" << global_token_ids[i] << "|>";
        }

        return oss.str();
    }

    std::string assemble_prompt(const std::string &global_token_input, const std::string &text)
    {
        std::ostringstream oss;
        oss << "<|task_tts|>"
            << "<|start_content|>"
            << text
            << "<|end_content|>"
            << "<|start_global_token|>"
            << global_token_input
            << "<|end_global_token|>";
        return oss.str();
    }

    std::vector<int64_t> extract_semantic_token_ids(const std::string &semantic_tokens_string)
    {
        // <|bicodec_semantic_3711|><|bicodec_semantic_3711|> to [3711, 3711]
        std::vector<int64_t> ids;
        std::regex regex(R"(<\|bicodec_semantic_(\d+)\|>)");
        std::sregex_iterator it(semantic_tokens_string.begin(), semantic_tokens_string.end(), regex);
        std::sregex_iterator end;
        while (it != end)
        {
            std::string match = (*it)[1].str();
            try
            {
                int64_t id = std::stoll(match);
                ids.push_back(id);
            }
            catch (const std::invalid_argument &)
            {
                break; // Invalid token ID
            }

            it++;
        }
        return ids;
    }
} // namespace spark_tts
