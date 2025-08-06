#include "tokenizer.h"

#include "profiler/profiler.h"

namespace spark_tts
{

    Tokenizer::Tokenizer(const std::string &openvino_tokenizer_path)
    {
        TRACE_EVENT("transformer", "Tokenizer::Tokenizer");

        // Initialize the tokenizer with the provided OpenVINO tokenizer path
        ov_tokenizer_ = new ov::genai::Tokenizer(openvino_tokenizer_path);
    }

    Tokenizer::~Tokenizer()
    {
        if (ov_tokenizer_)
        {
            delete ov_tokenizer_;
            ov_tokenizer_ = nullptr;
        }
    }

    std::vector<llama_token> Tokenizer::tokenize(const std::string &text) const
    {
        TRACE_EVENT("transformer", "Tokenizer::tokenize");

        auto ov_tokens = ov_tokenizer_->encode(text, {{"add_special_tokens", false}});
        std::vector<int64_t> token_ids(
            ov_tokens.input_ids.data<int64_t>(),
            ov_tokens.input_ids.data<int64_t>() + ov_tokens.input_ids.get_size());
        // NOTE: OpenVINO tokenizer uses int64_t for token IDs, while llama_token is typically int32_t.
        std::vector<llama_token> llama_tokens(token_ids.begin(), token_ids.end());
        return llama_tokens;
    }

    std::string Tokenizer::token_to_piece(llama_token token) const
    {
        TRACE_EVENT("transformer", "Tokenizer::token_to_piece");

        return ov_tokenizer_->decode({token});
    }
} // namespace spark_tts
