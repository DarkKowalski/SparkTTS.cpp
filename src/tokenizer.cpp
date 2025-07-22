#include "tokenizer.h"

#include "profiler/profiler.h"

namespace spark_tts
{
    static std::string load_bytes_from_file(const std::string &path)
    {
        std::ifstream fs(path, std::ios::in | std::ios::binary);
        if (fs.fail())
        {
            throw std::runtime_error("Cannot open file: " + path);
        }
        std::string data;
        fs.seekg(0, std::ios::end);
        size_t size = static_cast<size_t>(fs.tellg());
        fs.seekg(0, std::ios::beg);
        data.resize(size);
        fs.read(data.data(), size);
        return data;
    }

    Tokenizer::Tokenizer(const std::string &huggingface_tokenizer_path)
    {
        TRACE_EVENT("transformer", "Tokenizer::Tokenizer");

        auto blob = load_bytes_from_file(huggingface_tokenizer_path);
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
    }

    std::vector<llama_token> Tokenizer::tokenize(const std::string &text) const
    {
        TRACE_EVENT("transformer", "Tokenizer::tokenize");

        auto token_ids = tokenizer_->Encode(text);
        // NOTE: OpenVINO tokenizer uses int64_t for token IDs, while llama_token is typically int32_t.
        std::vector<llama_token> llama_tokens(token_ids.begin(), token_ids.end());
        return llama_tokens;
    }

    std::string Tokenizer::token_to_piece(llama_token token) const
    {
        TRACE_EVENT("transformer", "Tokenizer::token_to_piece");

        return tokenizer_->Decode({token});
    }
} // namespace spark_tts
