#pragma once

#include <llama-cpp.h>
#include <tokenizers_cpp.h>

#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>

namespace spark_tts
{
    class Tokenizer
    {
    public:
        Tokenizer(const std::string &huggingface_tokenizer_path);

    public:
        std::vector<llama_token> tokenize(const std::string &text) const;

        std::string token_to_piece(llama_token token) const;

    private:
        std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    };

} // namespace spark_tts
