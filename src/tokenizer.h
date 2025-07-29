#pragma once

#include <llama-cpp.h>
#include <openvino/openvino.hpp>
#include <openvino/genai/tokenizer.hpp>

#include <string>
#include <vector>
#include <stdexcept>

namespace spark_tts
{
    class Tokenizer
    {
    public:
        Tokenizer(const std::string& openvino_tokenizer_path);
        ~Tokenizer();

    public:
        std::vector<llama_token> tokenize(const std::string &text) const;

        std::string token_to_piece(llama_token token) const;

    private:
        ov::genai::Tokenizer* ov_tokenizer_; // OpenVINO tokenizer instance
    };

} // namespace spark_tts
