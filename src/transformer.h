#pragma once

#include <llama-cpp.h>

#include <vector>
#include <string>
#include <functional>

#include "sampler.h"
#include "tokenizer.h"

namespace spark_tts
{
    class Transformer
    {
    public:
        Transformer(const std::string &model_path,
                    std::function<void(std::string&)> on_decoded);
        ~Transformer();

    public:
        void infer(const std::string &prompt, const size_t n_predict, const size_t callback_tokens);

    private:
        llama_context *ctx_;
        llama_model *model_;
        const llama_vocab *vocab_;

        llama_context_params ctx_params_;
        llama_model_params model_params_;
        SamplerParameters sampler_params_;

        Tokenizer *tokenizer_;

        std::function<void(std::string&)> on_decoded_;
    };
}
