#pragma once

#include <llama-cpp.h>

#include <cstdint>
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
        struct Params
        {
            Params()
            {
                ctx_params = llama_context_default_params();
                model_params = llama_model_default_params();
                sampler_params = SamplerParameters();

                ctx_params.n_ctx = 2048;
                ctx_params.n_ubatch = 1;      // NVIDIA RTX4070 Vulkan backend will generate NaN logits with 512 n_ubatch.
                ctx_params.no_perf = true;    // Disable performance metrics
                ctx_params.flash_attn = true; // Enable flash attention

                model_params.n_gpu_layers = 25; // Offload all layers to GPU

                sampler_params.top_k = 50;    // Set top_k for sampling
                sampler_params.top_p = 0.95f; // Set top_p for sampling
                sampler_params.temp = 0.8f;   // Set temperature for sampling
            }

            llama_context_params ctx_params; // parameters for the context
            llama_model_params model_params; // parameters for the model
            SamplerParameters sampler_params;
        };

    public:
        enum class DecodeCallbackAction : uint8_t
        {
            Continue, // Continue decoding
            Stop,     // Stop decoding
        };
        typedef std::function<DecodeCallbackAction(std::string &)> DecodeCallback;

    public:
        Transformer(const std::string &model_path,
                    const std::string &tokenizer_path,
                    const Params params);
        ~Transformer();

    public:
        void infer(const std::string &prompt,
                   const size_t n_predict,       // max number of tokens to generate
                   const size_t callback_tokens, // number of tokens to trigger callback, 0 for immediate callback
                   DecodeCallback &callback);

    private:
        llama_context *ctx_;
        llama_model *model_;
        const llama_vocab *vocab_;

        llama_context_params ctx_params_;
        llama_model_params model_params_;
        SamplerParameters sampler_params_;

        Tokenizer *tokenizer_;
        Sampler *sampler_;
    };
}
