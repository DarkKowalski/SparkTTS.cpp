#include "transformer.h"

#include <iostream>
#include <chrono>

namespace spark_tts
{
    Transformer::Transformer(const std::string &model_path,
                             std::function<void(std::string &)> on_decoded)
        : on_decoded_(on_decoded)
    {
        ctx_params_ = llama_context_default_params();
        model_params_ = llama_model_default_params();
        // sampler_params_.top_k = 50;
        // sampler_params_.top_p = 0.95f;
        // sampler_params_.temp = 0.8f;
        ctx_params_.n_ctx = 2048;
        ctx_params_.n_batch = 1;
        ctx_params_.n_threads = 4;
        ctx_params_.n_seq_max = 1;
        ctx_params_.n_ubatch = 1;
        ctx_params_.no_perf = true;
        // ctx_params_.flash_attn = true; // Enable flash attention

        std::vector<ggml_backend_dev_t> devices = {
            // ggml_backend_dev_by_name("CPU"), // Add the CPU device
            ggml_backend_dev_by_name("Vulkan0"), // Add the first device (usually CPU)
            // ggml_backend_dev_by_name("CPU"), // Add the second device (if available)
            nullptr // add nullptr to the end of the list
        };
        model_params_.devices = devices.data();
        model_params_.n_gpu_layers = 25;
        model_params_.split_mode = LLAMA_SPLIT_MODE_NONE;

        // LLM_ARCH_QWEN2,
        // {
        //     { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        //     { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
        //     { LLM_TENSOR_OUTPUT,          "output" },
        //     { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
        //     { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
        //     { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
        //     { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
        //     { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
        //     { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
        //     { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
        //     { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        //     { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        // },

        // Override tensor offloading
        // std::map<std::string, ggml_backend_buffer_type_t> buft_list;
        // if (buft_list.empty())
        // {
        //     // enumerate all the devices and add their buffer types to the list
        //     for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        //     {
        //         auto *dev = ggml_backend_dev_get(i);
        //         auto *buft = ggml_backend_dev_buffer_type(dev);
        //         if (buft)
        //         {
        //             buft_list[ggml_backend_buft_name(buft)] = buft;
        //             std::cout << "Buffer type: " << ggml_backend_buft_name(buft) << std::endl;
        //         }
        //     }
        // }

        // Initialize the model
        model_ = llama_load_model_from_file(model_path.c_str(), model_params_);
        if (!model_)
        {
            throw std::runtime_error("Failed to load model from file: " + model_path);
        }

        vocab_ = llama_model_get_vocab(model_);
        if (!vocab_)
        {
            throw std::runtime_error("Failed to get vocabulary from model");
        }

        ctx_ = llama_init_from_model(model_, ctx_params_);
        if (!ctx_)
        {
            throw std::runtime_error("Failed to initialize context from model");
        }

        if (llama_n_ctx(ctx_) > llama_model_n_ctx_train(model_))
        {
            throw std::runtime_error("Context size exceeds model's training context size");
        }

        tokenizer_ = new Tokenizer("./models/Spark-TTS-0.5B/LLM/");
    }

    Transformer::~Transformer()
    {
        if (tokenizer_)
        {
            delete tokenizer_;
            tokenizer_ = nullptr;
        }

        if (ctx_)
        {
            llama_free(ctx_);
            ctx_ = nullptr;
        }

        if (model_)
        {
            llama_model_free(model_);
            model_ = nullptr;
        }
    }

    void Transformer::infer(const std::string &prompt, const size_t n_predict, const size_t callback_tokens)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        auto input_tokens = tokenizer_->tokenize(prompt);
        llama_batch batch = llama_batch_get_one(input_tokens.data(), input_tokens.size());
        size_t generated_tokens = 0;
        while (true)
        {
            int32_t decode_result = llama_decode(ctx_, batch);
            if (decode_result != 0)
            {
                std::cerr << "Decoding failed with error code: " << decode_result << std::endl;
                break;
            }

            const float *logits = llama_get_logits_ith(ctx_, -1);
            // std::cout << "Logit for token 0: " << logits[0] << std::endl;
            // should not ne NaN
            if (std::isnan(logits[0]))
            {
                std::cerr << "Logit is NaN, decoding failed." << std::endl;
                break;
            }

            llama_token new_token = llama_sampler_sample(smpl, ctx_, -1);
            // std::cout << "Sampled token: " << new_token << std::endl;

            if (llama_vocab_is_eog(vocab_, new_token))
            {
                break;
            }

            std::string decoded_text = tokenizer_->token_to_piece(new_token);
            on_decoded_(decoded_text);
            generated_tokens++;

            batch = llama_batch_get_one(&new_token, 1);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Inference completed in " << duration << " ms" << std::endl;
        std::cout << "Generated tokens: " << generated_tokens << std::endl;
    }

} // namespace spark_tts
