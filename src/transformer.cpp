#include "transformer.h"

#include "profiler/profiler.h"

namespace spark_tts
{
    Transformer::Transformer(const std::string &model_path,
                             const std::string &tokenizer_path,
                             const Params params)
        : ctx_params_(params.ctx_params),
          model_params_(params.model_params),
          sampler_params_(params.sampler_params)
    {
        TRACE_EVENT("transformer", "Transformer::Transformer");

        {
            TRACE_EVENT("transformer", "llama_model_load_from_file");
            model_ = llama_model_load_from_file(model_path.c_str(), model_params_);
            if (!model_)
            {
                throw std::runtime_error("Failed to load model from file: " + model_path);
            }
        }

        {
            TRACE_EVENT("transformer", "llama_model_get_vocab");
            vocab_ = llama_model_get_vocab(model_);
            if (!vocab_)
            {
                throw std::runtime_error("Failed to get vocabulary from model");
            }
        }

        // Initialize the context
        {
            TRACE_EVENT("transformer", "llama_init_from_model");
            ctx_ = llama_init_from_model(model_, ctx_params_);
            if (!ctx_)
            {
                throw std::runtime_error("Failed to initialize context from model");
            }

            if (llama_n_ctx(ctx_) > llama_model_n_ctx_train(model_))
            {
                throw std::runtime_error("Context size exceeds model's training context size");
            }
        }

        // Don't use llama.cpp tokenizer, use OpenVINO tokenizer instead
        tokenizer_ = new Tokenizer(tokenizer_path);

        sampler_ = new Sampler(sampler_params_, model_);
    }

    Transformer::~Transformer()
    {
        if (sampler_)
        {
            delete sampler_;
            sampler_ = nullptr;
        }

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

    bool Transformer::infer(const std::string &prompt,
                            const size_t n_predict,
                            const size_t callback_tokens,
                            const size_t first_callback_tokens,
                            DecodeCallback &callback)
    {
        TRACE_EVENT("transformer", "Transformer::infer");

        sampler_->reset();
        llama_memory_clear(llama_get_memory(ctx_), true);

        size_t n_total = 0;
        size_t n_callback = 0;
        bool first_callback_executed = false;

        std::string callback_buffer;
        constexpr size_t each_token_size = 25; // Approximate size of each token in characters
        callback_buffer.reserve(callback_tokens * each_token_size);

        auto input_tokens = tokenizer_->tokenize(prompt);
        llama_batch batch = llama_batch_get_one(input_tokens.data(), input_tokens.size());

        bool end_of_generation = false;

        while (n_total < n_predict)
        {
            TRACE_EVENT_BEGIN("transformer", "llama_decode");
            int32_t decode_result = llama_decode(ctx_, batch);
            TRACE_EVENT_END("transformer");
            if (decode_result != 0)
            {
                throw std::runtime_error("Decoding failed with error code: " + std::to_string(decode_result));
            }

            llama_token new_token = sampler_->sample(ctx_, -1, false);
            sampler_->accept(new_token, false);
            if (llama_vocab_is_eog(vocab_, new_token))
            {
                end_of_generation = true;
                break;
            }

            auto token = tokenizer_->token_to_piece(new_token);
            callback_buffer += token;
            n_callback++;
            n_total++;

            if (first_callback_executed && callback_tokens <= n_callback)
            {
                auto action = callback(callback_buffer);

                callback_buffer.clear();
                n_callback = 0;

                if (action == DecodeCallbackAction::Stop)
                {
                    break;
                }
            }
            else if (!first_callback_executed && first_callback_tokens <= n_callback)
            {
                auto action = callback(callback_buffer);
                first_callback_executed = true;

                callback_buffer.clear();
                n_callback = 0;

                if (action == DecodeCallbackAction::Stop)
                {
                    break;
                }
            }

            batch = llama_batch_get_one(&new_token, 1);
        }

        // callback remaining tokens
        if (!callback_buffer.empty())
        {
            callback(callback_buffer);
        }

        return end_of_generation;
    }

} // namespace spark_tts
