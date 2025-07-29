#include "sampler.h"

namespace spark_tts
{
    Sampler::Sampler(const SamplerParameters &params, const llama_model *model)
        : params_(params), prev_tokens_(std::max(32, params.n_prev))
    {
        const llama_vocab *vocab = llama_model_get_vocab(model);
        grammar_ = llama_sampler_init_grammar(vocab, params_.grammar.c_str(), "root");
        if (!grammar_)
        {
            throw std::runtime_error("Failed to create grammar");
        }

        llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
        sampler_params.no_perf = params_.no_perf;
        chain_ = llama_sampler_chain_init(sampler_params);
        if (!chain_)
        {
            throw std::runtime_error("Failed to create sampler chain");
        }

        auto logit_bias = llama_sampler_init_logit_bias(
            llama_vocab_n_tokens(vocab),
            params_.logit_bias.size(),
            params_.logit_bias.data());
        llama_sampler_chain_add(chain_, logit_bias);

        if (params_.mirostat == 0)
        {
            if (params_.top_n_sigma >= 0)
            {
                llama_sampler_chain_add(chain_, llama_sampler_init_top_k(params_.top_k));
                llama_sampler_chain_add(chain_, llama_sampler_init_temp(params_.temp));
                llama_sampler_chain_add(chain_, llama_sampler_init_top_n_sigma(params_.top_n_sigma));
            }
            else // mirostat == 0 && top_n_sigma < 0
            {
                for (const auto &sampler : params_.samplers)
                {
                    switch (sampler)
                    {
                    case SamplerType::Dry:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params_.dry_sequence_breakers.size());
                        for (const auto &str : params_.dry_sequence_breakers)
                        {
                            c_breakers.push_back(str.c_str());
                        }

                        auto dry = llama_sampler_init_dry(vocab,
                                                          llama_model_n_ctx_train(model),
                                                          params_.dry_multiplier,
                                                          params_.dry_base,
                                                          params_.dry_allowed_length,
                                                          params_.dry_penalty_last_n,
                                                          c_breakers.data(),
                                                          c_breakers.size());
                        llama_sampler_chain_add(chain_, dry);
                    }
                    break;
                    case SamplerType::TopK:
                        llama_sampler_chain_add(chain_, llama_sampler_init_top_k(params_.top_k));
                        break;
                    case SamplerType::TopP:
                        llama_sampler_chain_add(chain_, llama_sampler_init_top_p(params_.top_p, params_.min_keep));
                        break;
                    case SamplerType::MinP:
                        llama_sampler_chain_add(chain_, llama_sampler_init_min_p(params_.min_p, params_.min_keep));
                        break;
                    case SamplerType::Xtc:
                        llama_sampler_chain_add(chain_, llama_sampler_init_xtc(params_.xtc_probability, params_.xtc_threshold, params_.min_keep, params_.seed));
                        break;
                    case SamplerType::TypicalP:
                        llama_sampler_chain_add(chain_, llama_sampler_init_typical(params_.typ_p, params_.min_keep));
                        break;
                    case SamplerType::Temperature:
                        llama_sampler_chain_add(chain_, llama_sampler_init_temp_ext(params_.temp, params_.dynatemp_range, params_.dynatemp_exponent));
                        break;
                    case SamplerType::Infill:
                        llama_sampler_chain_add(chain_, llama_sampler_init_infill(vocab));
                        break;
                    case SamplerType::Penalties:
                        llama_sampler_chain_add(chain_, llama_sampler_init_penalties(params_.penalty_last_n, params_.penalty_repeat, params_.penalty_freq, params_.penalty_present));
                        break;
                    default:
                        throw std::runtime_error("Unknown sampler type");
                    }
                } // End of for loop
            } // End of if-else

            llama_sampler_chain_add(chain_, llama_sampler_init_dist(params_.seed));
        }
        else if (params_.mirostat == 1)
        {
            llama_sampler_chain_add(chain_, llama_sampler_init_temp(params_.temp));
            llama_sampler_chain_add(chain_, llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params_.seed, params_.mirostat_tau, params_.mirostat_eta, 100));
        }
        else if (params_.mirostat == 2)
        {
            llama_sampler_chain_add(chain_, llama_sampler_init_temp(params_.temp));
            llama_sampler_chain_add(chain_, llama_sampler_init_mirostat_v2(params_.seed, params_.mirostat_tau, params_.mirostat_eta));
        }
        else
        {
            throw std::runtime_error("Unknown mirostat version");
        }
    }

    Sampler::~Sampler()
    {
        if (chain_)
        {
            llama_sampler_free(chain_);
            chain_ = nullptr;
        }

        if (grammar_)
        {
            llama_sampler_free(grammar_);
            grammar_ = nullptr;
        }
    }

    void Sampler::reset()
    {
        llama_sampler_reset(chain_);
        llama_sampler_reset(grammar_);
    }

    void Sampler::accept(llama_token token, bool accept_grammar)
    {
        if (accept_grammar)
        {
            llama_sampler_accept(grammar_, token);
        }

        llama_sampler_accept(chain_, token);
        prev_tokens_.push_back(token);
    }

    void Sampler::set_logits(llama_context *ctx, int32_t idx)
    {
        const auto *logits = llama_get_logits_ith(ctx, idx);
        if (logits == nullptr)
        {
            throw std::runtime_error("Failed to get logits");
        }

        const llama_model *model = llama_get_model(ctx);
        const llama_vocab *vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);
        cur_.resize(n_vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++)
        {
            cur_[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }

        cur_p_ = {cur_.data(), cur_.size(), -1, false};
    }

    llama_token Sampler::sample(llama_context *ctx, int32_t idx, bool grammar_first)
    {
        set_logits(ctx, idx);

        if (grammar_first)
        {
            llama_sampler_apply(grammar_, &cur_p_);
        }

        llama_sampler_apply(chain_, &cur_p_);

        if (cur_p_.selected == -1)
        {
            throw std::runtime_error("no selected token during sampling - check your sampling configuration");
        }

        const llama_token id = cur_p_.data[cur_p_.selected].id;
        if (grammar_first)
        {
            return id;
        }

        // check if it the sampled token fits the grammar
        {
            llama_token_data single_token_data = {id, 1.0f, 0.0f};
            llama_token_data_array single_token_data_array = {&single_token_data, 1, -1, false};

            llama_sampler_apply(grammar_, &single_token_data_array);

            const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
            if (is_valid)
            {
                return id;
            }
        }

        // resampling:
        // if the token is not valid, sample again,
        // but first apply the grammar sampler and then the sampling chain
        set_logits(ctx, idx);
        llama_sampler_apply(grammar_, &cur_p_);
        llama_sampler_apply(chain_, &cur_p_);
        if (cur_p_.selected == -1)
        {
            throw std::runtime_error("no selected token during re-sampling - check your sampling configuration");
        }

        return cur_p_.data[cur_p_.selected].id;
    }
} // namespace spark_tts
