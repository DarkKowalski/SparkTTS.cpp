#pragma once

// Basically copy-pasted from llama-cpp common/sampling.h and common/sampling.cpp
// But in a RAII style

#include <llama-cpp.h>

#include <cstdint>
#include <string>
#include <vector>
#include <set>
#include <stdexcept>

#include "ring_buffer.hpp"

namespace spark_tts
{
    enum class SamplerType : uint32_t
    {
        None = 0,
        Dry = 1,
        TopK = 2,
        TopP = 3,
        MinP = 4,
        // TfsZ = 5,
        TypicalP = 6,
        Temperature = 7,
        Xtc = 8,
        Infill = 9,
        Penalties = 10,
    };

    static std::vector<std::string> valid_sampler_types()
    {
        return {
            "dry",
            "topk",
            "topp",
            "minp",
            // "tfsz",
            "typicalp",
            "temperature",
            "xtc",
            "infill",
            "penalties",
        };
    }

    static SamplerType sampler_type_from_string(const std::string &str)
    {
        if (str == "dry")
            return SamplerType::Dry;
        else if (str == "topk")
            return SamplerType::TopK;
        else if (str == "topp")
            return SamplerType::TopP;
        else if (str == "minp")
            return SamplerType::MinP;
        else if (str == "typicalp")
            return SamplerType::TypicalP;
        else if (str == "temperature")
            return SamplerType::Temperature;
        else if (str == "xtc")
            return SamplerType::Xtc;
        else if (str == "penalties")
            return SamplerType::Penalties;

        throw std::invalid_argument("Invalid sampler type: " + str);
    }

    static std::string sampler_type_to_string(const SamplerType type)
    {
        switch (type)
        {
        case SamplerType::Dry:
            return "dry";
        case SamplerType::TopK:
            return "topk";
        case SamplerType::TopP:
            return "topp";
        case SamplerType::MinP:
            return "minp";
        case SamplerType::TypicalP:
            return "typicalp";
        case SamplerType::Temperature:
            return "temperature";
        case SamplerType::Xtc:
            return "xtc";
        case SamplerType::Penalties:
            return "penalties";
        default:
            throw std::invalid_argument("Invalid sampler type");
        }
    }

    struct SamplerParameters
    {
        uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler

        int32_t n_prev = 64;             // number of previous tokens to remember
        int32_t n_probs = 0;             // if greater than 0, output the probabilities of top n_probs tokens.
        int32_t min_keep = 0;            // 0 = disabled, otherwise samplers should return at least min_keep tokens
        int32_t top_k = 40;              // <= 0 to use vocab size
        float top_p = 0.95f;             // 1.0 = disabled
        float min_p = 0.05f;             // 0.0 = disabled
        float xtc_probability = 0.00f;   // 0.0 = disabled
        float xtc_threshold = 0.10f;     // > 0.5 disables XTC
        float typ_p = 1.00f;             // typical_p, 1.0 = disabled
        float temp = 0.80f;              // <= 0.0 to sample greedily, 0.0 to not output probabilities
        float dynatemp_range = 0.00f;    // 0.0 = disabled
        float dynatemp_exponent = 1.00f; // controls how entropy maps to temperature in dynamic temperature sampler
        int32_t penalty_last_n = 64;     // last n tokens to penalize (0 = disable penalty, -1 = context size)
        float penalty_repeat = 1.00f;    // 1.0 = disabled
        float penalty_freq = 0.00f;      // 0.0 = disabled
        float penalty_present = 0.00f;   // 0.0 = disabled
        float dry_multiplier = 0.0f;     // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
        float dry_base = 1.75f;          // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
        int32_t dry_allowed_length = 2;  // tokens extending repetitions beyond this receive penalty
        int32_t dry_penalty_last_n = -1; // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
        int32_t mirostat = 0;            // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        float top_n_sigma = -1.00f;      // -1.0 = disabled
        float mirostat_tau = 5.00f;      // target entropy
        float mirostat_eta = 0.10f;      // learning rate
        bool ignore_eos = false;
        bool no_perf = false; // disable performance metrics

        std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"}; // default sequence breakers for DRY

        std::vector<SamplerType> samplers = {
            SamplerType::Penalties,
            SamplerType::Dry,
            SamplerType::TopK,
            SamplerType::TypicalP,
            SamplerType::TopP,
            SamplerType::MinP,
            SamplerType::Xtc,
            SamplerType::Temperature,
        };

        std::string grammar; // optional BNF-like grammar to constrain sampling
        std::set<llama_token> preserved_tokens;

        std::vector<llama_logit_bias> logit_bias; // logit biases to apply
    };

    class Sampler
    {
    public:
        Sampler(const SamplerParameters &params, const llama_model *model);
        ~Sampler();

    public:
        void set_logits(llama_context *ctx, int32_t idx);
        void reset();
        void accept(llama_token token, bool accept_grammar);
        llama_token sample(llama_context *ctx, int32_t idx, bool grammar_first);

        llama_token_data_array *get_candidates()
        {
            return &cur_p_;
        }

        llama_token last_token()
        {
            return prev_tokens_.at(0);
        }

    private:
        SamplerParameters params_;

        llama_sampler *grammar_ = nullptr;
        llama_sampler *chain_ = nullptr;
        RingBuffer<llama_token> prev_tokens_;

        std::vector<llama_token_data> cur_;
        llama_token_data_array cur_p_;
    };
} // namespace spark_tts
