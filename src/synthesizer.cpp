#include "synthesizer.h"

namespace spark_tts
{
    void Synthesizer::init_voice_feature_extraction(const std::string &wav2vec_model_path,
                                                    const std::string &mel_spectrogram_model_path,
                                                    const std::string &bicodec_tokenizer_model_path,
                                                    const std::string &device_name)
    {
        audio_tokenizer_ = std::make_unique<AudioTokenizer>(core_, wav2vec_model_path, mel_spectrogram_model_path, bicodec_tokenizer_model_path, device_name);
    }

    void Synthesizer::init_text_to_speech(const std::string &audio_detokenizer_model_path,
                                          const std::string &transformer_model_path,
                                          const std::string &tokenizer_path,
                                          const uint32_t transformer_n_ctx,
                                          const size_t overlapped_semantic_tokens,
                                          const size_t callback_semantic_tokens,
                                          const std::string &device_name)
    {
        if (overlapped_semantic_tokens >= 50 || callback_semantic_tokens >= 50)
        {
            throw std::invalid_argument("overlapped/callback semantic tokens must be less than 50");
        }
        overlapped_semantic_tokens_ = overlapped_semantic_tokens;
        callback_semantic_tokens_ = callback_semantic_tokens;

        audio_detokenizer_ = std::make_unique<AudioDetokenizer>(core_, audio_detokenizer_model_path, device_name);

        auto transformer_params = Transformer::Params();
        transformer_params.ctx_params.n_ctx = transformer_n_ctx;
        transformer_ = std::make_unique<Transformer>(transformer_model_path, tokenizer_path, transformer_params);

        token_buffer_ = std::make_unique<TokenBuffer>(50, overlapped_semantic_tokens_);
    }

    // Must call init_voice_feature_extraction before this method
    std::array<int32_t, 32> Synthesizer::extract_voice_features(const std::vector<float> &audio_data)
    {
        auto [semantic_tokens, global_tokens] = audio_tokenizer_->tokenize(audio_data);

        std::array<int32_t, 32> voice_features = {};
        std::copy(global_tokens.data<int32_t>(), global_tokens.data<int32_t>() + 32, voice_features.begin());

        return voice_features;
    }

    bool Synthesizer::synthesize(const std::array<int32_t, 32> &voice_features, std::vector<float> &generated_audio)
    {
        const std::vector<int64_t> &front_buffer = token_buffer_->front_buffer();
        if (front_buffer.size() <= overlapped_semantic_tokens_)
        {
            // Not enough tokens to generate audio, return false
            return false;
        }

        const std::size_t trim_mask = 50 - front_buffer.size();

        ov::Tensor semantic_tokens_tensor(ov::element::i64, {1, 50});
        ov::Tensor global_tokens_tensor(ov::element::i32, {1, 1, 32});
        std::copy(front_buffer.begin(), front_buffer.end(), semantic_tokens_tensor.data<int64_t>());
        std::copy(voice_features.begin(), voice_features.end(), global_tokens_tensor.data<int32_t>());

        token_buffer_->flip();

        constexpr size_t samples_per_token = 320; // 50 tokens per second, 320 samples per token
        auto sample = audio_detokenizer_->detokenize(semantic_tokens_tensor, global_tokens_tensor);

        const size_t valid_start = first_sample_generated_ ? samples_per_token * overlapped_semantic_tokens_ : 0;
        const size_t valid_end = sample.size() - trim_mask * samples_per_token;
        generated_audio.assign(sample.begin() + valid_start, sample.begin() + valid_end);

        first_sample_generated_ = true;

        return true;
    }

    Transformer::DecodeCallbackAction Synthesizer::decode_callback(std::string &semantic_tokens,
                                                                   std::array<int32_t, 32> &voice_features,
                                                                   TextToSpeechCallback &callback)

    {
        auto semantic_token_ids = extract_semantic_token_ids(semantic_tokens);

        bool ready_to_synthesize = token_buffer_->add_tokens(semantic_token_ids);
        if (!ready_to_synthesize)
        {
            // If the buffer is not full, we can continue decoding
            return Transformer::DecodeCallbackAction::Continue;
        }

        std::vector<float> audio_output;
        synthesize(voice_features, audio_output);
        bool go_on = callback(audio_output);
        return go_on ? Transformer::DecodeCallbackAction::Continue : Transformer::DecodeCallbackAction::Stop;
    }

    // Must call init_text_to_speech before this method
    void Synthesizer::text_to_speech(const std::string &text,
                                     std::array<int32_t, 32> &voice_features,
                                     const size_t n_sec,
                                     const bool drop_last,
                                     TextToSpeechCallback &callback)
    {
        const std::string prompt = assemble_prompt(stringify_global_tokens(voice_features), text);
        const size_t n_predict = n_sec * (50 + overlapped_semantic_tokens_);

        // Store the lambda in a variable to create an lvalue
        Transformer::DecodeCallback decode_cb = [&](std::string &semantic_tokens) -> Transformer::DecodeCallbackAction
        {
            // Decode the text and call the callback
            return decode_callback(semantic_tokens, voice_features, callback);
        };

        first_sample_generated_ = false;
        token_buffer_->clear(); // Clear the token buffer before starting a new inference
        transformer_->infer(prompt, n_predict, callback_semantic_tokens_, decode_cb);

        if (drop_last)
        {
            return;
        }

        std::vector<float> last_audio_output;
        if (synthesize(voice_features, last_audio_output))
        {
            callback(last_audio_output);
        }
    }

} // namespace spark_tts
