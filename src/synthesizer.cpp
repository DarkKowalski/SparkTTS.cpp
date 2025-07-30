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
                                          const std::string &device_name)
    {

        audio_detokenizer_ = std::make_unique<AudioDetokenizer>(core_, audio_detokenizer_model_path, device_name);

        transformer_ = std::make_unique<Transformer>(transformer_model_path, tokenizer_path, Transformer::Params());
    }

    // Must call init_voice_feature_extraction before this method
    std::array<int32_t, 32> Synthesizer::extract_voice_features(const std::vector<float> &audio_data)
    {
        auto [semantic_tokens, global_tokens] = audio_tokenizer_->tokenize(audio_data);

        std::array<int32_t, 32> voice_features = {};
        std::copy(global_tokens.data<int32_t>(), global_tokens.data<int32_t>() + 32, voice_features.begin());

        return voice_features;
    }

    Transformer::DecodeCallbackAction Synthesizer::decode_callback(std::string &semantic_tokens,
                                                                   std::array<int32_t, 32> &voice_features,
                                                                   TextToSpeechCallback &callback)

    {
        auto semantic_token_ids = extract_semantic_token_ids(semantic_tokens);

        // TODO: implement logic to decode audio


        std::vector<float> audio_output;
        bool cont = callback(audio_output);
        return cont ? Transformer::DecodeCallbackAction::Continue : Transformer::DecodeCallbackAction::Stop;
    }

    // Must call init_text_to_speech before this method
    void Synthesizer::text_to_speech(const std::string &text, std::array<int32_t, 32> &voice_features, const size_t n_sec, TextToSpeechCallback &callback)
    {
        const std::string prompt = assemble_prompt(text, stringify_global_tokens(voice_features));
        const size_t n_predict = n_sec * (50 - (overlapped_semantic_tokens_ * 2));

        // Store the lambda in a variable to create an lvalue
        Transformer::DecodeCallback decode_cb = [&](std::string &semantic_tokens) -> Transformer::DecodeCallbackAction
        {
            // Decode the text and call the callback
            return decode_callback(semantic_tokens, voice_features, callback);
        };

        transformer_->infer(prompt, n_predict, max_semantic_tokens_per_generation_, decode_cb);
    }

} // namespace spark_tts
