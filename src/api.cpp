#include "api.h"

#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>

#include "synthesizer.h"

extern "C"
{
    struct tts_context
    {
        spark_tts::Synthesizer synthesizer; // Synthesizer instance
    };

    struct tts_context *tts_create_context()
    {
        return new tts_context();
    }

    void tts_free_context(struct tts_context *ctx)
    {
        delete ctx;
    }

    bool tts_init_voice_feature_extraction(tts_context *ctx, const char *audio_tokenizer_model_path)
    {
        if (!ctx || !audio_tokenizer_model_path)
        {
            std::cerr << "Invalid parameters for voice feature extraction initialization." << std::endl;
            return false;
        }

        try
        {
            ctx->synthesizer.init_voice_feature_extraction(audio_tokenizer_model_path);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Initializing voice feature extraction: " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    bool tts_init_text_to_speech(tts_context *ctx,
                                 const char *audio_detokenizer_model_path,
                                 const char *transformer_model_path,
                                 const char *tokenizer_path,
                                 const uint32_t transformer_n_ctx,
                                 const size_t overlapped_semantic_tokens)
    {
        if (!ctx || !audio_detokenizer_model_path || !transformer_model_path || !tokenizer_path)
        {
            return false;
        }

        try
        {
            ctx->synthesizer.init_text_to_speech(
                audio_detokenizer_model_path, transformer_model_path, tokenizer_path,
                transformer_n_ctx, overlapped_semantic_tokens);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Initializing text to speech: " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    void tts_deinit_voice_feature_extraction(tts_context *ctx)
    {
        if (ctx)
        {
            ctx->synthesizer.deinit_voice_feature_extraction();
        }
    }

    void tts_deinit_text_to_speech(tts_context *ctx)
    {
        if (ctx)
        {
            ctx->synthesizer.deinit_text_to_speech();
        }
    }

    int32_t *tts_extract_voice_features(tts_context *ctx,
                                        const float *audio_data,
                                        const size_t audio_size,
                                        size_t *voice_features_size)
    {
        if (!ctx || !audio_data || audio_size == 0 || !voice_features_size)
        {
            std::cerr << "Invalid parameters for voice feature extraction." << std::endl;
            return nullptr;
        }

        try
        {
            std::vector<float> audio_vector(audio_data, audio_data + audio_size);
            auto voice_features = ctx->synthesizer.extract_voice_features(audio_vector);

            *voice_features_size = voice_features.size();
            int32_t *features_array = (int32_t *)std::malloc(voice_features.size() * sizeof(int32_t));
            std::copy(voice_features.begin(), voice_features.end(), features_array);

            return features_array;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Extracting voice features: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void tts_text_to_speech(tts_context *ctx,
                            const char *text,
                            const int32_t *voice_features, // array of size 32
                            const size_t n_sec,
                            void *user_data,
                            tts_synthesis_callback callback)
    {
        if (!ctx || !text || !voice_features || n_sec == 0 || !callback || std::strlen(text) == 0)
        {
            std::cerr << "Invalid parameters for text to speech." << std::endl;
            return;
        }

        try
        {
            std::array<int32_t, 32> voice_features_array;
            std::copy(voice_features, voice_features + 32, voice_features_array.begin());
            spark_tts::Synthesizer::TextToSpeechCallback cb = [&user_data, &callback](std::vector<float> &audio_data) -> bool
            {
                return callback(user_data, audio_data.data(), audio_data.size());
            };
            ctx->synthesizer.text_to_speech(text, voice_features_array, n_sec, cb);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Text to speech: " << e.what() << std::endl;
        }
    }
}
