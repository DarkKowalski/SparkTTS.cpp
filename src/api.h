// C API

#ifndef TTS_API_H
#define TTS_API_H

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef TTS_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef TTS_BUILD
#define TTS_API __declspec(dllexport)
#else
#define TTS_API __declspec(dllimport)
#endif
#else
#define TTS_API __attribute__((visibility("default")))
#endif
#else
#define TTS_API
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

    typedef bool (*tts_synthesis_callback)(void *user_data, const float *audio_data, const size_t audio_size); // return true to continue decoding, false to stop
    typedef struct tts_context tts_context;

    TTS_API struct tts_context *tts_create_context();

    TTS_API void tts_free_context(struct tts_context *ctx);

    TTS_API bool tts_init_voice_feature_extraction(tts_context *ctx,
                                                   const char *audio_tokenizer_model_path);

    TTS_API bool tts_init_text_to_speech(tts_context *ctx,
                                         const char *audio_detokenizer_model_path,
                                         const char *transformer_model_path,
                                         const char *tokenizer_path,
                                         const uint32_t transformer_n_ctx,
                                         const size_t overlapped_semantic_tokens,
                                         const size_t callback_semantic_tokens,
                                         const char *device_name);

    // free after use
    TTS_API int32_t *tts_extract_voice_features(tts_context *ctx,
                                                const float *audio_data,
                                                const size_t audio_size,
                                                size_t *voice_features_size);

    TTS_API void tts_text_to_speech(tts_context *ctx,
                                    const char *text,
                                    const int32_t *voice_features, // array of size 32
                                    const size_t n_sec,            // max number of seconds to generate
                                    const bool drop_last,
                                    void *user_data,
                                    tts_synthesis_callback callback);

#ifdef __cplusplus
}
#endif

#endif // TTS_API_H
