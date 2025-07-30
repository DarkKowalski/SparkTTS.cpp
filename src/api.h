// C API

#ifndef SPARK_TTS_API_H
#define SPARK_TTS_API_H

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef SPARK_TTS_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef SPARK_TTS_BUILD
#define SPARK_TTS_API __declspec(dllexport)
#else
#define SPARK_TTS_API __declspec(dllimport)
#endif
#else
#define SPARK_TTS_API __attribute__((visibility("default")))
#endif
#else
#define SPARK_TTS_API
#endif

#include <stdbool.h>
#include <stdint.h>

    typedef bool (*tts_synthesis_callback)(const float *audio_data, const size_t audio_size); // return true to continue decoding, false to stop
    typedef struct tts_context tts_context;

    SPARK_TTS_API struct tts_context *tts_create_context();
    SPARK_TTS_API void tts_free_context(struct tts_context *ctx);

#ifdef __cplusplus
}
#endif

#endif // SPARK_TTS_API_H
