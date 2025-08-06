#include <stdio.h>
#include <stdlib.h>

#include "api.h"
#include "utils.h"

typedef struct buffer
{
    float *data;
    size_t size;
    size_t capacity;
} buffer_t;

bool callback(void *user_data, const float *audio_data, const size_t samples)
{
    buffer_t *buffer = (buffer_t *)user_data;
    if (buffer->size + samples > buffer->capacity)
    {
        fprintf(stderr, "Buffer overflow: not enough space to store audio data.\n");
        return false; // Stop generating
    }

    memcpy(buffer->data + buffer->size, audio_data, samples * sizeof(float));
    buffer->size += samples;

    return true; // Continue generating
}

int main(void)
{
    struct tts_context *ctx = tts_create_context();
    if (!ctx)
    {
        fprintf(stderr, "Failed to create TTS context\n");
        return 1;
    }

    size_t audio_size = 0;
    float *audio_data = util_load_reference_audio("prompt_audio.wav", &audio_size);
    const char *text = "根据统计数据，法国奥德省人口第四多的是哪个市镇？";

#if defined(_WIN32) || defined(_WIN64)
    const char *wav2vec_model_path = "./models/Spark-TTS-0.5B/AudioTokenizer/wav2vec.xml";
    const char *mel_spectrogram_model_path = "./models/Spark-TTS-0.5B/AudioTokenizer/mel_spectrogram.xml";
    const char *bicodec_tokenizer_model_path = "./models/Spark-TTS-0.5B/AudioTokenizer/bicodec_tokenizer.xml";
    const char *audio_detokenizer_model_path = "./models/Spark-TTS-0.5B/AudioDetokenizer/bicodec_detokenizer.xml";
#elif defined(__APPLE__)
    const char *wav2vec_model_path = "./models/Spark-TTS-0.5B/AudioTokenizer/wav2vec.onnx";
    const char *mel_spectrogram_model_path = "./models/Spark-TTS-0.5B/AudioTokenizer/mel_spectrogram.onnx";
    const char *bicodec_tokenizer_model_path = "./models/Spark-TTS-0.5B/AudioTokenizer/bicodec_tokenizer.onnx";
    const char *audio_detokenizer_model_path = "./models/Spark-TTS-0.5B/AudioDetokenizer/bicodec_detokenizer.onnx";
#endif
    const char *transformer_model_path = "./models/Spark-TTS-0.5B/Transformer/model_q4_k.gguf";
    const char *tokenizer_path = "./models/Spark-TTS-0.5B/Tokenizer/tokenizer.json";

    tts_init_voice_feature_extraction(ctx,
                                      wav2vec_model_path,
                                      mel_spectrogram_model_path,
                                      bicodec_tokenizer_model_path,
                                      "CPU");

    tts_init_text_to_speech(ctx,
                            audio_detokenizer_model_path,
                            transformer_model_path,
                            tokenizer_path,
                            2048, // transformer_n_ctx
                            3,    // overlapped_semantic_tokens
                            10,   // callback_semantic_tokens, 0 for immediate callback
                            "CPU");

    size_t voice_features_size;
    int32_t *voice_features = tts_extract_voice_features(ctx, audio_data, audio_size, &voice_features_size);
    printf("Extracted voice features size: %zu\n", voice_features_size);
    free(audio_data);

    for (int i = 0; i < 10; i++)
    {
        const size_t n_sec = 30; // Max number of seconds to generate
        buffer_t buffer = {
            .data = (float *)malloc(n_sec * 16000 * sizeof(float)), // Assuming 16kHz sample rate
            .size = 0,
            .capacity = n_sec * 16000};

        printf("Starting text-to-speech generation...\n");
        tts_text_to_speech(ctx, text, voice_features, n_sec, false, &buffer, callback);
        printf("Text-to-speech generation completed.\n");

        if (buffer.data)
        {
            char output_filename[256];
            snprintf(output_filename, sizeof(output_filename), "output_%d.wav", i);
            util_save_generated_audio(output_filename, buffer.data, buffer.size);
            free(buffer.data);
            printf("Generated %f seconds of audio saved to %s\n", (float)buffer.size / 16000.0f, output_filename);
        }
    }

    free(voice_features);
    tts_free_context(ctx);
    return 0;
}
