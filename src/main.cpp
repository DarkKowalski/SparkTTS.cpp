#include <openvino/openvino.hpp>
#include <llama-cpp.h>

#include <sndfile.hh>

#include <argparse/argparse.hpp>

#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>
#include <functional>

#include "utils.h"
#include "synthesizer.h"

int main(int argc, char *argv[])
{

    auto ref_audio = spark_tts::load_reference_audio("prompt_audio.wav");

    spark_tts::Synthesizer synthesizer;

    synthesizer.init_voice_feature_extraction(
        "./models/Spark-TTS-0.5B/AudioTokenizer/wav2vec.xml",
        "./models/Spark-TTS-0.5B/AudioTokenizer/mel_spectrogram.xml",
        "./models/Spark-TTS-0.5B/AudioTokenizer/bicodec_tokenizer.xml",
        "CPU");

    synthesizer.init_text_to_speech(
        "./models/Spark-TTS-0.5B/AudioDetokenizer/bicodec_detokenizer.xml",
        "./models/Spark-TTS-0.5B/Transformer/model_f16.gguf",
        "./models/Spark-TTS-0.5B/Tokenizer/",
        2048, // transformer_n_ctx
        3,    // overlapped_semantic_tokens
        10,   // callback_semantic_tokens, 0 for immediate callback
        "CPU");

    const std::string text = "根据统计数据，法国奥德省人口第四多的是哪个市镇？";

    auto voice_features = synthesizer.extract_voice_features(ref_audio);

    for (int i = 0; i < 10; i++)
    {
        std::vector<float> generated;
        std::chrono::steady_clock::time_point first_sample_time;
        spark_tts::Synthesizer::TextToSpeechCallback callback = [&generated, &first_sample_time](std::vector<float> &audio_output) -> bool
        {
            if (generated.empty())
            {
                first_sample_time = std::chrono::steady_clock::now();
            }
            generated.insert(generated.end(), audio_output.begin(), audio_output.end());
            return true; // Return true to continue generating, false to stop
        };

        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        synthesizer.text_to_speech(text, voice_features, 100, false, callback);
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        std::chrono::duration<double> first_sample_latency = first_sample_time - start_time;

        std::cout << "Text-to-speech generation took " << elapsed_seconds.count() << " seconds." << std::endl;
        std::cout << "First sample latency: " << first_sample_latency.count() << " seconds." << std::endl;
        std::cout << "Generated audio seconds: " << generated.size() / 16000.0 << " seconds." << std::endl;
        spark_tts::save_generated_audio("output_" + std::to_string(i) + ".wav", generated);
    }

    return 0;
}
