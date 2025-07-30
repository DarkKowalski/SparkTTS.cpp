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
    try
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
            "./models/Spark-TTS-0.5B/GGUF/LLM-507M-F16.gguf",
            "./models/Spark-TTS-0.5B/LLM/",
            "CPU");

        const std::string text = "根据统计数据，法国奥德省人口第四多的是哪个市镇？";

        auto voice_features = synthesizer.extract_voice_features(ref_audio);

        std::vector<float> generated;
        spark_tts::Synthesizer::TextToSpeechCallback callback = [&generated](std::vector<float> &audio_output) -> bool
        {
            generated.insert(generated.end(), audio_output.begin(), audio_output.end());
            return true; // Return true to continue generating, false to stop
        };

        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        synthesizer.text_to_speech(text, voice_features, 100, callback);
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        std::cout << "Text-to-speech generation took " << elapsed_seconds.count() << " seconds." << std::endl;
        std::cout << "Generated audio seconds: " << generated.size() / 16000.0 << " seconds." << std::endl;

        spark_tts::save_generated_audio("output.wav", generated);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
