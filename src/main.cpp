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

#include "synthesizer.h"

int main(int argc, char *argv[])
{
    try
    {

        spark_tts::Synthesizer synthesizer;

        synthesizer.init_voice_feature_extraction(
            "./models/Spark-TTS-0.5B/AudioTokenizer/wav2vec.xml",
            "./models/Spark-TTS-0.5B/AudioTokenizer/mel_spectrogram.xml",
            "./models/Spark-TTS-0.5B/AudioTokenizer/bicodec_tokenizer.xml",
            "CPU");

        synthesizer.init_text_to_speech(
            "./models/Spark-TTS-0.5B/AudioDetokenizer/bicodec_detokenizer.xml",
            "./models/Spark-TTS-0.5B/GGUF/LLM-507M-Q4_K.gguf",
            "./models/Spark-TTS-0.5B/LLM/",
            "CPU");

        const std::string text = "Hi how are you";
        std::vector<float> audio_data(16000 * 1, 0.1f); // Dummy audio data, replace with actual audio data

        auto voice_features = synthesizer.extract_voice_features(audio_data);
        for (const auto &feature : voice_features)
        {
            std::cout << feature << " ";
        }
        std::cout << std::endl;


        spark_tts::Synthesizer::TextToSpeechCallback callback = [](std::vector<float> &audio_output) -> bool
        {
            // Here you can process the audio output, e.g., play it or save it to a file
            std::cout << "Generated audio data size: " << audio_output.size() << std::endl;
            for (const auto &sample : audio_output)
            {
                std::cout << sample << " ";
            }
            std::cout << std::endl;
            return true; // Return true to continue generating, false to stop
        };

        synthesizer.text_to_speech(text, voice_features, 5, callback);

    } catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
