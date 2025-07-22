#include <openvino/openvino.hpp>
#include <openvino/genai/llm_pipeline.hpp>

#include <sndfile.hh>

#include <argparse/argparse.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>
#include <functional>

#include "audio_tokenizer.h"
#include "audio_detokenizer.h"
#include "prompt.h"
#include "utils.h"
#include "llm_streamer.h"

int main(int argc, char *argv[])
{
    // Argparse
    argparse::ArgumentParser program("SparkTTS");

    program.add_argument("--model")
        .default_value(std::string("./models/Spark-TTS-0.5B"))
        .help("Path to the Spark-TTS model directory");

    program.add_argument("--device")
        .default_value(std::string("CPU"))
        .help("Device to run the model on (e.g., CPU, NPU, GPU, AUTO)");

    program.add_argument("--audio-output")
        .default_value(std::string("./output_audio.wav"))
        .help("Path to save the generated audio output");

    program.add_argument("--audio-input")
        .default_value(std::string("./prompt_audio.wav"))
        .help("Path to the input audio file for tokenization");

    program.add_argument("--prompt")
        .default_value(std::string("Hello, this is spark TTS speaking."))
        .help("Text prompt to generate audio from");

    program.add_argument("--pin-cpu")
        .default_value(false)
        .implicit_value(true)
        .help("Enable CPU pinning for better performance");

    program.add_argument("--num-threads")
        .default_value(4)
        .scan<'i', int32_t>()
        .help("Number of threads for inference");

    program.add_argument("--hyper-threading")
        .default_value(false)
        .implicit_value(true)
        .help("Enable hyper-threading for better performance");

    program.add_argument("--execution-mode")
        .default_value("performance")
        .help("Execution mode for OpenVINO (e.g., performance, accuracy)");

    program.add_argument("--execution-mode")
        .default_value("latency")
        .help("Execution mode for OpenVINO (e.g., throughput, latency)");

    program.add_argument("--max-new-tokens")
        .default_value(3000)
        .scan<'i', int32_t>()
        .help("Maximum number of new tokens to generate");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << "Error: " << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // check reference audio file
    if (!std::filesystem::exists(program.get<std::string>("--audio-input")))
    {
        std::cerr << "Error: Reference audio file does not exist: "
                  << program.get<std::string>("--audio-input") << std::endl;
        return 1;
    }

    // Initialize Models
    ov::Core core;
    const std::string device_name = program.get<std::string>("--device");
    const std::string model_path = program.get<std::string>("--model");

    std::shared_ptr<spark_tts::AudioTokenizer> audio_tokenizer = std::make_shared<spark_tts::AudioTokenizer>(
        core, model_path, device_name);
    std::cout << "Audio Tokenizer initialized." << std::endl;

    std::shared_ptr<spark_tts::AudioDetokenizer> audio_detokenizer = std::make_shared<spark_tts::AudioDetokenizer>(
        core, model_path, device_name);
    std::cout << "Audio Detokenizer initialized." << std::endl;

    const ov::hint::PerformanceMode performance_mode =
        program.get<std::string>("--execution-mode") == "latency" ? ov::hint::PerformanceMode::LATENCY
                                                                  : ov::hint::PerformanceMode::THROUGHPUT;
    const ov::hint::ExecutionMode execution_mode =
        program.get<std::string>("--execution-mode") == "performance" ? ov::hint::ExecutionMode::PERFORMANCE
                                                                      : ov::hint::ExecutionMode::ACCURACY;
    const ov::AnyMap llm_hints{
        {ov::hint::enable_cpu_pinning.name(), program.get<bool>("--pin-cpu")},
        {ov::hint::execution_mode.name(), execution_mode},
        {ov::hint::performance_mode.name(), performance_mode},
        {ov::inference_num_threads.name(), program.get<int32_t>("--num-threads")},
        {ov::hint::enable_hyper_threading.name(), program.get<bool>("--hyper-threading")},
    };

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = program.get<int32_t>("--max-new-tokens");
    generation_config.top_k = 50;
    generation_config.top_p = 0.95;
    generation_config.temperature = 0.8;
    generation_config.do_sample = true;
    generation_config.apply_chat_template = false;

    ov::genai::LLMPipeline llm(model_path + "/LLM/", device_name, llm_hints);
    llm.set_generation_config(generation_config);
    std::cout << "LLM Pipeline initialized." << std::endl;

    // Tokenize the reference audio and assemble the prompt
    const auto reference_audio = spark_tts::load_reference_audio(program.get<std::string>("--audio-input"));
    auto tokens = audio_tokenizer->tokenize(reference_audio);
    const auto &global_tokens = tokens.second;
    const std::string global_token_input_str = spark_tts::stringify_global_tokens(global_tokens);
    const std::string prompt = spark_tts::assemble_prompt(
        global_token_input_str,
        program.get<std::string>("--prompt"));
    std::cout << "Prompt assembled." << std::endl;

    // LLM streaming generation
    std::cout << "Starting LLM generation..." << std::endl;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    std::vector<float> generated_audio;
    auto streamer = std::make_shared<spark_tts::LLMStreamer>(
        llm.get_tokenizer(),
        audio_detokenizer,
        [&generated_audio, &start_time](std::vector<float> &audio)
        {
            auto elapsed_time = std::chrono::steady_clock::now() - start_time;
            std::cout << "Generated audio at " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count()
                      << " ms" << std::endl;
            generated_audio.insert(generated_audio.end(), audio.begin(), audio.end());
        });
    // Set global tokens for the "Tone Color" feature
    streamer->set_global_tokens(global_tokens);

    llm.generate(prompt, ov::genai::streamer(streamer));

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Generation took " << elapsed_seconds.count() << " seconds." << std::endl;

    // Save the generated audio to a file
    const auto samples = spark_tts::save_generated_audio(
        program.get<std::string>("--audio-output"),
        generated_audio);
    std::cout << "Generated " << samples << " samples, duration: "
              << static_cast<double>(samples) / 16000.0 << " seconds." << std::endl;

    return 0;
}
