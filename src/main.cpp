#include <sndfile.hh>
#include <nlohmann/json.hpp>
#include <argparse/argparse.hpp>

#include <atomic>
#include <thread>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <functional>
#include <variant>

#include "utils.h"
#include "synthesizer.h"

namespace tool
{
    // Protocol for interactive mode

    // In
    // {
    //     "method": "tts",
    //     "params": {
    //         "text": "text to synthesize",
    //         "features": [3363, 2367, 2615, 3369, 278, 3556, 1194,
    //                       1558, 3141, 3778, 2442, 3109, 1017, 3844,
    //                       3194, 3158, 2751, 1586, 1096, 3133, 3711,
    //                       3178, 2767, 133, 2354, 1838, 3644, 2401,
    //                       3450, 2400, 50, 2751],
    //         "output": "path/to/output.wav"
    //     }
    // }
    // Out
    // {
    //     "ok": true,
    //     "message": "optional message"
    // }

    // Input in one line you can use:
    // { "method": "tts", "params": { "text": "Hello, world!", "features": [3363, 2367, 2615, 3369, 278, 3556, 1194, 1558, 3141, 3778, 2442, 3109, 1017, 3844, 3194, 3158, 2751, 1586, 1096, 3133, 3711, 3178, 2767, 133, 2354, 1838, 3644, 2401, 3450, 2400, 50, 2751], "output": "output.wav" } }
    struct TextToSpeechInput
    {
        std::string text;
        std::array<int32_t, 32> features; // 32 integers
        std::string output_path;
    };

    struct TextToSpeechOutput
    {
        bool ok;
        std::string message;
    };

    // In
    // {
    //     "method": "clone",
    //     "params": {
    //         "    ": "path/to/source.wav",
    //     }
    // }
    // Out
    // {
    //     "ok": true,
    //     "message": "optional message",
    //     "features": [3363, 2367, 2615, 3369, 278, 3556, 1194,
    //                  1558, 3141, 3778, 2442, 3109, 1017, 3844,
    //                  3194, 3158, 2751, 1586, 1096, 3133, 3711,
    //                  3178, 2767, 133, 2354, 1838, 3644, 2401,
    //                  3450, 2400, 50, 2751]
    // }
    struct VoiceCloneInput
    {
        std::string source;
    };

    struct VoiceCloneOutput
    {
        bool ok;
        std::string message;
        std::vector<int32_t> features; // 32 integers
    };

    typedef std::variant<std::monostate, TextToSpeechInput, VoiceCloneInput> ProtocolInput;
    typedef std::variant<std::monostate, TextToSpeechOutput, VoiceCloneOutput> ProtocolOutput;

    class SerDes
    {
    public:
        // To keep it simple, we don't validate the input JSON structure here.
        static ProtocolInput deserialize_input(const std::string &json_str)
        {
            nlohmann::json j = nlohmann::json::parse(json_str);
            const std::string method = j.value("method", "");
            if (method == "tts")
            {
                TextToSpeechInput input;
                input.text = j["params"]["text"].get<std::string>();
                input.output_path = j["params"]["output"].get<std::string>();
                for (size_t i = 0; i < 32; ++i)
                {
                    input.features[i] = j["params"]["features"][i].get<int32_t>();
                }
                return input;
            }
            else if (method == "clone")
            {
                VoiceCloneInput input;
                input.source = j["params"]["source"].get<std::string>();
                return input;
            }

            throw std::runtime_error("Invalid input");
        }

        static std::string serialize_output(const ProtocolOutput &output)
        {
            nlohmann::json j;
            if (std::holds_alternative<TextToSpeechOutput>(output))
            {
                const auto &tts_output = std::get<TextToSpeechOutput>(output);
                j["ok"] = tts_output.ok;
                j["message"] = tts_output.message;
            }
            else if (std::holds_alternative<VoiceCloneOutput>(output))
            {
                const auto &clone_output = std::get<VoiceCloneOutput>(output);
                j["ok"] = clone_output.ok;
                j["message"] = clone_output.message;
                j["features"] = clone_output.features;
            }

            return j.dump();
        }
    };

    class CommandLineInterface
    {
    public:
        CommandLineInterface(int argc, char *argv[]) : program_("tts_cli")
        {
            program_.add_argument("-m", "--model")
                .help("Path to the model directory")
                .required()
                .default_value(std::string("./models/Spark-TTS-0.5B"));

            program_.add_argument("-it", "--interactive")
                .help("Run in interactive mode")
                .default_value(false)
                .implicit_value(true);

            program_.add_argument("--enable-perf")
                .help("Enable performance monitoring")
                .default_value(false)
                .implicit_value(true);

            program_.add_argument("--enable-clone")
                .help("Enable voice cloning feature")
                .default_value(false)
                .implicit_value(true);

            program_.add_argument("--enable-tts")
                .help("Enable text-to-speech feature")
                .default_value(false)
                .implicit_value(true);

            program_.add_argument("--n-ctx")
                .help("Transformer context size")
                .default_value(transformer_n_ctx_)
                .scan<'u', uint32_t>();

            program_.add_argument("-sec", "--n-seconds")
                .help("Max seconds to generate (default 120)")
                .default_value(tts_n_seconds_)
                .scan<'i', int32_t>();

            program_.add_argument("-ost", "--overlapped-semantic-tokens")
                .help("Number of tokens to overlap between generations (0 to 25, default 3)")
                .default_value(overlapped_semantic_tokens_)
                .scan<'i', int32_t>();

            program_.add_argument("-cst", "--callback-semantic-tokens")
                .help("Number of tokens to trigger callback (0 for immediate callback, default 10)")
                .default_value(callback_semantic_tokens_)
                .scan<'i', int32_t>();

            program_.add_argument("-i", "--input")
                .help("Path to the input audio file for voice cloning")
                .default_value(std::string("./prompt_audio.wav"));

            program_.add_argument("-o", "--output")
                .help("Path to the output audio file for voice cloning")
                .default_value(std::string("./output.wav"));

            program_.add_argument("-t", "--text")
                .help("Text to synthesize for text-to-speech")
                .default_value(std::string("Hello, this is a test of the Spark TTS system."));

            program_.parse_args(argc, argv);

            interactive_mode_ = program_.get<bool>("--interactive");
            enable_clone_ = program_.get<bool>("--enable-clone");
            enable_tts_ = program_.get<bool>("--enable-tts");
            enable_perf_ = program_.get<bool>("--enable-perf");

            model_path_ = program_.get<std::string>("--model");
            transformer_n_ctx_ = program_.get<uint32_t>("--n-ctx");
            tts_n_seconds_ = program_.get<int32_t>("--n-seconds");
            overlapped_semantic_tokens_ = program_.get<int32_t>("--overlapped-semantic-tokens");
            callback_semantic_tokens_ = program_.get<int32_t>("--callback-semantic-tokens");

            one_shot_output_audio_path_ = program_.get<std::string>("--output");
            one_shot_input_audio_path_ = program_.get<std::string>("--input");
            one_shot_text_ = program_.get<std::string>("--text");

            if (!interactive_mode_)
            {
                enable_tts_ = true;   // Enable TTS in one-shot mode by default
                enable_clone_ = true; // Enable cloning in one-shot mode by default
            }
        }

        void run()
        {
#if defined(_WIN32)
            const std::string audio_tokenizer_model_path = model_path_ + "/AudioTokenizer/AudioTokenizer.onnx";
            const std::string audio_detokenizer_model_path = model_path_ + "/AudioDetokenizer/AudioDetokenizer.onnx";
#elif defined(__APPLE__)
            const std::string audio_tokenizer_model_path = model_path_ + "/AudioTokenizer/AudioTokenizer.mlmodelc";
            const std::string audio_detokenizer_model_path = model_path_ + "/AudioDetokenizer/AudioDetokenizer.mlmodelc";
#endif
            const std::string transformer_model_path = model_path_ + "/Transformer/model.gguf";
            const std::string tokenizer_path = model_path_ + "/Tokenizer/tokenizer.json";

            if (enable_clone_)
            {
                std::cerr << "Voice cloning feature is enabled." << std::endl;
                synthesizer_.init_voice_feature_extraction(audio_tokenizer_model_path);
            }

            if (enable_tts_)
            {
                std::cerr << "Text-to-speech feature is enabled." << std::endl;
                synthesizer_.init_text_to_speech(
                    audio_detokenizer_model_path,
                    transformer_model_path,
                    tokenizer_path,
                    transformer_n_ctx_,
                    overlapped_semantic_tokens_,
                    callback_semantic_tokens_);
            }

            if (interactive_mode_)
            {
                run_interactive_mode();
            }
            else
            {
                run_one_shot_mode();
            }
        }

    private:
        void run_one_shot_mode()
        {
            std::cerr << "Running in one-shot mode." << std::endl;

            VoiceCloneInput clone_input;
            clone_input.source = one_shot_input_audio_path_;
            VoiceCloneOutput clone_output = voice_clone_sync(clone_input);

            if (!clone_output.ok)
            {
                std::cerr << "Voice cloning failed: " << clone_output.message << std::endl;
                return;
            }

            std::array<int32_t, 32> voice_features;
            std::copy(clone_output.features.begin(), clone_output.features.end(), voice_features.begin());
            TextToSpeechInput tts_input;
            tts_input.text = one_shot_text_;
            tts_input.features = voice_features;
            tts_input.output_path = one_shot_output_audio_path_;

            TextToSpeechOutput tts_output = text_to_speech_sync(tts_input);
            if (!tts_output.ok)
            {
                std::cerr << "Text-to-speech failed: " << tts_output.message << std::endl;
                return;
            }
            std::cout << "Text-to-speech completed successfully. Output saved to: " << one_shot_output_audio_path_ << std::endl;

            std::cout << "Generated audio features: ";
            for (const auto &feature : clone_output.features)
            {
                std::cout << feature << " ";
            }
            std::cout << std::endl;

            if (enable_perf_)
            {
                std::cout << "Performance info: " << tts_output.message << std::endl;
            }
        }

        void run_interactive_mode()
        {
            if (!enable_clone_ && !enable_tts_)
            {
                std::cerr << "No features enabled. Exiting." << std::endl;
                return;
            }

            std::cerr << "Running in interactive mode. Press Ctrl+C to exit." << std::endl;

            // Read input from stdin
            std::string input_line;
            while (std::getline(std::cin, input_line))
            {
                if (input_line.empty())
                {
                    continue; // Skip empty lines
                }

                ProtocolInput input;
                try
                {
                    input = SerDes::deserialize_input(input_line);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error parsing input: " << e.what() << std::endl;
                    continue;
                }

                if (std::holds_alternative<TextToSpeechInput>(input))
                {
                    const auto &tts_input = std::get<TextToSpeechInput>(input);
                    TextToSpeechOutput output = text_to_speech_sync(tts_input);
                    std::cout << SerDes::serialize_output(output) << std::endl;
                }
                else if (std::holds_alternative<VoiceCloneInput>(input))
                {
                    const auto &clone_input = std::get<VoiceCloneInput>(input);
                    VoiceCloneOutput output = voice_clone_sync(clone_input);
                    std::cout << SerDes::serialize_output(output) << std::endl;
                }
                else
                {
                    std::cerr << "Unknown input type" << std::endl;
                }
            }
        }

        VoiceCloneOutput voice_clone_sync(const VoiceCloneInput &input)
        {
            const std::string &source_path = input.source;

            if (!enable_clone_)
            {
                return {false, "Voice cloning feature is not enabled."};
            }

            std::vector<float> ref_audio;
            // Load the audio file
            try
            {
                ref_audio = spark_tts::load_reference_audio(source_path);
            }
            catch (const std::exception &e)
            {
                return {false, "Error loading reference audio: " + std::string(e.what())};
            }

            std::array<int32_t, 32> voice_features = synthesizer_.extract_voice_features(ref_audio);

            return {true, "", std::vector<int32_t>(voice_features.begin(), voice_features.end())}; // Return the features
        }

        TextToSpeechOutput text_to_speech_sync(const TextToSpeechInput &input)
        {
            const std::string &text = input.text;
            const std::array<int32_t, 32> &features = input.features;
            const std::string &output_path = input.output_path;

            if (!enable_tts_)
            {
                return {false, "Text-to-speech feature is not enabled."};
            }

            std::array<int32_t, 32> voice_features = features;
            std::vector<float> audio_data;
            std::string perf_info;

            if (enable_perf_)
            {
                std::chrono::steady_clock::time_point first_sample_time;
                spark_tts::Synthesizer::TextToSpeechCallback callback = [&](std::vector<float> &audio_output) -> bool
                {
                    if (audio_data.empty() && enable_perf_)
                    {
                        first_sample_time = std::chrono::steady_clock::now();
                    }
                    audio_data.insert(audio_data.end(), audio_output.begin(), audio_output.end());
                    return true; // Continue generating
                };

                std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
                synthesizer_.text_to_speech(text, voice_features, tts_n_seconds_, false, callback);
                std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_time = end_time - start_time;

                std::chrono::duration<double> first_sample_latency = first_sample_time - start_time;
                perf_info = "total, " + std::to_string(elapsed_time.count()) +
                            ", first_sample_latency, " + std::to_string(first_sample_latency.count()) +
                            ", generated_seconds, " + std::to_string(audio_data.size() / 16000.0);
            }
            else
            {
                spark_tts::Synthesizer::TextToSpeechCallback callback = [&](std::vector<float> &audio_output) -> bool
                {
                    audio_data.insert(audio_data.end(), audio_output.begin(), audio_output.end());
                    return true; // Continue generating
                };
                synthesizer_.text_to_speech(text, voice_features, tts_n_seconds_, false, callback);
            }

            // Write the audio data to a file
            spark_tts::save_generated_audio(output_path, audio_data);

            return {true, perf_info};
        }

    private:
        argparse::ArgumentParser program_;
        spark_tts::Synthesizer synthesizer_;

    private:
        bool interactive_mode_ = false;
        bool enable_clone_ = false;
        bool enable_tts_ = false;
        bool enable_perf_ = false;

        std::string model_path_;

        std::string one_shot_output_audio_path_;
        std::string one_shot_input_audio_path_;
        std::string one_shot_text_;

        uint32_t transformer_n_ctx_ = 2048;      // Default context size
        int32_t tts_n_seconds_ = 120;            // Default max seconds to generate
        int32_t overlapped_semantic_tokens_ = 3; // Default overlap for semantic tokens
        int32_t callback_semantic_tokens_ = 10;  // Default callback tokens, 0 for immediate callback
    };

} // namespace tool

void signal_int_handler(int signal)
{
    std::cerr << "Received SIGINT, shutting down..." << std::endl;
    std::exit(0);
}

void signal_term_handler(int signal)
{
    std::cerr << "Received SIGTERM, shutting down..." << std::endl;
    std::exit(0);
}

int main(int argc, char *argv[])
{
    std::signal(SIGINT, signal_int_handler);
    std::signal(SIGTERM, signal_term_handler);

    try
    {
        tool::CommandLineInterface cli(argc, argv);

        cli.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
