#include <openvino/openvino.hpp>
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <llama-cpp.h>

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

#include "transformer.h"

int main(int argc, char *argv[])
{

    try
    {
        //ggml_backend_load_all();
        llama_backend_init();
        auto callback = [](std::string &text)
        {
            std::cout << "Decoded text: " << text << std::endl;
        };

        auto llm = std::make_unique<spark_tts::Transformer>("./models/Spark-TTS-0.5B/GGUF/LLM-507M-F16.gguf", callback);

        std::string prompt = "<|task_tts|><|start_content|>Hi how are you<|end_content|><|start_global_token|><|bicodec_global_3363|><|bicodec_global_2367|><|bicodec_global_1591|><|bicodec_global_3625|><|bicodec_global_294|><|bicodec_global_3556|><|bicodec_global_1198|><|bicodec_global_2646|><|bicodec_global_3397|><|bicodec_global_3778|><|bicodec_global_2459|><|bicodec_global_3109|><|bicodec_global_1017|><|bicodec_global_3860|><|bicodec_global_3178|><|bicodec_global_3414|><|bicodec_global_1727|><|bicodec_global_561|><|bicodec_global_1096|><|bicodec_global_3133|><|bicodec_global_3647|><|bicodec_global_3178|><|bicodec_global_2767|><|bicodec_global_73|><|bicodec_global_2418|><|bicodec_global_1838|><|bicodec_global_3645|><|bicodec_global_2338|><|bicodec_global_2427|><|bicodec_global_2144|><|bicodec_global_306|><|bicodec_global_2799|><|end_global_token|>";

        llm->infer(prompt, 3000, 50);

        llama_backend_free();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "An unknown error occurred." << std::endl;
        return 1;
    }

    return 0;
}
