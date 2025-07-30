#include "api.h"

#include <openvino/openvino.hpp>
#include <llama-cpp.h>

#include <iostream>

#include "audio_tokenizer.h"
#include "audio_detokenizer.h"
#include "prompt.h"
#include "transformer.h"

extern "C"
{
    struct tts_context
    {
        ov::Core core; // OpenVINO core
    };

    struct tts_context *tts_create_context()
    {
       return nullptr;
    }

    void tts_free_context(struct tts_context *ctx)
    {
       return;
    }
}
