#include "audio_detokenizer.h"

namespace spark_tts
{
    AudioDetokenizer::AudioDetokenizer(ov::Core &core, const std::string &model_path, const std::string &device_name)
    {
        auto bicodec_detokenizer_model = core.read_model(model_path);
        bicodec_detokenizer_ = core.compile_model(model_path, device_name);
    }

    // Semantic tokens [1, 50] i64
    // Global tokens [1, 1, 32] i32
    std::array<float, 16000 * 1> AudioDetokenizer::detokenize(const ov::Tensor &semantic_tokens, const ov::Tensor &global_tokens)
    {
        ov::InferRequest infer_request = bicodec_detokenizer_.create_infer_request();
        infer_request.set_input_tensor(0, semantic_tokens);
        infer_request.set_input_tensor(1, global_tokens);
        infer_request.infer();

        auto output_tensor = infer_request.get_output_tensor();
        std::array<float, 16000 * 1> audio_output;
        // output [1, 1, 16000] f32
        if (output_tensor.get_shape() == ov::Shape{1, 1, 16000})
        {
            std::copy(output_tensor.data<float>(), output_tensor.data<float>() + 16000, audio_output.begin());
        }
        else
        {
            throw std::runtime_error("Unexpected output shape: " + std::to_string(output_tensor.get_shape().size()));
        }

        return audio_output;
    }

} // namespace spark_tts
