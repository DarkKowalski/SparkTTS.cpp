#include "audio_detokenizer.h"

#include <onnxruntime/dml_provider_factory.h>
#include <onnxruntime/onnxruntime_c_api.h>

namespace spark_tts
{
    AudioDetokenizer::AudioDetokenizer(const std::string &model_path) : env_(ORT_LOGGING_LEVEL_ERROR, "AudioDetokenizer"),
                                                                        memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault))
    {
        Ort::SessionOptions session_options;
        session_options.DisableMemPattern();
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
        bicodec_detokenizer_session_ = std::make_unique<Ort::Session>(env_, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
    }

    std::array<float, 16000 * 1> AudioDetokenizer::detokenize(std::array<int64_t, 50> &semantic_tokens,
                                                              std::array<int32_t, 32> &global_tokens)
    {
        std::array<float, 16000 * 1> wav_recon_data = {};

        std::array<Ort::Value, 2> input_tensors = {
            Ort::Value::CreateTensor<int64_t>(
                memory_info_,
                semantic_tokens.data(), semantic_tokens.size(),
                bicodec_input_semantic_tokens_shape_.data(), bicodec_input_semantic_tokens_shape_.size()),
            Ort::Value::CreateTensor<int32_t>(
                memory_info_,
                global_tokens.data(), global_tokens.size(),
                bicodec_input_global_tokens_shape_.data(), bicodec_input_global_tokens_shape_.size())};

        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            wav_recon_data.data(), wav_recon_data.size(),
            bicodec_output_wav_recon_shape_.data(), bicodec_output_wav_recon_shape_.size());

        bicodec_detokenizer_session_->Run(
            Ort::RunOptions{nullptr},
            bicodec_input_names_.data(), input_tensors.data(), input_tensors.size(),
            bicodec_output_names_.data(), &output_tensor, 1);

        return wav_recon_data;
    }

} // namespace spark_tts
