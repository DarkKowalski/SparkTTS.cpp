#include "audio_detokenizer.h"

#include <onnxruntime/dml_provider_factory.h>
#include <onnxruntime/onnxruntime_c_api.h>
#include <iostream>

namespace spark_tts
{
    AudioDetokenizer::AudioDetokenizer(const std::string &model_path) : env_(ORT_LOGGING_LEVEL_ERROR, "AudioDetokenizer"),
                                                                        memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault))
    {
        try
        {
            Ort::SessionOptions session_options;
            // Enable DirectML
            session_options.DisableMemPattern();
            session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));

            bicodec_detokenizer_session_ = std::make_unique<Ort::Session>(env_, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
        }
        catch (const Ort::Exception &e)
        {
            // fallback to CPU if DML fails
            std::cerr << "Failed to create DML session: " << e.what() << std::endl;
            Ort::SessionOptions session_options;
            bicodec_detokenizer_session_ = std::make_unique<Ort::Session>(env_, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
            std::cerr << "Falling back to CPU execution provider." << std::endl;
        }

        input_tensors_ = {Ort::Value::CreateTensor<int64_t>(
                              memory_info_,
                              semantic_tokens_data_.data(), semantic_tokens_data_.size(),
                              bicodec_input_semantic_tokens_shape_.data(), bicodec_input_semantic_tokens_shape_.size()),

                          Ort::Value::CreateTensor<int32_t>(
                              memory_info_,
                              global_tokens_data_.data(), global_tokens_data_.size(),
                              bicodec_input_global_tokens_shape_.data(), bicodec_input_global_tokens_shape_.size())};

        output_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info_,
            wav_recon_data_.data(), wav_recon_data_.size(),
            bicodec_output_wav_recon_shape_.data(), bicodec_output_wav_recon_shape_.size());
    }

    std::array<float, 16000 * 1> AudioDetokenizer::detokenize(std::array<int64_t, 50> &semantic_tokens,
                                                              std::array<int32_t, 32> &global_tokens)
    {
        std::copy(semantic_tokens.begin(), semantic_tokens.end(), semantic_tokens_data_.begin());
        std::copy(global_tokens.begin(), global_tokens.end(), global_tokens_data_.begin());
        std::fill(wav_recon_data_.begin(), wav_recon_data_.end(), 0.0f);

        bicodec_detokenizer_session_->Run(
            Ort::RunOptions{nullptr},
            bicodec_input_names_.data(), input_tensors_.data(), input_tensors_.size(),
            bicodec_output_names_.data(), &output_tensor_, 1);

        return wav_recon_data_;
    }

} // namespace spark_tts
