#include "../profiler/profiler.h"

#include "audio_tokenizer_impl.h"
#include "dxgi_device_selector.h"

#include <onnxruntime/dml_provider_factory.h>
#include <onnxruntime/onnxruntime_c_api.h>
#include <iostream>

namespace spark_tts
{
    AudioTokenizerImpl::AudioTokenizerImpl(const std::string &model_path) : env_(ORT_LOGGING_LEVEL_ERROR, "AudioTokenizer"),
                                                                            memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault))
    {
        TRACE_EVENT("audio_tokenizer", "AudioTokenizer::AudioTokenizer");

        try
        {
            std::unique_ptr<DXGIDeviceSelector> dxgi_device_selector = std::make_unique<DXGIDeviceSelector>();
            int device_id = dxgi_device_selector->get_high_performance_adapter_index();
            const std::string device_description = dxgi_device_selector->get_high_performance_adapter_description();
            std::cerr << "DirectML device ID: " << device_id << ", Description: " << device_description << std::endl;

            Ort::SessionOptions session_options;
            // Enable DirectML
            session_options.DisableMemPattern();
            session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, device_id));

            audio_tokenizer_session_ = std::make_unique<Ort::Session>(env_, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
        }
        catch (const Ort::Exception &e)
        {
            // fallback to CPU if DML fails
            std::cerr << "Failed to create DML session: " << e.what() << std::endl;
            Ort::SessionOptions session_options;
            audio_tokenizer_session_ = std::make_unique<Ort::Session>(env_, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
            std::cerr << "Falling back to CPU execution provider." << std::endl;
        }

        input_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info_,
            audio_input_data_.data(), audio_input_data_.size(),
            audio_tokenizer_input_shape_.data(), audio_tokenizer_input_shape_.size());

        output_tensors_ = {Ort::Value::CreateTensor<int64_t>(
                               memory_info_,
                               semantic_tokens_data_.data(), semantic_tokens_data_.size(),
                               audio_tokenizer_output_semantic_tokens_shape_.data(), audio_tokenizer_output_semantic_tokens_shape_.size()),

                           Ort::Value::CreateTensor<int32_t>(
                               memory_info_,
                               global_tokens_data_.data(), global_tokens_data_.size(),
                               audio_tokenizer_output_global_tokens_shape_.data(), audio_tokenizer_output_global_tokens_shape_.size())};
    }

    std::array<float, 16000 * 6> AudioTokenizerImpl::pad_or_trim_audio(const std::vector<float> &mono_audio) const
    {
        TRACE_EVENT("audio_tokenizer", "AudioTokenizer::pad_or_trim_audio");

        std::array<float, 16000 * 6> padded_audio = {};
        size_t audio_size = mono_audio.size();

        if (audio_size > 16000 * 6)
        {
            // Trim the audio
            std::copy(mono_audio.begin(), mono_audio.begin() + 16000 * 6, padded_audio.begin());
        }
        else
        {
            // Pad the audio
            std::copy(mono_audio.begin(), mono_audio.end(), padded_audio.begin());
            std::fill(padded_audio.begin() + audio_size, padded_audio.end(), 0.0f);
        }

        return padded_audio;
    }

    std::array<int32_t, 32> AudioTokenizerImpl::tokenize(const std::vector<float> &mono_audio)
    {
        TRACE_EVENT("audio_tokenizer", "AudioTokenizer::tokenize");

        std::array<float, 16000 * 6> padded_audio = pad_or_trim_audio(mono_audio);
        std::copy(padded_audio.begin(), padded_audio.end(), audio_input_data_.begin());
        std::fill(semantic_tokens_data_.begin(), semantic_tokens_data_.end(), 0);
        std::fill(global_tokens_data_.begin(), global_tokens_data_.end(), 0);

        audio_tokenizer_session_->Run(
            Ort::RunOptions{nullptr},
            audio_tokenizer_input_names_.data(), &input_tensor_, 1,
            audio_tokenizer_output_names_.data(), output_tensors_.data(), output_tensors_.size());

        return global_tokens_data_;
    }

} // namespace spark_tts
