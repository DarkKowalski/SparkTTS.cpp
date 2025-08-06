#include "audio_tokenizer.h"

namespace spark_tts
{
    AudioTokenizer::AudioTokenizer(const std::string &wav2vec_model_path,
                                   const std::string &mel_spectrogram_model_path,
                                   const std::string &bicodec_tokenizer_model_path)
        : env_(ORT_LOGGING_LEVEL_ERROR, "AudioTokenizer"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault))
    {
        Ort::SessionOptions session_options;
        std::unordered_map<std::string, std::string> provider_options;
        provider_options["ModelFormat"] = "NeuralNetwork"; // MLProgram crashed on macOS 15.5/Apple Silicon M2
        session_options.AppendExecutionProvider("CoreML", provider_options);

        // Load the ONNX models
        wav2vec_session_ = std::make_unique<Ort::Session>(env_, wav2vec_model_path.c_str(), session_options);
        mel_spectrogram_session_ = std::make_unique<Ort::Session>(env_, mel_spectrogram_model_path.c_str(), session_options);
        bicodec_tokenizer_session_ = std::make_unique<Ort::Session>(env_, bicodec_tokenizer_model_path.c_str(), session_options);
    }

    std::array<float, 16000 * 6> AudioTokenizer::pad_or_trim_audio(const std::vector<float> &mono_audio) const
    {
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

    std::array<int32_t, 32> AudioTokenizer::tokenize(const std::vector<float> &mono_audio)
    {
        auto processed_audio = pad_or_trim_audio(mono_audio);

        std::array<float, 128 * 302> mel_output_data = {};
        Ort::Value mel_input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            processed_audio.data(), processed_audio.size(),
            mel_input_shape_.data(), mel_input_shape_.size());
        Ort::Value mel_output_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            mel_output_data.data(), mel_output_data.size(),
            mel_output_shape_.data(), mel_output_shape_.size());
        mel_spectrogram_session_->Run(Ort::RunOptions{nullptr}, mel_input_names_.data(), &mel_input_tensor, 1, mel_output_names_.data(), &mel_output_tensor, 1);

        std::array<float, 299 * 1024> wav2vec_output_data = {};
        Ort::Value wav2vec_input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            processed_audio.data(), processed_audio.size(),
            wav2vec_input_shape_.data(), wav2vec_input_shape_.size());
        Ort::Value wav2vec_output_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            wav2vec_output_data.data(), wav2vec_output_data.size(),
            wav2vec_output_shape_.data(), wav2vec_output_shape_.size());
        wav2vec_session_->Run(Ort::RunOptions{nullptr}, wav2vec_input_names_.data(), &wav2vec_input_tensor, 1, wav2vec_output_names_.data(), &wav2vec_output_tensor, 1);

        std::array<Ort::Value, 2> bicodec_inputs = {
            Ort::Value::CreateTensor<float>(
                memory_info_,
                wav2vec_output_data.data(), wav2vec_output_data.size(),
                wav2vec_output_shape_.data(), wav2vec_output_shape_.size()),
            Ort::Value::CreateTensor<float>(
                memory_info_,
                mel_output_data.data(), mel_output_data.size(),
                mel_output_shape_.data(), mel_output_shape_.size())};

        std::array<int64_t, 299> semantic_tokens = {};
        std::array<int32_t, 32> global_tokens = {};
        std::array<Ort::Value, 2> bicodec_outputs = {
            Ort::Value::CreateTensor<int64_t>(
                memory_info_,
                semantic_tokens.data(), semantic_tokens.size(),
                bicodec_output_semantic_tokens_shape_.data(), bicodec_output_semantic_tokens_shape_.size()),
            Ort::Value::CreateTensor<int32_t>(
                memory_info_,
                global_tokens.data(), global_tokens.size(),
                bicodec_output_global_tokens_shape_.data(), bicodec_output_global_tokens_shape_.size())};
        bicodec_tokenizer_session_->Run(
            Ort::RunOptions{nullptr},
            bicodec_input_names_.data(), bicodec_inputs.data(), bicodec_inputs.size(),
            bicodec_output_names_.data(), bicodec_outputs.data(), bicodec_outputs.size());

        return global_tokens;
    }

} // namespace spark_tts
