#include "audio_tokenizer.h"

#include "../profiler/profiler.h"

namespace spark_tts
{
    AudioTokenizer::AudioTokenizer(
        const std::string &wav2vec_model_path,
        const std::string &mel_spectrogram_model_path,
        const std::string &bicodec_tokenizer_model_path) : core_(ov::Core())
    {
        TRACE_EVENT("audio_tokenizer", "AudioTokenizer::AudioTokenizer");

        auto wav2vec_model = core_.read_model(wav2vec_model_path);
        auto mel_spectrogram_model = core_.read_model(mel_spectrogram_model_path);
        auto bicodec_tokenizer_model = core_.read_model(bicodec_tokenizer_model_path);

        wav2vec_ = core_.compile_model(wav2vec_model, device_name_);
        mel_spectrogram_ = core_.compile_model(mel_spectrogram_model, device_name_);
        bicodec_tokenizer_ = core_.compile_model(bicodec_tokenizer_model, device_name_);
    }

    std::array<float, 16000 * 6> AudioTokenizer::pad_or_trim_audio(const std::vector<float> &mono_audio) const
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

    std::array<int32_t, 32> AudioTokenizer::tokenize(const std::vector<float> &mono_audio)
    {
        TRACE_EVENT("audio_tokenizer", "AudioTokenizer::tokenize");

        auto processed_audio = pad_or_trim_audio(mono_audio);

        // mel_input [1, 1, 96000]
        ov::Tensor mel_input = ov::Tensor(ov::element::f32, {1, 1, 96000});
        std::copy(processed_audio.begin(), processed_audio.end(), mel_input.data<float>());

        // wav2vec_input [1, 16000 * 6]
        ov::Tensor wav2vec_input = ov::Tensor(ov::element::f32, {1, 16000 * 6});
        std::copy(processed_audio.begin(), processed_audio.end(), wav2vec_input.data<float>());

        // Run wav2vec model
        auto wav2vec_infer = wav2vec_.create_infer_request();
        wav2vec_infer.set_input_tensor(wav2vec_input);
        wav2vec_infer.infer();

        // Run mel spectrogram model
        auto mel_spectrogram_infer = mel_spectrogram_.create_infer_request();
        mel_spectrogram_infer.set_input_tensor(mel_input);
        mel_spectrogram_infer.infer();

        // Run bicodec tokenizer model
        auto bicodec_tokenizer_infer = bicodec_tokenizer_.create_infer_request();
        bicodec_tokenizer_infer.set_input_tensor(0, wav2vec_infer.get_output_tensor());
        bicodec_tokenizer_infer.set_input_tensor(1, mel_spectrogram_infer.get_output_tensor());
        bicodec_tokenizer_infer.infer();

        // Get outputs
        // auto semantic_tokens = bicodec_tokenizer_infer.get_output_tensor(0);
        auto global_tokens = bicodec_tokenizer_infer.get_output_tensor(1);

        std::array<int32_t, 32> global_token_ids = {};
        std::copy(global_tokens.data<int32_t>(), global_tokens.data<int32_t>() + 32, global_token_ids.begin());
        return global_token_ids;
    }

} // namespace spark_tts
