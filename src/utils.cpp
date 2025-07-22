#include "utils.h"

namespace spark_tts
{
    static constexpr size_t audio_sample_rate = 16000;

    std::vector<float> load_reference_audio(const std::filesystem::path &file_path)
    {
        SndfileHandle file_handle(file_path.string());
        if (!file_handle)
        {
            throw std::runtime_error("Failed to open audio file: " + file_path.string() + ", Error: " + file_handle.strError());
        }

        if (file_handle.channels() != 1)
        {
            throw std::runtime_error("Only mono audio is supported, but found " + std::to_string(file_handle.channels()) + " channels.");
        }

        if (file_handle.samplerate() != audio_sample_rate)
        {
            throw std::runtime_error("Unsupported sample rate: " + std::to_string(file_handle.samplerate()) +
                                     ". Expected: " + std::to_string(audio_sample_rate) + " Hz.");
        }

        std::vector<float> audio_data(file_handle.frames() * file_handle.channels());
        sf_count_t frames_read = file_handle.readf(audio_data.data(), file_handle.frames());
        if (frames_read < 0)
        {
            throw std::runtime_error("Error reading audio data: " + std::string(file_handle.strError()));
        }

        return audio_data;
    }

    size_t save_generated_audio(const std::filesystem::path &output_path, const std::vector<float> &audio_data)
    {
        SndfileHandle output_file(output_path.string(), SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_FLOAT, 1, audio_sample_rate);
        if (!output_file)
        {
            throw std::runtime_error("Failed to open output file: " + output_path.string() + ", Error: " + output_file.strError());
        }

        sf_count_t frames_written = output_file.writef(audio_data.data(), audio_data.size());
        if (frames_written < 0)
        {
            throw std::runtime_error("Error writing audio data: " + std::string(output_file.strError()));
        }

        return frames_written;
    }

} // namespace spark_tts
