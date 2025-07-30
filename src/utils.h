#include <sndfile.hh>

#include <vector>
#include <filesystem>

namespace spark_tts
{
    std::vector<float> load_reference_audio(const std::filesystem::path &file_path);
    size_t save_generated_audio(const std::filesystem::path &output_path, const std::vector<float> &audio_data);

} // namespace spark_tts
