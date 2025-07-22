#ifdef __cplusplus

#include <sndfile.hh>

#include <vector>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>
namespace spark_tts
{
    std::vector<float> load_reference_audio(const std::filesystem::path &file_path);
    size_t save_generated_audio(const std::filesystem::path &output_path, const std::vector<float> &audio_data);

} // namespace spark_tts

#endif

#ifdef __cplusplus
extern "C"
{
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#endif
    float *util_load_reference_audio(const char *file_path, size_t *audio_size);
    size_t util_save_generated_audio(const char *output_path, const float *audio_data, size_t audio_size);

#ifdef __cplusplus
}
#endif
