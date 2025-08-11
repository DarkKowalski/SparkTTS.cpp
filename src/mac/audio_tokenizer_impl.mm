#include "audio_tokenizer_impl.h"

#include <stdexcept>
#include <iostream>
#include <cstdlib>

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#import "gen/AudioTokenizer.h"

namespace spark_tts
{
    struct ObjectiveContext {
        const void *model = nullptr;
    };

    AudioTokenizerImpl::AudioTokenizerImpl(const std::string &model_path)
    {
        NSString * model_path_str = [[NSString alloc] initWithUTF8String:model_path.c_str()];
        NSURL * url = [NSURL fileURLWithPath: model_path_str];
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        // MLComputeUnitsCPUAndGPU
        // MLComputeUnitsCPUOnly
        // MLComputeUnitsAll
        // MLComputeUnitsCPUAndNeuralEngine
        config.computeUnits = MLComputeUnitsCPUAndGPU;

        NSError *error = nil;
        const void * model = CFBridgingRetain([[AudioTokenizer alloc] initWithContentsOfURL:url configuration:config error:&error]);
        if (model == nullptr) {
            throw std::runtime_error("Failed to load CoreML model"
                                     + std::string([error localizedDescription].UTF8String));
        }

        objc_context_ = new ObjectiveContext();
        static_cast<ObjectiveContext *>(objc_context_)->model = model;
    }

    AudioTokenizerImpl::~AudioTokenizerImpl()
    {
        if (objc_context_) {
            CFRelease(static_cast<ObjectiveContext *>(objc_context_)->model);
            delete static_cast<ObjectiveContext *>(objc_context_);
        }
    }

    std::array<float, 16000 * 6> AudioTokenizerImpl::pad_or_trim_audio(const std::vector<float> &mono_audio) const
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

    std::array<int32_t, 32> AudioTokenizerImpl::tokenize(const std::vector<float> &mono_audio)
    {
        auto processed_audio = pad_or_trim_audio(mono_audio);

        ObjectiveContext *context = static_cast<ObjectiveContext *>(objc_context_);

        MLMultiArray *audio_input = [[MLMultiArray alloc] initWithDataPointer:processed_audio.data()
                                                                       shape:@[@96000]
                                                                    dataType:MLMultiArrayDataTypeFloat32
                                                                     strides:@[@1]
                                                                 deallocator:nil
                                                                       error:nil];

        std::array<int32_t, 32> global_tokens_output = {};
        
        @autoreleasepool {
            NSError *error = nil;
            AudioTokenizerOutput *output = [(__bridge id) context->model predictionFromAudio_input:audio_input error:&error];
            if (error) {
                throw std::runtime_error("Error during CoreML prediction: "
                                         + std::string([error localizedDescription].UTF8String));
            }

            std::memcpy(global_tokens_output.data(), output.global_tokens.dataPointer, output.global_tokens.count * sizeof(int32_t));
        }

        return global_tokens_output;
    }

} // namespace spark_tts
