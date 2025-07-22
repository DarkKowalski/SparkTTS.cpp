#include "audio_detokenizer_impl.h"

#include <stdexcept>
#include <iostream>
#include <cstdlib>

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#import "gen/AudioDetokenizer.h"


namespace spark_tts
{
    struct ObjectiveContext {
        const void *model = nullptr;
    };

    AudioDetokenizerImpl::AudioDetokenizerImpl(const std::string &model_path) {
        NSString * model_path_str = [[NSString alloc] initWithUTF8String:model_path.c_str()];
        NSURL * url = [NSURL fileURLWithPath: model_path_str];
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        // MLComputeUnitsCPUAndGPU
        // MLComputeUnitsCPUOnly
        // MLComputeUnitsAll
        // MLComputeUnitsCPUAndNeuralEngine
        config.computeUnits = MLComputeUnitsCPUAndGPU;

        NSError *error = nil;
        const void * model = CFBridgingRetain([[AudioDetokenizer alloc] initWithContentsOfURL:url configuration:config error:&error]);
        if (model == nullptr) {
            throw std::runtime_error("Failed to load CoreML model"
                                     + std::string([error localizedDescription].UTF8String));
        }

        objc_context_ = new ObjectiveContext();
        static_cast<ObjectiveContext *>(objc_context_)->model = model;
    }

    AudioDetokenizerImpl::~AudioDetokenizerImpl() {
        if (objc_context_) {
            CFRelease(static_cast<ObjectiveContext *>(objc_context_)->model);
            delete static_cast<ObjectiveContext *>(objc_context_);
        }
    }

    std::array<float, 16000 * 1> AudioDetokenizerImpl::detokenize(std::array<int64_t, 50> &semantic_tokens,
                                                              std::array<int32_t, 32> &global_tokens)
    {
        // assume objc_context_ is not null and model is loaded
        ObjectiveContext *context = static_cast<ObjectiveContext *>(objc_context_);

        std::array<int32_t, 50> semantic_tokens_i32 = {};
        std::copy(semantic_tokens.begin(), semantic_tokens.end(), semantic_tokens_i32.begin());

        MLMultiArray * semantic_tokens_input = [
            [MLMultiArray alloc] initWithDataPointer: semantic_tokens_i32.data()
                                               shape: @[@1, @50]
                                            dataType: MLMultiArrayDataTypeInt32
                                             strides: @[@50, @1]
                                         deallocator: nil
                                               error: nil
        ];

        MLMultiArray * global_tokens_input = [
            [MLMultiArray alloc] initWithDataPointer: global_tokens.data()
                                               shape: @[@1, @1, @32]
                                            dataType: MLMultiArrayDataTypeInt32
                                             strides: @[@32, @32, @1]
                                         deallocator: nil
                                               error: nil
        ];

        std::array<float, 16000 * 1> audio_output = {};

        @autoreleasepool {
            NSError *error = nil;
            AudioDetokenizerOutput *output = [(__bridge id) context->model predictionFromSemantic_tokens: semantic_tokens_input
                                                                                             global_tokens: global_tokens_input
                                                                                                   error:&error];
            if (error) {
                std::cerr << "Error during CoreML prediction: " << [error localizedDescription].UTF8String << std::endl;
                return {};
            }

            std::memcpy(audio_output.data(), output.wav_recon.dataPointer, output.wav_recon.count * sizeof(float));
        }

        return audio_output;
    }

} // namespace spark_tts
