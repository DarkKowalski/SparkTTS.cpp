#include "synthesizer.h"

#if defined(_WIN32) || defined(_WIN64)
#include "win/audio_detokenizer_impl.h"
#include "win/audio_tokenizer_impl.h"
#elif defined(__APPLE__)
#include "mac/audio_detokenizer_impl.h"
#include "mac/audio_tokenizer_impl.h"
#endif

#include "profiler/profiler.h"

namespace spark_tts
{

    Synthesizer::Synthesizer()
    {
        Profiler::instance().start(1024 * 32); // Start profiler with 32 MB buffer size

        {
            TRACE_EVENT("synthesizer", "Initialize llama backend");
            llama_backend_init();
        }
    }

    Synthesizer::~Synthesizer()
    {
        {
            TRACE_EVENT("synthesizer", "Unload llama backend");
            llama_backend_free();
        }

        Profiler::instance().stop("spark_tts.pftrace"); // Stop profiler and write trace to file
    }

    void Synthesizer::init_voice_feature_extraction(const std::string &audio_tokenizer_model_path)

    {
        TRACE_EVENT("synthesizer", "init_voice_feature_extraction");

        audio_tokenizer_ = std::make_unique<AudioTokenizerImpl>(audio_tokenizer_model_path);
    }

    void Synthesizer::init_text_to_speech(const std::string &audio_detokenizer_model_path,
                                          const std::string &transformer_model_path,
                                          const std::string &tokenizer_path,
                                          const uint32_t transformer_n_ctx,
                                          const size_t overlapped_semantic_tokens)
    {
        TRACE_EVENT("synthesizer", "init_text_to_speech");

        if (overlapped_semantic_tokens >= 25)
        {
            throw std::invalid_argument("overlapped_semantic_tokens must be less than 25");
        }
        overlapped_semantic_tokens_ = overlapped_semantic_tokens;

        audio_detokenizer_ = std::make_unique<AudioDetokenizerImpl>(audio_detokenizer_model_path);

        auto transformer_params = Transformer::Params();
        transformer_params.ctx_params.n_ctx = transformer_n_ctx;
        transformer_ = std::make_unique<Transformer>(transformer_model_path, tokenizer_path, transformer_params);

        token_buffer_ = std::make_unique<TokenBuffer>(50, overlapped_semantic_tokens_);
    }

    // Must call init_voice_feature_extraction before this method
    std::array<int32_t, 32> Synthesizer::extract_voice_features(const std::vector<float> &audio_data)
    {
        TRACE_EVENT("synthesizer", "extract_voice_features");
        return audio_tokenizer_->tokenize(audio_data);
    }

    std::vector<float> Synthesizer::synthesize(std::array<int32_t, 32> &voice_features)
    {
        TRACE_EVENT("synthesizer", "synthesize");

        const std::vector<int64_t> &front_buffer = token_buffer_->front_buffer();
        if (front_buffer.size() <= overlapped_semantic_tokens_)
        {
            // Not enough tokens to generate audio, return empty vector
            return {};
        }

        std::array<int64_t, 50> semantic_tokens_array = {};
        std::copy(front_buffer.begin(), front_buffer.end(), semantic_tokens_array.begin());

        // If buffer is not full, it must be the last generation, don't trim the tail
        const size_t tail_trim_tokens = front_buffer.size() == 50 ? overlapped_semantic_tokens_ : 50 - front_buffer.size();

        // If this is the first sample, we don't trim the head
        const size_t head_trim_tokens = synthesized_frames_ == 0 ? 0 : overlapped_semantic_tokens_;

        token_buffer_->flip();

        auto sample = audio_detokenizer_->detokenize(semantic_tokens_array, voice_features);

        constexpr size_t samples_per_token = 320; // 50 tokens per second, 320 samples per token
        std::vector<float> generated_audio(sample.begin() + head_trim_tokens * samples_per_token,
                                           sample.end() - tail_trim_tokens * samples_per_token);

        synthesized_frames_++;

        return generated_audio;
    }

    Transformer::DecodeCallbackAction Synthesizer::decode_callback(std::string &semantic_tokens,
                                                                   std::array<int32_t, 32> &voice_features,
                                                                   TextToSpeechCallback &callback)

    {
        TRACE_EVENT("synthesizer", "decode_callback");

        auto semantic_token_ids = extract_semantic_token_ids(semantic_tokens);

        bool ready_to_synthesize = token_buffer_->add_tokens(semantic_token_ids);
        if (!ready_to_synthesize)
        {
            // If the buffer is not full, we can continue decoding
            return Transformer::DecodeCallbackAction::Continue;
        }

        auto audio_output = synthesize(voice_features);

        return callback(audio_output) ? Transformer::DecodeCallbackAction::Continue : Transformer::DecodeCallbackAction::Stop;
    }

    // Must call init_text_to_speech before this method
    void Synthesizer::text_to_speech(const std::string &text,
                                     std::array<int32_t, 32> &voice_features,
                                     const size_t n_sec,
                                     TextToSpeechCallback &callback)
    {
        TRACE_EVENT("synthesizer", "text_to_speech");

        const std::string prompt = assemble_prompt(stringify_global_tokens(voice_features), text);
        const size_t n_predict = n_sec * (50 + overlapped_semantic_tokens_);

        // Store the lambda in a variable to create an lvalue
        Transformer::DecodeCallback decode_cb = [&](std::string &semantic_tokens) -> Transformer::DecodeCallbackAction
        {
            // Decode the text and call the callback
            return decode_callback(semantic_tokens, voice_features, callback);
        };

        synthesized_frames_ = 0;                         // Reset the synthesized frames count
        token_buffer_->clear();                          // Clear the token buffer before starting a new inference
        constexpr size_t first_callback_tokens = 50 + 1; // The first token cannot generate audio
        const size_t callback_tokens = 50 - overlapped_semantic_tokens_;
        transformer_->infer(prompt, n_predict, callback_tokens, first_callback_tokens, decode_cb);

        auto last_audio_output = synthesize(voice_features);
        if (!last_audio_output.empty())
        {
            callback(last_audio_output);
        }
    }

} // namespace spark_tts
