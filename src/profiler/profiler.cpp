#include "profiler.h"

namespace spark_tts
{
#ifdef ENABLE_PERFETTO

    Profiler::Profiler()
    {
        perfetto::TracingInitArgs args;

        // Only use the in-process backend
        args.backends = perfetto::kInProcessBackend;

        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();
    }

    void Profiler::start(const uint32_t buffer_size_kb)
    {
        perfetto::TraceConfig config;

        config.add_buffers()->set_size_kb(buffer_size_kb);
        auto *ds_cfg = config.add_data_sources()->mutable_config();
        ds_cfg->set_name("track_event");

        tracing_session_ = perfetto::Tracing::NewTrace();
        tracing_session_->Setup(config);
        tracing_session_->StartBlocking();

        running_ = true;
    }

    bool Profiler::stop(const std::string &trace_path)
    {
        if (!running_)
        {
            return false;
        }
        running_ = false;

        // Make sure the last event is closed for this example.
        perfetto::TrackEvent::Flush();

        // Stop tracing and read the trace data.
        tracing_session_->StopBlocking();
        std::vector<char> trace_data(tracing_session_->ReadTraceBlocking());

        // Write the trace data to a file.
        std::ofstream output;
        output.open(trace_path, std::ios::out | std::ios::binary);
        if (!output)
        {
            std::cerr << "Failed to open trace file: " << trace_path << std::endl;
            return false;
        }
        output.write(&trace_data[0], std::streamsize(trace_data.size()));
        output.close();

        return true;
    }

#else
    Profiler::Profiler()
    {
        // No-op
    }

    void Profiler::start(const uint32_t buffer_size_kb)
    {
        // No-op
    }

    bool Profiler::stop(const std::string &trace_path)
    {
        // No-op
        return false;
    }
#endif

} // namespace spark_tts
