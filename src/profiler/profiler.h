#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <stdexcept>

#ifdef ENABLE_PERFETTO

#include "perfetto_categories.h"

#else

#define TRACE_EVENT(...)
#define TRACE_EVENT_BEGIN(...)
#define TRACE_EVENT_END(...)
#define TRACE_EVENT_INSTANT(...)
#define TRACE_COUNTER(...)

#endif

namespace spark_tts
{
    class Profiler
    {
    public:
        static Profiler &instance()
        {
            static Profiler instance;
            return instance;
        }

        // Delete copy constructor and assignment operator
        Profiler(const Profiler &) = delete;
        Profiler &operator=(const Profiler &) = delete;

        // Delete move constructor and assignment operator
        Profiler(Profiler &&) = delete;
        Profiler &operator=(Profiler &&) = delete;

    private:
        Profiler();

    public:
        void start(const uint32_t buffer_size_kb);
        bool stop(const std::string &trace_path);
        bool running() const { return running_; }

    private:
        bool running_ = false;

#ifdef ENABLE_PERFETTO
        std::unique_ptr<perfetto::TracingSession> tracing_session_;
#endif
    };
} // namespace spark_tts
