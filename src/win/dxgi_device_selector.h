#pragma once

#include <string>
#include <optional>

#include <dxgi1_6.h>
#include <wrl/client.h>

#pragma comment(lib, "dxgi.lib")

namespace spark_tts
{
    class DXGIDeviceSelector
    {
    public:
        DXGIDeviceSelector();

    public:
        int get_high_performance_adapter_index() const { return high_performance_adapter_index_; }
        std::string get_high_performance_adapter_description() const { return high_performance_adapter_description_; }

    private:
        std::optional<DXGI_ADAPTER_DESC1> enumerate_high_performance_adapter();
        bool select_high_performance_adapter();

    private:
        int high_performance_adapter_index_ = 0;
        std::string high_performance_adapter_description_;
    };
} // namespace spark_tts
