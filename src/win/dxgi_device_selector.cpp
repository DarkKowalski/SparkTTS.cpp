#include "dxgi_device_selector.h"

#include <dxgi1_6.h>
#include <stdexcept>
#include <iostream>
#include <locale>
#include <codecvt>

namespace spark_tts
{
    DXGIDeviceSelector::DXGIDeviceSelector()
    {
        select_high_performance_adapter();
    }

    std::optional<DXGI_ADAPTER_DESC1> DXGIDeviceSelector::enumerate_high_performance_adapter()
    {
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory1;
        Microsoft::WRL::ComPtr<IDXGIFactory6> factory6;

        HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory1));
        if (FAILED(hr))
        {
            std::cerr << "Failed to create DXGI factory: " << std::hex << hr << std::endl;
            return std::nullopt; // Return nullopt if factory creation fails
        }

        factory1.As(&factory6);
        if (!factory6)
        {
            std::cerr << "Failed to query IDXGIFactory6 interface." << std::endl;
            return std::nullopt; // Return nullopt if factory6 query fails
        }

        UINT preference_index = 0;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> current_adapter;
        // Enumerate High Performance Adapter
        while (true)
        {
            hr = factory6->EnumAdapterByGpuPreference(preference_index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&current_adapter));
            if (hr == DXGI_ERROR_NOT_FOUND)
            {
                break; // No more adapters
            }

            if (FAILED(hr))
            {
                preference_index++;
                continue; // Skip this adapter if it fails to enumerate
            }

            DXGI_ADAPTER_DESC1 desc;
            hr = current_adapter->GetDesc1(&desc);
            if (FAILED(hr))
            {
                std::cerr << "Failed to get adapter description: " << std::hex << hr << std::endl;
                preference_index++;
                continue; // Skip this adapter if it fails to get description
            }

            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            {
                preference_index++;
                continue; // Skip software adapters
            }

            return {desc}; // Return the found adapter description
        }

        return std::nullopt; // Return nullopt if no high-performance adapter is found
    }

    bool DXGIDeviceSelector::select_high_performance_adapter()
    {
        std::optional<DXGI_ADAPTER_DESC1> target_desc = enumerate_high_performance_adapter();
        if (!target_desc.has_value())
        {
            std::cerr << "No high-performance adapter found." << std::endl;
            return false;
        }

        Microsoft::WRL::ComPtr<IDXGIFactory> factory;
        HRESULT hr = CreateDXGIFactory(IID_PPV_ARGS(&factory));
        if (FAILED(hr))
        {
            std::cerr << "Failed to create DXGI factory: " << std::hex << hr << std::endl;
            return false;
        }

        UINT index = 0;
        Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
        while (true)
        {
            hr = factory->EnumAdapters(index, &adapter);
            if (hr == DXGI_ERROR_NOT_FOUND)
            {
                break; // No more adapters
            }

            if (FAILED(hr))
            {
                std::cerr << "Failed to enumerate adapters: " << std::hex << hr << std::endl;
                index++;
                continue; // Skip this adapter if it fails to enumerate
            }

            DXGI_ADAPTER_DESC desc;
            hr = adapter->GetDesc(&desc);
            if (FAILED(hr))
            {
                std::cerr << "Failed to get adapter description: " << std::hex << hr << std::endl;
                index++;
                continue; // Skip this adapter if it fails to get description
            }

            // By LUID
            if (desc.AdapterLuid.HighPart == target_desc->AdapterLuid.HighPart &&
                desc.AdapterLuid.LowPart == target_desc->AdapterLuid.LowPart)
            {
                // Found
                high_performance_adapter_index_ = static_cast<int>(index);
                high_performance_adapter_description_ = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(desc.Description);
                return true; // Successfully selected the high-performance adapter
            }

            index++;
        }

        std::cerr << "High-performance adapter not found in the enumerated adapters." << std::endl;

        return false;
    }
}
