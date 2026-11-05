using System.Runtime.InteropServices;
using Silk.NET.Core.Native;
using Silk.NET.WebGPU;
using Silk.NET.Windowing;

namespace SurfelGI.Core;

public unsafe class WebGpuContext : IDisposable
{
    public WebGPU Wgpu { get; }
    public Instance* Instance { get; private set; }
    public Surface* Surface { get; private set; }
    public Adapter* Adapter { get; private set; }
    public Device* Device { get; private set; }
    public Queue* Queue { get; private set; }

    public TextureFormat SwapchainFormat { get; private set; } = TextureFormat.Bgra8UnormSrgb;
    public int Width { get; private set; }
    public int Height { get; private set; }
    public uint FrameIndex { get; set; }

    public string? LastError { get; private set; }
    private PfnErrorCallback? _uncapturedErrorCallback;

    public WebGpuContext(IWindow window)
    {
        Wgpu = WebGPU.GetApi();

        var instanceDesc = new InstanceDescriptor();
        Instance = Wgpu.CreateInstance(&instanceDesc);
        if (Instance == null)
            throw new InvalidOperationException("Failed to create WebGPU Instance.");

        Surface = window.CreateWebGPUSurface(Wgpu, Instance);
        if (Surface == null)
            throw new InvalidOperationException("Failed to create WebGPU Surface.");

        RequestAdapter();
        RequestDevice();
        Queue = Wgpu.DeviceGetQueue(Device);

        Width = Math.Max(1, window.FramebufferSize.X);
        Height = Math.Max(1, window.FramebufferSize.Y);
        ConfigureSurface(Width, Height);
    }

    public WebGpuContext()
    {
        Wgpu = WebGPU.GetApi();

        var instanceDesc = new InstanceDescriptor();
        Instance = Wgpu.CreateInstance(&instanceDesc);
        if (Instance == null)
            throw new InvalidOperationException("Failed to create WebGPU Instance.");

        Surface = null;
        RequestAdapter();
        RequestDevice();
        Queue = Wgpu.DeviceGetQueue(Device);

        Width = 1280;
        Height = 720;
    }

    private void RequestAdapter()
    {
        Adapter* result = null;
        var options = new RequestAdapterOptions
        {
            CompatibleSurface = Surface,
            PowerPreference = PowerPreference.HighPerformance
        };

        PfnRequestAdapterCallback callback = (status, adapter, message, _, _) =>
        {
            if (status == RequestAdapterStatus.Success)
                result = adapter;
            else
                Console.Error.WriteLine($"RequestAdapter failed ({status}): {message}");
        };

        var cbInfo = new RequestAdapterCallbackInfo
        {
            Mode = CallbackMode.AllowSpontaneous,
            Callback = Marshal.GetFunctionPointerForDelegate(callback)
        };

        Wgpu.InstanceRequestAdapter(Instance, &options, cbInfo);
        GC.KeepAlive(callback);

        if (result == null)
            throw new InvalidOperationException("Could not acquire WebGPU Adapter.");

        Adapter = result;
    }

    private void RequestDevice()
    {
        Limits limits = default;
        Wgpu.AdapterGetLimits(Adapter, &limits);

        limits.MaxStorageBuffersPerShaderStage = Math.Max(12u, limits.MaxStorageBuffersPerShaderStage);
        limits.MaxComputeWorkgroupSizeX = Math.Max(1024u, limits.MaxComputeWorkgroupSizeX);
        limits.MaxComputeInvocationsPerWorkgroup = Math.Max(1024u, limits.MaxComputeInvocationsPerWorkgroup);

        _uncapturedErrorCallback = (_, type, message, _, _) =>
        {
            LastError = $"[WebGPU {type}] {message}";
            Console.Error.WriteLine(LastError);
        };

        Device* result = null;
        var deviceDesc = new DeviceDescriptor
        {
            RequiredLimits = &limits,
            UncapturedErrorCallbackInfo = new UncapturedErrorCallbackInfo
            {
                Callback = Marshal.GetFunctionPointerForDelegate(_uncapturedErrorCallback)
            }
        };

        PfnRequestDeviceCallback callback = (status, device, message, _, _) =>
        {
            if (status == RequestDeviceStatus.Success)
                result = device;
            else
                Console.Error.WriteLine($"RequestDevice failed ({status}): {message}");
        };

        var cbInfo = new RequestDeviceCallbackInfo
        {
            Mode = CallbackMode.AllowSpontaneous,
            Callback = Marshal.GetFunctionPointerForDelegate(callback)
        };

        Wgpu.AdapterRequestDevice(Adapter, &deviceDesc, cbInfo);
        GC.KeepAlive(callback);

        if (result == null)
            throw new InvalidOperationException("Could not create WebGPU Device.");

        Device = result;
    }

    public void ConfigureSurface(int width, int height)
    {
        Width = Math.Max(1, width);
        Height = Math.Max(1, height);
        if (Surface == null)
            return;

        SurfaceCapabilities caps = default;
        Wgpu.SurfaceGetCapabilities(Surface, Adapter, &caps);
        if (caps.FormatCount > 0 && caps.Formats != null)
        {
            SwapchainFormat = caps.Formats[0];
            for (nuint i = 0; i < caps.FormatCount; i++)
            {
                if (caps.Formats[i] == TextureFormat.Bgra8UnormSrgb || caps.Formats[i] == TextureFormat.Rgba8UnormSrgb)
                {
                    SwapchainFormat = caps.Formats[i];
                    break;
                }
            }
        }

        var config = new SurfaceConfiguration
        {
            Device = Device,
            Format = SwapchainFormat,
            Usage = TextureUsage.RenderAttachment,
            Width = (uint)Width,
            Height = (uint)Height,
            PresentMode = PresentMode.Fifo,
            AlphaMode = CompositeAlphaMode.Auto
        };
        Wgpu.SurfaceConfigure(Surface, &config);
    }

    public void Dispose()
    {
        if (Queue != null) Wgpu.QueueRelease(Queue);
        if (Device != null) Wgpu.DeviceRelease(Device);
        if (Adapter != null) Wgpu.AdapterRelease(Adapter);
        if (Surface != null) Wgpu.SurfaceRelease(Surface);
        if (Instance != null) Wgpu.InstanceRelease(Instance);
        Wgpu.Dispose();
    }
}
