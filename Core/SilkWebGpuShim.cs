using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Silk.NET.Core.Native
{
    public enum NativeStringEncoding { UTF8 }

    public static unsafe class SilkMarshal
    {
        public static nint StringToPtr(string? str, NativeStringEncoding encoding = NativeStringEncoding.UTF8)
            => str == null ? 0 : Marshal.StringToCoTaskMemUTF8(str);

        public static void Free(nint ptr)
        {
            if (ptr != 0) Marshal.FreeCoTaskMem(ptr);
        }

        public static string? PtrToString(nint ptr)
            => ptr == 0 ? null : Marshal.PtrToStringUTF8(ptr);
    }
}

namespace Silk.NET.Maths
{
    public struct Vector2D<T> where T : struct
    {
        public T X;
        public T Y;
        public Vector2D(T x, T y) { X = x; Y = y; }
    }
}

namespace Silk.NET.Input
{
    public enum MouseButton { Left = 0, Right = 1, Middle = 2 }
    public enum Key
    {
        Unknown = 0,
        Number1 = 49, Number2 = 50, Number3 = 51, Number4 = 52, Number5 = 53, Number6 = 54,
        M = 77, R = 82
    }

    public struct ScrollWheel
    {
        public float X;
        public float Y;
    }

    public interface IMouse
    {
        Vector2 Position { get; }
        event Action<IMouse, Vector2>? MouseMove;
        event Action<IMouse, MouseButton>? MouseDown;
        event Action<IMouse, MouseButton>? MouseUp;
        event Action<IMouse, ScrollWheel>? Scroll;
    }

    public interface IKeyboard
    {
        event Action<IKeyboard, Key, int>? KeyDown;
    }

    public interface IInputContext
    {
        IReadOnlyList<IMouse> Mice { get; }
        IReadOnlyList<IKeyboard> Keyboards { get; }
    }
}

namespace Silk.NET.Windowing
{
    using Silk.NET.Input;
    using Silk.NET.Maths;
    using Silk.NET.WebGPU;

    public struct GraphicsAPI
    {
        public static GraphicsAPI None => default;
    }

    public struct WindowOptions
    {
        public string Title;
        public Vector2D<int> Size;
        public GraphicsAPI API;
        public bool VSync;

        public static WindowOptions Default => new()
        {
            Title = "Window",
            Size = new Vector2D<int>(1280, 720),
            VSync = true
        };
    }

    public unsafe interface IWindow : IDisposable
    {
        Vector2D<int> FramebufferSize { get; }
        event Action? Load;
        event Action<double>? Render;
        event Action<Vector2D<int>>? FramebufferResize;
        event Action? Closing;
        Surface* CreateWebGPUSurface(WebGPU api, Instance* instance);
        IInputContext CreateInput();
        void Run();
        void Close();
    }

    public static class Window
    {
        public static IWindow Create(WindowOptions options) => new GlfwWindowImpl(options);
    }

    internal sealed unsafe class GlfwWindowImpl : IWindow, IInputContext, IMouse, IKeyboard
    {
        private WindowOptions _opts;
        private nint _window;
        public Vector2D<int> FramebufferSize { get; private set; }
        public event Action? Load;
        public event Action<double>? Render;
        public event Action<Vector2D<int>>? FramebufferResize;
        public event Action? Closing;

        public IReadOnlyList<IMouse> Mice => new IMouse[] { this };
        public IReadOnlyList<IKeyboard> Keyboards => new IKeyboard[] { this };

        public Vector2 Position { get; private set; }
        public event Action<IMouse, Vector2>? MouseMove;
        public event Action<IMouse, MouseButton>? MouseDown;
        public event Action<IMouse, MouseButton>? MouseUp;
        public event Action<IMouse, ScrollWheel>? Scroll;
        public event Action<IKeyboard, Key, int>? KeyDown;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void GlfwCursorPosFun(nint win, double x, double y);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void GlfwMouseButtonFun(nint win, int button, int action, int mods);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void GlfwScrollFun(nint win, double xoff, double yoff);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void GlfwKeyFun(nint win, int key, int scancode, int action, int mods);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void GlfwFramebufferSizeFun(nint win, int w, int h);

        private GlfwCursorPosFun? _cursorCb;
        private GlfwMouseButtonFun? _mouseBtnCb;
        private GlfwScrollFun? _scrollCb;
        private GlfwKeyFun? _keyCb;
        private GlfwFramebufferSizeFun? _fbSizeCb;

        private const string GlfwLib = "libglfw.so.3";
        [DllImport(GlfwLib)] private static extern int glfwInit();
        [DllImport(GlfwLib)] private static extern void glfwTerminate();
        [DllImport(GlfwLib)] private static extern int glfwGetPlatform();
        [DllImport(GlfwLib)] private static extern void glfwWindowHint(int hint, int value);
        [DllImport(GlfwLib)] private static extern nint glfwCreateWindow(int width, int height, [MarshalAs(UnmanagedType.LPUTF8Str)] string title, nint monitor, nint share);
        [DllImport(GlfwLib)] private static extern void glfwDestroyWindow(nint window);
        [DllImport(GlfwLib)] private static extern int glfwWindowShouldClose(nint window);
        [DllImport(GlfwLib)] private static extern void glfwSetWindowShouldClose(nint window, int value);
        [DllImport(GlfwLib)] private static extern void glfwPollEvents();
        [DllImport(GlfwLib)] private static extern double glfwGetTime();
        [DllImport(GlfwLib)] private static extern void glfwGetFramebufferSize(nint window, out int width, out int height);
        [DllImport(GlfwLib)] private static extern nint glfwGetX11Display();
        [DllImport(GlfwLib)] private static extern nuint glfwGetX11Window(nint window);
        [DllImport(GlfwLib)] private static extern nint glfwGetWaylandDisplay();
        [DllImport(GlfwLib)] private static extern nint glfwGetWaylandWindow(nint window);
        [DllImport(GlfwLib)] private static extern nint glfwSetCursorPosCallback(nint window, GlfwCursorPosFun cb);
        [DllImport(GlfwLib)] private static extern nint glfwSetMouseButtonCallback(nint window, GlfwMouseButtonFun cb);
        [DllImport(GlfwLib)] private static extern nint glfwSetScrollCallback(nint window, GlfwScrollFun cb);
        [DllImport(GlfwLib)] private static extern nint glfwSetKeyCallback(nint window, GlfwKeyFun cb);
        [DllImport(GlfwLib)] private static extern nint glfwSetFramebufferSizeCallback(nint window, GlfwFramebufferSizeFun cb);

        public GlfwWindowImpl(WindowOptions opts)
        {
            _opts = opts;
            FramebufferSize = opts.Size;
        }

        public IInputContext CreateInput() => this;

        public Surface* CreateWebGPUSurface(WebGPU api, Instance* instance)
        {
            const int GLFW_PLATFORM_WAYLAND = 0x00060003;
            int platform = glfwGetPlatform();

            if (platform == GLFW_PLATFORM_WAYLAND)
            {
                nint wlDisplay = glfwGetWaylandDisplay();
                nint wlSurface = glfwGetWaylandWindow(_window);
                if (wlDisplay != 0 && wlSurface != 0)
                {
                    var wlDesc = new SurfaceSourceWaylandSurface
                    {
                        Chain = new ChainedStruct { SType = SType.SurfaceSourceWaylandSurface },
                        Display = (void*)wlDisplay,
                        Surface = (void*)wlSurface
                    };
                    var surfDesc = new SurfaceDescriptor
                    {
                        NextInChain = (ChainedStruct*)&wlDesc
                    };
                    return api.InstanceCreateSurface(instance, &surfDesc);
                }
            }

            nint x11Display = glfwGetX11Display();
            nuint x11Window = glfwGetX11Window(_window);
            if (x11Display != 0 && x11Window != 0)
            {
                var x11Desc = new SurfaceDescriptorFromXlibWindow
                {
                    Chain = new ChainedStruct { SType = SType.SurfaceSourceXlibWindow },
                    Display = (void*)x11Display,
                    Window = (ulong)x11Window
                };
                var surfDesc = new SurfaceDescriptor
                {
                    NextInChain = (ChainedStruct*)&x11Desc
                };
                return api.InstanceCreateSurface(instance, &surfDesc);
            }

            throw new InvalidOperationException($"Unable to acquire native display/window handles from GLFW (platform=0x{platform:X}).");
        }

        public void Run()
        {
            if (glfwInit() == 0)
                throw new InvalidOperationException("glfwInit failed.");

            glfwWindowHint(0x00022001 /* GLFW_CLIENT_API */, 0 /* GLFW_NO_API */);
            _window = glfwCreateWindow(_opts.Size.X, _opts.Size.Y, _opts.Title, 0, 0);
            if (_window == 0)
                throw new InvalidOperationException("glfwCreateWindow failed.");

            glfwGetFramebufferSize(_window, out int fbW, out int fbH);
            if (fbW > 0 && fbH > 0)
                FramebufferSize = new Vector2D<int>(fbW, fbH);

            _cursorCb = (_, x, y) =>
            {
                Position = new Vector2((float)x, (float)y);
                MouseMove?.Invoke(this, Position);
            };
            _mouseBtnCb = (_, btn, action, _) =>
            {
                var mb = btn == 0 ? MouseButton.Left : btn == 1 ? MouseButton.Right : MouseButton.Middle;
                if (action == 1) MouseDown?.Invoke(this, mb);
                else if (action == 0) MouseUp?.Invoke(this, mb);
            };
            _scrollCb = (_, xoff, yoff) => Scroll?.Invoke(this, new ScrollWheel { X = (float)xoff, Y = (float)yoff });
            _keyCb = (_, key, sc, action, _) =>
            {
                if (action == 1) KeyDown?.Invoke(this, (Key)key, sc);
            };
            _fbSizeCb = (_, w, h) =>
            {
                FramebufferSize = new Vector2D<int>(w, h);
                FramebufferResize?.Invoke(FramebufferSize);
            };

            glfwSetCursorPosCallback(_window, _cursorCb);
            glfwSetMouseButtonCallback(_window, _mouseBtnCb);
            glfwSetScrollCallback(_window, _scrollCb);
            glfwSetKeyCallback(_window, _keyCb);
            glfwSetFramebufferSizeCallback(_window, _fbSizeCb);

            Load?.Invoke();

            double last = glfwGetTime();
            while (glfwWindowShouldClose(_window) == 0)
            {
                glfwPollEvents();
                double now = glfwGetTime();
                Render?.Invoke(now - last);
                last = now;
            }

            Closing?.Invoke();
        }

        public void Close()
        {
            if (_window != 0)
                glfwSetWindowShouldClose(_window, 1);
        }

        public void Dispose()
        {
            if (_window != 0)
            {
                glfwDestroyWindow(_window);
                _window = 0;
            }
            glfwTerminate();
        }
    }
}

namespace Silk.NET.WebGPU
{
    public struct Instance { }
    public struct Adapter { }
    public struct Device { }
    public struct Queue { }
    public struct Surface { }
    public struct Buffer { }
    public struct Texture { }
    public struct TextureView { }
    public struct Sampler { }
    public struct ShaderModule { }
    public struct BindGroupLayout { }
    public struct BindGroup { }
    public struct PipelineLayout { }
    public struct ComputePipeline { }
    public struct RenderPipeline { }
    public struct CommandEncoder { }
    public struct ComputePassEncoder { }
    public struct RenderPassEncoder { }
    public struct CommandBuffer { }

    public struct Future
    {
        public ulong Id;
    }

    public enum SType : uint
    {
        Invalid = 0,
        ShaderSourceSPIRV = 1,
        ShaderSourceWGSL = 2,
        RenderPassMaxDrawCount = 3,
        SurfaceSourceMetalLayer = 4,
        SurfaceSourceWindowsHWND = 5,
        SurfaceSourceXlibWindow = 6,
        SurfaceSourceWaylandSurface = 7,
        SurfaceSourceAndroidNativeWindow = 8,
        SurfaceSourceXCBWindow = 9
    }

    public enum CallbackMode : uint
    {
        WaitAnyOnly = 1,
        AllowProcessEvents = 2,
        AllowSpontaneous = 3
    }

    public unsafe struct StringView
    {
        public byte* Data;
        public nuint Length;

        public StringView(byte* data)
        {
            Data = data;
            Length = data == null ? 0 : unchecked((nuint)(-1));
        }

        public StringView(byte* data, nuint length)
        {
            Data = data;
            Length = length;
        }

        public static implicit operator StringView(byte* ptr) => new(ptr);

        public override string ToString()
        {
            if (Data == null || Length == 0) return string.Empty;
            if (Length == unchecked((nuint)(-1)))
                return Marshal.PtrToStringUTF8((nint)Data) ?? string.Empty;
            return Marshal.PtrToStringUTF8((nint)Data, (int)Length) ?? string.Empty;
        }
    }

    public enum PowerPreference : uint { Undefined = 0, LowPower = 1, HighPerformance = 2 }
    public enum RequestAdapterStatus : uint { Undefined = 0, Success = 1, CallbackCancelled = 2, Unavailable = 3, Error = 4 }
    public enum RequestDeviceStatus : uint { Undefined = 0, Success = 1, CallbackCancelled = 2, Error = 3 }
    public enum ErrorType : uint { NoError = 1, Validation = 2, OutOfMemory = 3, Internal = 4, Unknown = 5 }
    public enum PresentMode : uint { Undefined = 0, Fifo = 1, FifoRelaxed = 2, Immediate = 3, Mailbox = 4 }
    public enum CompositeAlphaMode : uint { Auto = 0, Opaque = 1, Premultiplied = 2, Unpremultiplied = 3, Inherit = 4 }
    public enum SurfaceGetCurrentTextureStatus : uint
    {
        Undefined = 0,
        Success = 1,
        SuccessOptimal = 1,
        SuccessSuboptimal = 2,
        Timeout = 3,
        Outdated = 4,
        Lost = 5,
        Error = 6
    }

    [Flags]
    public enum BufferUsage : ulong
    {
        None = 0,
        MapRead = 1,
        MapWrite = 2,
        CopySrc = 4,
        CopyDst = 8,
        Index = 16,
        Vertex = 32,
        Uniform = 64,
        Storage = 128,
        Indirect = 256,
        QueryResolve = 512
    }

    [Flags]
    public enum TextureUsage : ulong
    {
        None = 0,
        CopySrc = 1,
        CopyDst = 2,
        TextureBinding = 4,
        StorageBinding = 8,
        RenderAttachment = 16,
        TransientAttachment = 32
    }

    public enum TextureDimension : uint { Undefined = 0, Dimension1D = 1, Dimension2D = 2, Dimension3D = 3 }
    public enum TextureViewDimension : uint
    {
        Undefined = 0,
        Dimension1D = 1,
        Dimension2D = 2,
        Dimension2DArray = 3,
        DimensionCube = 4,
        DimensionCubeArray = 5,
        Dimension3D = 6
    }
    public enum TextureAspect : uint { Undefined = 0, All = 1, StencilOnly = 2, DepthOnly = 3 }
    public enum TextureFormat : uint
    {
        Undefined = 0,
        Rgba8Unorm = 0x16,
        Rgba8UnormSrgb = 0x17,
        Bgra8Unorm = 0x1B,
        Bgra8UnormSrgb = 0x1C,
        Rgba16Float = 0x28,
        Rgba32Float = 0x29,
        Depth24Plus = 0x2E,
        Depth32Float = 0x30
    }

    public enum AddressMode : uint { Undefined = 0, ClampToEdge = 1, Repeat = 2, MirrorRepeat = 3 }
    public enum FilterMode : uint { Undefined = 0, Nearest = 1, Linear = 2 }
    public enum MipmapFilterMode : uint { Undefined = 0, Nearest = 1, Linear = 2 }
    public enum CompareFunction : uint
    {
        Undefined = 0, Never = 1, Less = 2, Equal = 3, LessEqual = 4, Greater = 5, NotEqual = 6, GreaterEqual = 7, Always = 8
    }

    [Flags]
    public enum ShaderStage : ulong { None = 0, Vertex = 1, Fragment = 2, Compute = 4 }
    public enum BufferBindingType : uint { BindingNotUsed = 0, Undefined = 1, Uniform = 2, Storage = 3, ReadOnlyStorage = 4 }
    public enum SamplerBindingType : uint { BindingNotUsed = 0, Undefined = 1, Filtering = 2, NonFiltering = 3, Comparison = 4 }
    public enum TextureSampleType : uint { BindingNotUsed = 0, Undefined = 1, Float = 2, UnfilterableFloat = 3, Depth = 4, Sint = 5, Uint = 6 }
    public enum StorageTextureAccess : uint { BindingNotUsed = 0, Undefined = 1, WriteOnly = 2, ReadOnly = 3, ReadWrite = 4 }

    public enum VertexStepMode : uint { Undefined = 0, Vertex = 1, Instance = 2 }
    public enum VertexFormat : uint
    {
        Undefined = 0,
        Float32 = 0x1C,
        Float32x2 = 0x1D,
        Float32x3 = 0x1E,
        Float32x4 = 0x1F,
        Uint32 = 0x20
    }
    public enum PrimitiveTopology : uint { Undefined = 0, PointList = 1, LineList = 2, LineStrip = 3, TriangleList = 4, TriangleStrip = 5 }
    public enum IndexFormat : uint { Undefined = 0, Uint16 = 1, Uint32 = 2 }
    public enum FrontFace : uint { Undefined = 0, Ccw = 1, Cw = 2 }
    public enum CullMode : uint { Undefined = 0, None = 1, Front = 2, Back = 3 }
    public enum BlendOperation : uint { Undefined = 0, Add = 1, Subtract = 2, ReverseSubtract = 3, Min = 4, Max = 5 }
    public enum BlendFactor : uint
    {
        Undefined = 0, Zero = 1, One = 2, Src = 3, OneMinusSrc = 4, SrcAlpha = 5, OneMinusSrcAlpha = 6, Dst = 7, OneMinusDst = 8, DstAlpha = 9, OneMinusDstAlpha = 10
    }
    [Flags]
    public enum ColorWriteMask : ulong { None = 0, Red = 1, Green = 2, Blue = 4, Alpha = 8, All = 15 }
    public enum LoadOp : uint { Undefined = 0, Load = 1, Clear = 2 }
    public enum StoreOp : uint { Undefined = 0, Store = 1, Discard = 2 }
    public enum StencilOperation : uint { Undefined = 0, Keep = 1, Zero = 2, Replace = 3, Invert = 4, IncrementClamp = 5, DecrementClamp = 6, IncrementWrap = 7, DecrementWrap = 8 }

    public unsafe struct ChainedStruct
    {
        public ChainedStruct* Next;
        public SType SType;
    }

    public unsafe struct InstanceDescriptor
    {
        public ChainedStruct* NextInChain;
        public nuint RequiredFeatureCount;
        public void* RequiredFeatures;
        public void* RequiredLimits;
    }

    public unsafe struct SurfaceDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
    }

    public unsafe struct SurfaceDescriptorFromXlibWindow
    {
        public ChainedStruct Chain;
        public void* Display;
        public ulong Window;
    }

    public unsafe struct SurfaceSourceWaylandSurface
    {
        public ChainedStruct Chain;
        public void* Display;
        public void* Surface;
    }

    public unsafe struct RequestAdapterOptions
    {
        public ChainedStruct* NextInChain;
        public uint FeatureLevel;
        public PowerPreference PowerPreference;
        [MarshalAs(UnmanagedType.I4)]
        public bool ForceFallbackAdapter;
        public uint BackendType;
        public Surface* CompatibleSurface;
    }

    public unsafe struct Limits
    {
        public ChainedStruct* NextInChain;
        public uint MaxTextureDimension1D;
        public uint MaxTextureDimension2D;
        public uint MaxTextureDimension3D;
        public uint MaxTextureArrayLayers;
        public uint MaxBindGroups;
        public uint MaxBindGroupsPlusVertexBuffers;
        public uint MaxBindingsPerBindGroup;
        public uint MaxDynamicUniformBuffersPerPipelineLayout;
        public uint MaxDynamicStorageBuffersPerPipelineLayout;
        public uint MaxSampledTexturesPerShaderStage;
        public uint MaxSamplersPerShaderStage;
        public uint MaxStorageBuffersPerShaderStage;
        public uint MaxStorageTexturesPerShaderStage;
        public uint MaxUniformBuffersPerShaderStage;
        public ulong MaxUniformBufferBindingSize;
        public ulong MaxStorageBufferBindingSize;
        public uint MinUniformBufferOffsetAlignment;
        public uint MinStorageBufferOffsetAlignment;
        public uint MaxVertexBuffers;
        public ulong MaxBufferSize;
        public uint MaxVertexAttributes;
        public uint MaxVertexBufferArrayStride;
        public uint MaxInterStageShaderVariables;
        public uint MaxColorAttachments;
        public uint MaxColorAttachmentBytesPerSample;
        public uint MaxComputeWorkgroupStorageSize;
        public uint MaxComputeInvocationsPerWorkgroup;
        public uint MaxComputeWorkgroupSizeX;
        public uint MaxComputeWorkgroupSizeY;
        public uint MaxComputeWorkgroupSizeZ;
        public uint MaxComputeWorkgroupsPerDimension;
        public uint MaxImmediateSize;
    }

    public unsafe struct QueueDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public unsafe delegate void PfnRequestAdapterCallback(RequestAdapterStatus status, Adapter* adapter, StringView message, void* userdata1, void* userdata2);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public unsafe delegate void PfnRequestDeviceCallback(RequestDeviceStatus status, Device* device, StringView message, void* userdata1, void* userdata2);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public unsafe delegate void PfnErrorCallback(Device** device, ErrorType type, StringView message, void* userdata1, void* userdata2);

    public unsafe struct RequestAdapterCallbackInfo
    {
        public ChainedStruct* NextInChain;
        public CallbackMode Mode;
        public nint Callback;
        public void* Userdata1;
        public void* Userdata2;
    }

    public unsafe struct RequestDeviceCallbackInfo
    {
        public ChainedStruct* NextInChain;
        public CallbackMode Mode;
        public nint Callback;
        public void* Userdata1;
        public void* Userdata2;
    }

    public unsafe struct DeviceLostCallbackInfo
    {
        public ChainedStruct* NextInChain;
        public CallbackMode Mode;
        public nint Callback;
        public void* Userdata1;
        public void* Userdata2;
    }

    public unsafe struct UncapturedErrorCallbackInfo
    {
        public ChainedStruct* NextInChain;
        public nint Callback;
        public void* Userdata1;
        public void* Userdata2;
    }

    public unsafe struct DeviceDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public nuint RequiredFeatureCount;
        public uint* RequiredFeatures;
        public Limits* RequiredLimits;
        public QueueDescriptor DefaultQueue;
        public DeviceLostCallbackInfo DeviceLostCallbackInfo;
        public UncapturedErrorCallbackInfo UncapturedErrorCallbackInfo;
    }

    public unsafe struct SurfaceCapabilities
    {
        public ChainedStruct* NextInChain;
        public TextureUsage Usages;
        public nuint FormatCount;
        public TextureFormat* Formats;
        public nuint PresentModeCount;
        public PresentMode* PresentModes;
        public nuint AlphaModeCount;
        public CompositeAlphaMode* AlphaModes;
    }

    public unsafe struct SurfaceConfiguration
    {
        public ChainedStruct* NextInChain;
        public Device* Device;
        public TextureFormat Format;
        public TextureUsage Usage;
        public uint Width;
        public uint Height;
        public nuint ViewFormatCount;
        public TextureFormat* ViewFormats;
        public CompositeAlphaMode AlphaMode;
        public PresentMode PresentMode;
    }

    public unsafe struct SurfaceTexture
    {
        public ChainedStruct* NextInChain;
        public Texture* Texture;
        public SurfaceGetCurrentTextureStatus Status;
    }

    public unsafe struct BufferDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public BufferUsage Usage;
        public ulong Size;
        [MarshalAs(UnmanagedType.I4)]
        public bool MappedAtCreation;
    }

    public struct Extent3D
    {
        public uint Width;
        public uint Height;
        public uint DepthOrArrayLayers;
        public Extent3D(uint w, uint h, uint d) { Width = w; Height = h; DepthOrArrayLayers = d; }
    }

    public struct Origin3D
    {
        public uint X;
        public uint Y;
        public uint Z;
        public Origin3D(uint x, uint y, uint z) { X = x; Y = y; Z = z; }
    }

    public unsafe struct TextureDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public TextureUsage Usage;
        public TextureDimension Dimension;
        public Extent3D Size;
        public TextureFormat Format;
        public uint MipLevelCount;
        public uint SampleCount;
        public nuint ViewFormatCount;
        public TextureFormat* ViewFormats;
    }

    public unsafe struct TextureViewDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public TextureFormat Format;
        public TextureViewDimension Dimension;
        public uint BaseMipLevel;
        public uint MipLevelCount;
        public uint BaseArrayLayer;
        public uint ArrayLayerCount;
        public TextureAspect Aspect;
        public TextureUsage Usage;
    }

    public unsafe struct SamplerDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public AddressMode AddressModeU;
        public AddressMode AddressModeV;
        public AddressMode AddressModeW;
        public FilterMode MagFilter;
        public FilterMode MinFilter;
        public MipmapFilterMode MipmapFilter;
        public float LodMinClamp;
        public float LodMaxClamp;
        public CompareFunction Compare;
        public ushort MaxAnisotropy;
    }

    public unsafe struct ImageCopyTexture
    {
        public Texture* Texture;
        public uint MipLevel;
        public Origin3D Origin;
        public TextureAspect Aspect;
    }

    public struct TextureDataLayout
    {
        public ulong Offset;
        public uint BytesPerRow;
        public uint RowsPerImage;
    }

    public unsafe struct ShaderModuleWGSLDescriptor
    {
        public ChainedStruct Chain;
        public StringView Code;
    }

    public unsafe struct ShaderModuleDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
    }

    public unsafe struct BufferBindingLayout
    {
        public ChainedStruct* NextInChain;
        public BufferBindingType Type;
        public int HasDynamicOffset;
        public ulong MinBindingSize;
    }

    public unsafe struct SamplerBindingLayout
    {
        public ChainedStruct* NextInChain;
        public SamplerBindingType Type;
    }

    public unsafe struct TextureBindingLayout
    {
        public ChainedStruct* NextInChain;
        public TextureSampleType SampleType;
        public TextureViewDimension ViewDimension;
        public int Multisampled;
    }

    public unsafe struct StorageTextureBindingLayout
    {
        public ChainedStruct* NextInChain;
        public StorageTextureAccess Access;
        public TextureFormat Format;
        public TextureViewDimension ViewDimension;
    }

    public unsafe struct BindGroupLayoutEntry
    {
        public ChainedStruct* NextInChain;
        public uint Binding;
        public ShaderStage Visibility;
        public uint BindingArraySize;
        public BufferBindingLayout Buffer;
        public SamplerBindingLayout Sampler;
        public TextureBindingLayout Texture;
        public StorageTextureBindingLayout StorageTexture;
    }

    public unsafe struct BindGroupLayoutDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public nuint EntryCount;
        public BindGroupLayoutEntry* Entries;
    }

    public unsafe struct BindGroupEntry
    {
        public ChainedStruct* NextInChain;
        public uint Binding;
        public Buffer* Buffer;
        public ulong Offset;
        public ulong Size;
        public Sampler* Sampler;
        public TextureView* TextureView;
    }

    public unsafe struct BindGroupDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public BindGroupLayout* Layout;
        public nuint EntryCount;
        public BindGroupEntry* Entries;
    }

    public unsafe struct PipelineLayoutDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public nuint BindGroupLayoutCount;
        public BindGroupLayout** BindGroupLayouts;
        public uint ImmediateSize;
    }

    public unsafe struct ProgrammableStageDescriptor
    {
        public ChainedStruct* NextInChain;
        public ShaderModule* Module;
        public StringView EntryPoint;
        public nuint ConstantCount;
        public void* Constants;
    }

    public unsafe struct ComputePipelineDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public PipelineLayout* Layout;
        public ProgrammableStageDescriptor Compute;
    }

    public unsafe struct VertexAttribute
    {
        public ChainedStruct* NextInChain;
        public VertexFormat Format;
        public ulong Offset;
        public uint ShaderLocation;
    }

    public unsafe struct VertexBufferLayout
    {
        public ChainedStruct* NextInChain;
        public VertexStepMode StepMode;
        public ulong ArrayStride;
        public nuint AttributeCount;
        public VertexAttribute* Attributes;
    }

    public unsafe struct VertexState
    {
        public ChainedStruct* NextInChain;
        public ShaderModule* Module;
        public StringView EntryPoint;
        public nuint ConstantCount;
        public void* Constants;
        public nuint BufferCount;
        public VertexBufferLayout* Buffers;
    }

    public unsafe struct PrimitiveState
    {
        public ChainedStruct* NextInChain;
        public PrimitiveTopology Topology;
        public IndexFormat StripIndexFormat;
        public FrontFace FrontFace;
        public CullMode CullMode;
        public int UnclippedDepth;
    }

    public struct StencilFaceState
    {
        public CompareFunction Compare;
        public StencilOperation FailOp;
        public StencilOperation DepthFailOp;
        public StencilOperation PassOp;
    }

    public unsafe struct DepthStencilState
    {
        public ChainedStruct* NextInChain;
        public TextureFormat Format;
        [MarshalAs(UnmanagedType.I4)]
        public bool DepthWriteEnabled;
        public CompareFunction DepthCompare;
        public StencilFaceState StencilFront;
        public StencilFaceState StencilBack;
        public uint StencilReadMask;
        public uint StencilWriteMask;
        public int DepthBias;
        public float DepthBiasSlopeScale;
        public float DepthBiasClamp;
    }

    public unsafe struct MultisampleState
    {
        public ChainedStruct* NextInChain;
        public uint Count;
        public uint Mask;
        public int AlphaToCoverageEnabled;
    }

    public struct BlendComponent
    {
        public BlendOperation Operation;
        public BlendFactor SrcFactor;
        public BlendFactor DstFactor;
    }

    public struct BlendState
    {
        public BlendComponent Color;
        public BlendComponent Alpha;
    }

    public unsafe struct ColorTargetState
    {
        public ChainedStruct* NextInChain;
        public TextureFormat Format;
        public BlendState* Blend;
        public ColorWriteMask WriteMask;
    }

    public unsafe struct FragmentState
    {
        public ChainedStruct* NextInChain;
        public ShaderModule* Module;
        public StringView EntryPoint;
        public nuint ConstantCount;
        public void* Constants;
        public nuint TargetCount;
        public ColorTargetState* Targets;
    }

    public unsafe struct RenderPipelineDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public PipelineLayout* Layout;
        public VertexState Vertex;
        public PrimitiveState Primitive;
        public DepthStencilState* DepthStencil;
        public MultisampleState Multisample;
        public FragmentState* Fragment;
    }

    public unsafe struct CommandEncoderDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
    }

    public unsafe struct ComputePassDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public void* TimestampWrites;
    }

    public struct Color
    {
        public double R;
        public double G;
        public double B;
        public double A;
        public Color(double r, double g, double b, double a) { R = r; G = g; B = b; A = a; }
    }

    public unsafe struct RenderPassColorAttachment
    {
        public ChainedStruct* NextInChain;
        public TextureView* View;
        public uint DepthSlice;
        public TextureView* ResolveTarget;
        public LoadOp LoadOp;
        public StoreOp StoreOp;
        public Color ClearValue;
    }

    public unsafe struct RenderPassDepthStencilAttachment
    {
        public ChainedStruct* NextInChain;
        public TextureView* View;
        public LoadOp DepthLoadOp;
        public StoreOp DepthStoreOp;
        public float DepthClearValue;
        public int DepthReadOnly;
        public LoadOp StencilLoadOp;
        public StoreOp StencilStoreOp;
        public uint StencilClearValue;
        public int StencilReadOnly;
    }

    public unsafe struct RenderPassDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
        public nuint ColorAttachmentCount;
        public RenderPassColorAttachment* ColorAttachments;
        public RenderPassDepthStencilAttachment* DepthStencilAttachment;
        public void* OcclusionQuerySet;
        public void* TimestampWrites;
    }

    public unsafe struct CommandBufferDescriptor
    {
        public ChainedStruct* NextInChain;
        public StringView Label;
    }

    public sealed unsafe class WebGPU : IDisposable
    {
        private const string Lib = "wgpu_native";
        [DllImport(Lib)] private static extern uint wgpuDevicePoll(Device* device, uint wait, ulong* submissionIndex);
        public void DeviceWaitIdle(Device* device) => wgpuDevicePoll(device, 1, null);

        static WebGPU()
        {
            NativeLibrary.SetDllImportResolver(typeof(WebGPU).Assembly, (libraryName, assembly, searchPath) =>
            {
                if (libraryName == "wgpu_native" || libraryName == "libwgpu_native.so")
                {
                    string candidate1 = System.IO.Path.Combine(AppContext.BaseDirectory, "libwgpu_native.so");
                    if (System.IO.File.Exists(candidate1) && NativeLibrary.TryLoad(candidate1, out nint handle1))
                        return handle1;

                    string candidate2 = System.IO.Path.Combine(AppContext.BaseDirectory, "runtimes", "linux-x64", "native", "libwgpu_native.so");
                    if (System.IO.File.Exists(candidate2) && NativeLibrary.TryLoad(candidate2, out nint handle2))
                        return handle2;
                }
                return nint.Zero;
            });
        }

        public static WebGPU GetApi() => new();

        [DllImport(Lib)] private static extern Instance* wgpuCreateInstance(InstanceDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuInstanceRelease(Instance* instance);
        [DllImport(Lib)] private static extern Surface* wgpuInstanceCreateSurface(Instance* instance, SurfaceDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuSurfaceRelease(Surface* surface);
        [DllImport(Lib)] private static extern Future wgpuInstanceRequestAdapter(Instance* instance, RequestAdapterOptions* options, RequestAdapterCallbackInfo callbackInfo);
        [DllImport(Lib)] private static extern int wgpuAdapterGetLimits(Adapter* adapter, Limits* limits);
        [DllImport(Lib)] private static extern Future wgpuAdapterRequestDevice(Adapter* adapter, DeviceDescriptor* desc, RequestDeviceCallbackInfo callbackInfo);
        [DllImport(Lib)] private static extern void wgpuAdapterRelease(Adapter* adapter);
        [DllImport(Lib)] private static extern Queue* wgpuDeviceGetQueue(Device* device);
        [DllImport(Lib)] private static extern void wgpuDeviceRelease(Device* device);
        [DllImport(Lib)] private static extern void wgpuQueueRelease(Queue* queue);
        [DllImport(Lib)] private static extern int wgpuSurfaceGetCapabilities(Surface* surface, Adapter* adapter, SurfaceCapabilities* caps);
        [DllImport(Lib)] private static extern void wgpuSurfaceConfigure(Surface* surface, SurfaceConfiguration* config);
        [DllImport(Lib)] private static extern void wgpuSurfaceGetCurrentTexture(Surface* surface, SurfaceTexture* tex);
        [DllImport(Lib)] private static extern int wgpuSurfacePresent(Surface* surface);
        [DllImport(Lib)] private static extern Buffer* wgpuDeviceCreateBuffer(Device* device, BufferDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuBufferDestroy(Buffer* buffer);
        [DllImport(Lib)] private static extern void wgpuBufferRelease(Buffer* buffer);
        [DllImport(Lib)] private static extern void wgpuQueueWriteBuffer(Queue* queue, Buffer* buffer, ulong offset, void* data, nuint size);
        [DllImport(Lib)] private static extern Texture* wgpuDeviceCreateTexture(Device* device, TextureDescriptor* desc);
        [DllImport(Lib)] private static extern TextureView* wgpuTextureCreateView(Texture* texture, TextureViewDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuTextureDestroy(Texture* texture);
        [DllImport(Lib)] private static extern void wgpuTextureRelease(Texture* texture);
        [DllImport(Lib)] private static extern void wgpuTextureViewRelease(TextureView* view);
        [DllImport(Lib)] private static extern Sampler* wgpuDeviceCreateSampler(Device* device, SamplerDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuSamplerRelease(Sampler* sampler);
        [DllImport(Lib)] private static extern void wgpuQueueWriteTexture(Queue* queue, ImageCopyTexture* dst, void* data, nuint dataSize, TextureDataLayout* layout, Extent3D* size);
        [DllImport(Lib)] private static extern ShaderModule* wgpuDeviceCreateShaderModule(Device* device, ShaderModuleDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuShaderModuleRelease(ShaderModule* mod);
        [DllImport(Lib)] private static extern BindGroupLayout* wgpuDeviceCreateBindGroupLayout(Device* device, BindGroupLayoutDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuBindGroupLayoutRelease(BindGroupLayout* bgl);
        [DllImport(Lib)] private static extern BindGroup* wgpuDeviceCreateBindGroup(Device* device, BindGroupDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuBindGroupRelease(BindGroup* bg);
        [DllImport(Lib)] private static extern PipelineLayout* wgpuDeviceCreatePipelineLayout(Device* device, PipelineLayoutDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuPipelineLayoutRelease(PipelineLayout* pl);
        [DllImport(Lib)] private static extern ComputePipeline* wgpuDeviceCreateComputePipeline(Device* device, ComputePipelineDescriptor* desc);
        [DllImport(Lib)] private static extern BindGroupLayout* wgpuComputePipelineGetBindGroupLayout(ComputePipeline* pipeline, uint idx);
        [DllImport(Lib)] private static extern void wgpuComputePipelineRelease(ComputePipeline* pipeline);
        [DllImport(Lib)] private static extern RenderPipeline* wgpuDeviceCreateRenderPipeline(Device* device, RenderPipelineDescriptor* desc);
        [DllImport(Lib)] private static extern BindGroupLayout* wgpuRenderPipelineGetBindGroupLayout(RenderPipeline* pipeline, uint idx);
        [DllImport(Lib)] private static extern void wgpuRenderPipelineRelease(RenderPipeline* pipeline);
        [DllImport(Lib)] private static extern CommandEncoder* wgpuDeviceCreateCommandEncoder(Device* device, CommandEncoderDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuCommandEncoderCopyBufferToBuffer(CommandEncoder* enc, Buffer* src, ulong srcOff, Buffer* dst, ulong dstOff, ulong size);
        [DllImport(Lib)] private static extern ComputePassEncoder* wgpuCommandEncoderBeginComputePass(CommandEncoder* enc, ComputePassDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuComputePassEncoderSetPipeline(ComputePassEncoder* pass, ComputePipeline* pipeline);
        [DllImport(Lib)] private static extern void wgpuComputePassEncoderSetBindGroup(ComputePassEncoder* pass, uint idx, BindGroup* bg, nuint dynCount, uint* dynOff);
        [DllImport(Lib)] private static extern void wgpuComputePassEncoderDispatchWorkgroups(ComputePassEncoder* pass, uint x, uint y, uint z);
        [DllImport(Lib)] private static extern void wgpuComputePassEncoderDispatchWorkgroupsIndirect(ComputePassEncoder* pass, Buffer* indirectBuf, ulong offset);
        [DllImport(Lib)] private static extern void wgpuComputePassEncoderEnd(ComputePassEncoder* pass);
        [DllImport(Lib)] private static extern void wgpuComputePassEncoderRelease(ComputePassEncoder* pass);
        [DllImport(Lib)] private static extern RenderPassEncoder* wgpuCommandEncoderBeginRenderPass(CommandEncoder* enc, RenderPassDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuRenderPassEncoderSetPipeline(RenderPassEncoder* pass, RenderPipeline* pipeline);
        [DllImport(Lib)] private static extern void wgpuRenderPassEncoderSetBindGroup(RenderPassEncoder* pass, uint idx, BindGroup* bg, nuint dynCount, uint* dynOff);
        [DllImport(Lib)] private static extern void wgpuRenderPassEncoderSetVertexBuffer(RenderPassEncoder* pass, uint slot, Buffer* buf, ulong offset, ulong size);
        [DllImport(Lib)] private static extern void wgpuRenderPassEncoderSetIndexBuffer(RenderPassEncoder* pass, Buffer* buf, IndexFormat fmt, ulong offset, ulong size);
        [DllImport(Lib)] private static extern void wgpuRenderPassEncoderSetViewport(RenderPassEncoder* pass, float x, float y, float w, float h, float minD, float maxD);
        [DllImport(Lib)] private static extern void wgpuRenderPassEncoderSetScissorRect(RenderPassEncoder* pass, uint x, uint y, uint w, uint h);
        [DllImport(Lib)] private static extern void wgpuRenderPassEncoderDraw(RenderPassEncoder* pass, uint vtxCount, uint instCount, uint firstVtx, uint firstInst);
        [DllImport(Lib)] private static extern void wgpuRenderPassEncoderDrawIndexed(RenderPassEncoder* pass, uint idxCount, uint instCount, uint firstIdx, int baseVtx, uint firstInst);
        [DllImport(Lib)] private static extern void wgpuRenderPassEncoderEnd(RenderPassEncoder* pass);
        [DllImport(Lib)] private static extern void wgpuRenderPassEncoderRelease(RenderPassEncoder* pass);
        [DllImport(Lib)] private static extern CommandBuffer* wgpuCommandEncoderFinish(CommandEncoder* enc, CommandBufferDescriptor* desc);
        [DllImport(Lib)] private static extern void wgpuCommandEncoderRelease(CommandEncoder* enc);
        [DllImport(Lib)] private static extern void wgpuCommandBufferRelease(CommandBuffer* cmd);
        [DllImport(Lib)] private static extern void wgpuQueueSubmit(Queue* queue, nuint count, CommandBuffer** cmds);

        public Instance* CreateInstance(InstanceDescriptor* desc) => wgpuCreateInstance(desc);
        public void InstanceRelease(Instance* i) => wgpuInstanceRelease(i);
        public Surface* InstanceCreateSurface(Instance* i, SurfaceDescriptor* desc) => wgpuInstanceCreateSurface(i, desc);
        public void SurfaceRelease(Surface* s) => wgpuSurfaceRelease(s);
        public Future InstanceRequestAdapter(Instance* i, RequestAdapterOptions* opts, RequestAdapterCallbackInfo cbInfo) => wgpuInstanceRequestAdapter(i, opts, cbInfo);
        public int AdapterGetLimits(Adapter* a, Limits* l) => wgpuAdapterGetLimits(a, l);
        public Future AdapterRequestDevice(Adapter* a, DeviceDescriptor* d, RequestDeviceCallbackInfo cbInfo) => wgpuAdapterRequestDevice(a, d, cbInfo);
        public void AdapterRelease(Adapter* a) => wgpuAdapterRelease(a);
        public Queue* DeviceGetQueue(Device* d) => wgpuDeviceGetQueue(d);
        public void DeviceRelease(Device* d) => wgpuDeviceRelease(d);
        public void QueueRelease(Queue* q) => wgpuQueueRelease(q);
        public int SurfaceGetCapabilities(Surface* s, Adapter* a, SurfaceCapabilities* c) => wgpuSurfaceGetCapabilities(s, a, c);
        public void SurfaceConfigure(Surface* s, SurfaceConfiguration* c) => wgpuSurfaceConfigure(s, c);
        public void SurfaceGetCurrentTexture(Surface* s, SurfaceTexture* t) => wgpuSurfaceGetCurrentTexture(s, t);
        public int SurfacePresent(Surface* s) => wgpuSurfacePresent(s);

        public Buffer* DeviceCreateBuffer(Device* d, BufferDescriptor* desc) => wgpuDeviceCreateBuffer(d, desc);
        public Buffer* DeviceCreateBuffer(Device* d, in BufferDescriptor desc) { fixed (BufferDescriptor* p = &desc) return wgpuDeviceCreateBuffer(d, p); }
        public void BufferDestroy(Buffer* b) => wgpuBufferDestroy(b);
        public void BufferRelease(Buffer* b) => wgpuBufferRelease(b);
        public void QueueWriteBuffer(Queue* q, Buffer* b, ulong off, void* data, nuint sz) => wgpuQueueWriteBuffer(q, b, off, data, sz);

        public Texture* DeviceCreateTexture(Device* d, TextureDescriptor* desc) => wgpuDeviceCreateTexture(d, desc);
        public Texture* DeviceCreateTexture(Device* d, in TextureDescriptor desc) { fixed (TextureDescriptor* p = &desc) return wgpuDeviceCreateTexture(d, p); }
        public TextureView* TextureCreateView(Texture* t, TextureViewDescriptor* desc) => wgpuTextureCreateView(t, desc);
        public TextureView* TextureCreateView(Texture* t, in TextureViewDescriptor desc) { fixed (TextureViewDescriptor* p = &desc) return wgpuTextureCreateView(t, p); }
        public void TextureDestroy(Texture* t) => wgpuTextureDestroy(t);
        public void TextureRelease(Texture* t) => wgpuTextureRelease(t);
        public void TextureViewRelease(TextureView* v) => wgpuTextureViewRelease(v);

        public Sampler* DeviceCreateSampler(Device* d, SamplerDescriptor* desc)
        {
            if (desc != null)
            {
                if (desc->LodMaxClamp == 0f) desc->LodMaxClamp = 32f;
                if (desc->MaxAnisotropy == 0) desc->MaxAnisotropy = 1;
            }
            return wgpuDeviceCreateSampler(d, desc);
        }

        public Sampler* DeviceCreateSampler(Device* d, in SamplerDescriptor desc)
        {
            SamplerDescriptor copy = desc;
            return DeviceCreateSampler(d, &copy);
        }

        public void SamplerRelease(Sampler* s) => wgpuSamplerRelease(s);
        public void QueueWriteTexture(Queue* q, ImageCopyTexture* dst, void* data, nuint sz, TextureDataLayout* l, Extent3D* e) => wgpuQueueWriteTexture(q, dst, data, sz, l, e);
        public void QueueWriteTexture(Queue* q, in ImageCopyTexture dst, void* data, nuint sz, in TextureDataLayout l, in Extent3D e)
        {
            fixed (ImageCopyTexture* pd = &dst)
            fixed (TextureDataLayout* pl = &l)
            fixed (Extent3D* pe = &e)
                wgpuQueueWriteTexture(q, pd, data, sz, pl, pe);
        }

        public ShaderModule* DeviceCreateShaderModule(Device* d, ShaderModuleDescriptor* desc) => wgpuDeviceCreateShaderModule(d, desc);
        public ShaderModule* DeviceCreateShaderModule(Device* d, in ShaderModuleDescriptor desc) { fixed (ShaderModuleDescriptor* p = &desc) return wgpuDeviceCreateShaderModule(d, p); }
        public void ShaderModuleRelease(ShaderModule* m) => wgpuShaderModuleRelease(m);
        public BindGroupLayout* DeviceCreateBindGroupLayout(Device* d, BindGroupLayoutDescriptor* desc) => wgpuDeviceCreateBindGroupLayout(d, desc);
        public BindGroupLayout* DeviceCreateBindGroupLayout(Device* d, in BindGroupLayoutDescriptor desc) { fixed (BindGroupLayoutDescriptor* p = &desc) return wgpuDeviceCreateBindGroupLayout(d, p); }
        public void BindGroupLayoutRelease(BindGroupLayout* b) => wgpuBindGroupLayoutRelease(b);
        public BindGroup* DeviceCreateBindGroup(Device* d, BindGroupDescriptor* desc) => wgpuDeviceCreateBindGroup(d, desc);
        public BindGroup* DeviceCreateBindGroup(Device* d, in BindGroupDescriptor desc) { fixed (BindGroupDescriptor* p = &desc) return wgpuDeviceCreateBindGroup(d, p); }
        public void BindGroupRelease(BindGroup* b) => wgpuBindGroupRelease(b);
        public PipelineLayout* DeviceCreatePipelineLayout(Device* d, PipelineLayoutDescriptor* desc) => wgpuDeviceCreatePipelineLayout(d, desc);
        public PipelineLayout* DeviceCreatePipelineLayout(Device* d, in PipelineLayoutDescriptor desc) { fixed (PipelineLayoutDescriptor* p = &desc) return wgpuDeviceCreatePipelineLayout(d, p); }
        public void PipelineLayoutRelease(PipelineLayout* p) => wgpuPipelineLayoutRelease(p);
        public ComputePipeline* DeviceCreateComputePipeline(Device* d, ComputePipelineDescriptor* desc) => wgpuDeviceCreateComputePipeline(d, desc);
        public ComputePipeline* DeviceCreateComputePipeline(Device* d, in ComputePipelineDescriptor desc) { fixed (ComputePipelineDescriptor* p = &desc) return wgpuDeviceCreateComputePipeline(d, p); }
        public BindGroupLayout* ComputePipelineGetBindGroupLayout(ComputePipeline* p, uint idx) => wgpuComputePipelineGetBindGroupLayout(p, idx);
        public void ComputePipelineRelease(ComputePipeline* p) => wgpuComputePipelineRelease(p);
        public RenderPipeline* DeviceCreateRenderPipeline(Device* d, RenderPipelineDescriptor* desc) => wgpuDeviceCreateRenderPipeline(d, desc);
        public RenderPipeline* DeviceCreateRenderPipeline(Device* d, in RenderPipelineDescriptor desc) { fixed (RenderPipelineDescriptor* p = &desc) return wgpuDeviceCreateRenderPipeline(d, p); }
        public BindGroupLayout* RenderPipelineGetBindGroupLayout(RenderPipeline* p, uint idx) => wgpuRenderPipelineGetBindGroupLayout(p, idx);
        public void RenderPipelineRelease(RenderPipeline* p) => wgpuRenderPipelineRelease(p);

        public CommandEncoder* DeviceCreateCommandEncoder(Device* d, CommandEncoderDescriptor* desc) => wgpuDeviceCreateCommandEncoder(d, desc);
        public CommandEncoder* DeviceCreateCommandEncoder(Device* d, in CommandEncoderDescriptor desc) { fixed (CommandEncoderDescriptor* p = &desc) return wgpuDeviceCreateCommandEncoder(d, p); }
        public void CommandEncoderCopyBufferToBuffer(CommandEncoder* e, Buffer* s, ulong so, Buffer* d, ulong doff, ulong sz) => wgpuCommandEncoderCopyBufferToBuffer(e, s, so, d, doff, sz);
        public ComputePassEncoder* CommandEncoderBeginComputePass(CommandEncoder* e, ComputePassDescriptor* desc) => wgpuCommandEncoderBeginComputePass(e, desc);
        public ComputePassEncoder* CommandEncoderBeginComputePass(CommandEncoder* e, in ComputePassDescriptor desc) { fixed (ComputePassDescriptor* p = &desc) return wgpuCommandEncoderBeginComputePass(e, p); }
        public void ComputePassEncoderSetPipeline(ComputePassEncoder* p, ComputePipeline* pl) => wgpuComputePassEncoderSetPipeline(p, pl);
        public void ComputePassEncoderSetBindGroup(ComputePassEncoder* p, uint idx, BindGroup* bg, nuint dc, uint* doff) => wgpuComputePassEncoderSetBindGroup(p, idx, bg, dc, doff);
        public void ComputePassEncoderDispatchWorkgroups(ComputePassEncoder* p, uint x, uint y, uint z) => wgpuComputePassEncoderDispatchWorkgroups(p, x, y, z);
        public void ComputePassEncoderDispatchWorkgroupsIndirect(ComputePassEncoder* p, Buffer* b, ulong off) => wgpuComputePassEncoderDispatchWorkgroupsIndirect(p, b, off);
        public void ComputePassEncoderEnd(ComputePassEncoder* p) => wgpuComputePassEncoderEnd(p);
        public void ComputePassEncoderRelease(ComputePassEncoder* p) => wgpuComputePassEncoderRelease(p);

        public RenderPassEncoder* CommandEncoderBeginRenderPass(CommandEncoder* e, RenderPassDescriptor* desc)
        {
            if (desc != null && desc->ColorAttachments != null)
            {
                for (nuint i = 0; i < desc->ColorAttachmentCount; i++)
                {
                    if (desc->ColorAttachments[i].DepthSlice == 0)
                        desc->ColorAttachments[i].DepthSlice = 0xFFFFFFFFu;
                }
            }
            return wgpuCommandEncoderBeginRenderPass(e, desc);
        }

        public RenderPassEncoder* CommandEncoderBeginRenderPass(CommandEncoder* e, in RenderPassDescriptor desc)
        {
            RenderPassDescriptor copy = desc;
            return CommandEncoderBeginRenderPass(e, &copy);
        }

        public void RenderPassEncoderSetPipeline(RenderPassEncoder* p, RenderPipeline* pl) => wgpuRenderPassEncoderSetPipeline(p, pl);
        public void RenderPassEncoderSetBindGroup(RenderPassEncoder* p, uint idx, BindGroup* bg, nuint dc, uint* doff) => wgpuRenderPassEncoderSetBindGroup(p, idx, bg, dc, doff);
        public void RenderPassEncoderSetVertexBuffer(RenderPassEncoder* p, uint slot, Buffer* b, ulong off, ulong sz) => wgpuRenderPassEncoderSetVertexBuffer(p, slot, b, off, sz);
        public void RenderPassEncoderSetIndexBuffer(RenderPassEncoder* p, Buffer* b, IndexFormat fmt, ulong off, ulong sz) => wgpuRenderPassEncoderSetIndexBuffer(p, b, fmt, off, sz);
        public void RenderPassEncoderSetViewport(RenderPassEncoder* p, float x, float y, float w, float h, float minD, float maxD) => wgpuRenderPassEncoderSetViewport(p, x, y, w, h, minD, maxD);
        public void RenderPassEncoderSetScissorRect(RenderPassEncoder* p, uint x, uint y, uint w, uint h) => wgpuRenderPassEncoderSetScissorRect(p, x, y, w, h);
        public void RenderPassEncoderDraw(RenderPassEncoder* p, uint vc, uint ic, uint fv, uint fi) => wgpuRenderPassEncoderDraw(p, vc, ic, fv, fi);
        public void RenderPassEncoderDrawIndexed(RenderPassEncoder* p, uint ic, uint instC, uint fi, int bv, uint finst) => wgpuRenderPassEncoderDrawIndexed(p, ic, instC, fi, bv, finst);
        public void RenderPassEncoderEnd(RenderPassEncoder* p) => wgpuRenderPassEncoderEnd(p);
        public void RenderPassEncoderRelease(RenderPassEncoder* p) => wgpuRenderPassEncoderRelease(p);

        public CommandBuffer* CommandEncoderFinish(CommandEncoder* e, CommandBufferDescriptor* desc) => wgpuCommandEncoderFinish(e, desc);
        public CommandBuffer* CommandEncoderFinish(CommandEncoder* e, in CommandBufferDescriptor desc) { fixed (CommandBufferDescriptor* p = &desc) return wgpuCommandEncoderFinish(e, p); }
        public void CommandEncoderRelease(CommandEncoder* e) => wgpuCommandEncoderRelease(e);
        public void CommandBufferRelease(CommandBuffer* c) => wgpuCommandBufferRelease(c);
        public void QueueSubmit(Queue* q, nuint c, CommandBuffer** cmds) => wgpuQueueSubmit(q, c, cmds);

        public void Dispose() { }
    }
}
