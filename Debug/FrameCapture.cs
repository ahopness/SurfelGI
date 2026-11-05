using System.Runtime.InteropServices;
using Silk.NET.WebGPU;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SurfelGI.Core;
using Buffer = Silk.NET.WebGPU.Buffer;
namespace SurfelGI.Debug;

// Offscreen validation only: read back a completed 8-bit render target.
public static unsafe class FrameCapture
{
    private const string Lib = "wgpu_native";
    [StructLayout(LayoutKind.Sequential)] private struct CopyBuffer { public TextureDataLayout Layout; public Buffer* Buffer; }
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void MapCallback(uint status, StringView message, void* user1, void* user2);
    [DllImport(Lib)] private static extern void wgpuCommandEncoderCopyTextureToBuffer(CommandEncoder* e, ImageCopyTexture* s, CopyBuffer* d, Extent3D* extent);
    [DllImport(Lib)] private static extern Future wgpuBufferMapAsync(Buffer* buffer, ulong mode, nuint offset, nuint size, RequestDeviceCallbackInfo callback);
    [DllImport(Lib)] private static extern void* wgpuBufferGetConstMappedRange(Buffer* b, nuint offset, nuint size);
    [DllImport(Lib)] private static extern void wgpuBufferUnmap(Buffer* b);
    public static T[] ReadBuffer<T>(WebGpuContext ctx, GpuBuffer<T> source) where T : unmanaged
    {
        using var staging = new GpuBuffer<byte>(ctx, (int)source.SizeInBytes, BufferUsage.CopyDst | BufferUsage.MapRead, "ValidationReadback");
        var desc = new CommandEncoderDescriptor(); var encoder = ctx.Wgpu.DeviceCreateCommandEncoder(ctx.Device, in desc);
        ctx.Wgpu.CommandEncoderCopyBufferToBuffer(encoder, source.Handle, 0, staging.Handle, 0, source.SizeInBytes);
        var cbDesc = new CommandBufferDescriptor(); var command = ctx.Wgpu.CommandEncoderFinish(encoder, in cbDesc);
        if (ctx.LastError != null) throw new InvalidOperationException(ctx.LastError);
        ctx.Wgpu.QueueSubmit(ctx.Queue, 1, &command);
        uint status = 0;
        MapCallback callback = (s, _, _, _) => status = s;
        var info = new RequestDeviceCallbackInfo { Mode = CallbackMode.AllowSpontaneous, Callback = Marshal.GetFunctionPointerForDelegate(callback) };
        wgpuBufferMapAsync(staging.Handle, 1, 0, (nuint)source.SizeInBytes, info);
        ctx.Wgpu.DeviceWaitIdle(ctx.Device); GC.KeepAlive(callback);
        if (status != 1) throw new InvalidOperationException($"Buffer readback failed: {status}");
        var values = new ReadOnlySpan<T>(wgpuBufferGetConstMappedRange(staging.Handle, 0, (nuint)source.SizeInBytes), source.Count).ToArray();
        wgpuBufferUnmap(staging.Handle);
        ctx.Wgpu.CommandBufferRelease(command); ctx.Wgpu.CommandEncoderRelease(encoder);
        return values;
    }

    public static void Save(WebGpuContext ctx, GpuTexture target, string path)
    {
        int stride = (target.Width * 4 + 255) & ~255;
        int bytes = stride * target.Height;
        using var staging = new GpuBuffer<byte>(ctx, bytes, BufferUsage.CopyDst | BufferUsage.MapRead, "FrameReadback");
        var desc = new CommandEncoderDescriptor(); var encoder = ctx.Wgpu.DeviceCreateCommandEncoder(ctx.Device, in desc);
        var src = new ImageCopyTexture { Texture = target.Handle, Aspect = TextureAspect.All };
        var dst = new CopyBuffer { Buffer = staging.Handle, Layout = new TextureDataLayout { BytesPerRow = (uint)stride, RowsPerImage = (uint)target.Height } };
        var extent = new Extent3D { Width = (uint)target.Width, Height = (uint)target.Height, DepthOrArrayLayers = 1 };
        wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);
        var cbDesc = new CommandBufferDescriptor(); var command = ctx.Wgpu.CommandEncoderFinish(encoder, in cbDesc);
        if (ctx.LastError != null) throw new InvalidOperationException(ctx.LastError);
        ctx.Wgpu.QueueSubmit(ctx.Queue, 1, &command);
        uint status = 0;
        MapCallback callback = (s, _, _, _) => status = s;
        var info = new RequestDeviceCallbackInfo { Mode = CallbackMode.AllowSpontaneous, Callback = Marshal.GetFunctionPointerForDelegate(callback) };
        wgpuBufferMapAsync(staging.Handle, 1, 0, (nuint)bytes, info);
        ctx.Wgpu.DeviceWaitIdle(ctx.Device); GC.KeepAlive(callback);
        if (status != 1) throw new InvalidOperationException($"Readback failed: {status}");
        var data = new ReadOnlySpan<byte>(wgpuBufferGetConstMappedRange(staging.Handle, 0, (nuint)bytes), bytes);
        byte[] pixels = new byte[target.Width * target.Height * 4];
        bool bgra = target.Format == TextureFormat.Bgra8UnormSrgb || target.Format == TextureFormat.Bgra8Unorm;
        for (int y=0;y<target.Height;y++) for(int x=0;x<target.Width;x++) {
            int input=y*stride+x*4, output=(y*target.Width+x)*4;
            pixels[output]=data[input+(bgra?2:0)]; pixels[output+1]=data[input+1]; pixels[output+2]=data[input+(bgra?0:2)]; pixels[output+3]=255;
        }
        wgpuBufferUnmap(staging.Handle);
        using var image = Image.LoadPixelData<Rgba32>(pixels, target.Width, target.Height); image.Save(path);
        ctx.Wgpu.CommandBufferRelease(command); ctx.Wgpu.CommandEncoderRelease(encoder);
    }
}
