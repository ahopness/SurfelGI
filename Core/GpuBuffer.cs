using System.Runtime.CompilerServices;
using Silk.NET.Core.Native;
using Silk.NET.WebGPU;
using Buffer = Silk.NET.WebGPU.Buffer;

namespace SurfelGI.Core;

public unsafe class GpuBuffer<T> : IDisposable where T : unmanaged
{
    private readonly WebGpuContext _ctx;
    public Buffer* Handle { get; private set; }
    public int Count { get; private set; }
    public ulong SizeInBytes { get; private set; }
    public BufferUsage Usage { get; }
    public string Label { get; }
    public int Version { get; private set; }

    public GpuBuffer(WebGpuContext ctx, int count, BufferUsage usage, string label = "")
    {
        _ctx = ctx;
        Usage = usage | BufferUsage.CopyDst;
        Label = label;
        Allocate(count);
    }

    public GpuBuffer(WebGpuContext ctx, ReadOnlySpan<T> initialData, BufferUsage usage, string label = "")
    {
        _ctx = ctx;
        Usage = usage | BufferUsage.CopyDst;
        Label = label;
        Allocate(Math.Max(1, initialData.Length));
        if (!initialData.IsEmpty)
            Upload(initialData);
    }

    private void Allocate(int count)
    {
        if (Handle != null)
        {
            _ctx.Wgpu.BufferDestroy(Handle);
            _ctx.Wgpu.BufferRelease(Handle);
            Handle = null;
        }

        Count = Math.Max(1, count);
        SizeInBytes = (ulong)Count * (ulong)sizeof(T);
        // Align to 4 bytes minimum (or 16 for uniform buffers)
        if ((Usage & BufferUsage.Uniform) != 0)
            SizeInBytes = (SizeInBytes + 15UL) & ~15UL;
        else
            SizeInBytes = (SizeInBytes + 3UL) & ~3UL;

        byte* labelPtr = (byte*)SilkMarshal.StringToPtr(Label);
        try
        {
            var desc = new BufferDescriptor
            {
                Label = labelPtr,
                Size = SizeInBytes,
                Usage = Usage,
                MappedAtCreation = false
            };
            Handle = _ctx.Wgpu.DeviceCreateBuffer(_ctx.Device, &desc);
            Version++;
        }
        finally
        {
            SilkMarshal.Free((nint)labelPtr);
        }
    }

    public bool EnsureCapacity(int neededCount)
    {
        if (neededCount <= Count && Handle != null)
            return false;

        Allocate(neededCount);
        return true;
    }

    public void Upload(ReadOnlySpan<T> data, ulong dstOffsetElements = 0)
    {
        if (data.IsEmpty) return;
        ulong byteOffset = dstOffsetElements * (ulong)sizeof(T);
        ulong byteSize = (ulong)data.Length * (ulong)sizeof(T);
        fixed (T* ptr = data)
        {
            _ctx.Wgpu.QueueWriteBuffer(_ctx.Queue, Handle, byteOffset, ptr, (nuint)byteSize);
        }
    }

    public void UploadSingle(in T value, ulong dstOffsetElements = 0)
    {
        ulong byteOffset = dstOffsetElements * (ulong)sizeof(T);
        fixed (T* ptr = &value)
        {
            _ctx.Wgpu.QueueWriteBuffer(_ctx.Queue, Handle, byteOffset, ptr, (nuint)sizeof(T));
        }
    }

    public BindGroupEntry AsBindGroupEntry(uint binding)
    {
        return new BindGroupEntry
        {
            Binding = binding,
            Buffer = Handle,
            Offset = 0,
            Size = SizeInBytes
        };
    }

    public void Dispose()
    {
        if (Handle != null)
        {
            _ctx.Wgpu.BufferDestroy(Handle);
            _ctx.Wgpu.BufferRelease(Handle);
            Handle = null;
        }
    }
}
