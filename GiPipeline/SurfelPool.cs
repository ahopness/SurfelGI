using System.Numerics;
using Silk.NET.WebGPU;
using SurfelGI.Core;

namespace SurfelGI.GiPipeline;

public class SurfelPool : IDisposable
{
    private readonly WebGpuContext _ctx;
    public int Capacity { get; private set; }
    private int _frameParity;

    public GpuBuffer<SurfelPacked> SurfelBuffer { get; private set; } = null!;
    public GpuBuffer<int> AliveAtomicBuffer { get; private set; } = null!;
    public GpuBuffer<int> PoolBuffer { get; private set; } = null!;
    public GpuBuffer<int> PoolAllocAtomicBuffer { get; private set; } = null!;
    public GpuBuffer<int> PoolMaxAtomicBuffer { get; private set; } = null!;
    public GpuBuffer<SurfelMoments> MomentsBuffer { get; private set; } = null!;
    public GpuBuffer<int> TouchedAtomicBuffer { get; private set; } = null!;
    public GpuBuffer<float> GuidingBuffer { get; private set; } = null!;
    public GpuBuffer<Vector4> SurfelDepthBuffer { get; private set; } = null!;

    public SurfelPool(WebGpuContext ctx, int capacity = Constants.MAX_SURFELS)
    {
        _ctx = ctx;
        EnsureCapacity(capacity);
    }

    public void EnsureCapacity(int cap)
    {
        if (Capacity == cap && SurfelBuffer != null)
            return;

        Dispose();
        Capacity = cap;

        SurfelBuffer = new GpuBuffer<SurfelPacked>(_ctx, cap, BufferUsage.Storage | BufferUsage.CopySrc, "SurfelPacked");
        AliveAtomicBuffer = new GpuBuffer<int>(_ctx, new[] { 0 }, BufferUsage.Storage, "AliveAtomic");

        int[] poolIdx = new int[cap];
        for (int i = 0; i < cap; i++) poolIdx[i] = i;
        PoolBuffer = new GpuBuffer<int>(_ctx, poolIdx, BufferUsage.Storage, "PoolStack");

        PoolAllocAtomicBuffer = new GpuBuffer<int>(_ctx, new[] { 0 }, BufferUsage.Storage, "PoolAllocAtomic");
        PoolMaxAtomicBuffer = new GpuBuffer<int>(_ctx, new[] { 0 }, BufferUsage.Storage, "PoolMaxAtomic");

        MomentsBuffer = new GpuBuffer<SurfelMoments>(_ctx, cap * 2, BufferUsage.Storage, "SurfelMoments");
        TouchedAtomicBuffer = new GpuBuffer<int>(_ctx, cap, BufferUsage.Storage, "TouchedAtomic");
        GuidingBuffer = new GpuBuffer<float>(_ctx, cap * Constants.SLG_TOTAL_FLOATS, BufferUsage.Storage, "GuidingBuffer");

        int tileTexels = Constants.SURFEL_DEPTH_TEXELS * Constants.SURFEL_DEPTH_TEXELS;
        SurfelDepthBuffer = new GpuBuffer<Vector4>(_ctx, cap * tileTexels, BufferUsage.Storage, "SurfelDepth");
    }

    public void SwapMoments()
    {
        _frameParity = 1 - _frameParity;
    }

    public (uint ReadOffset, uint WriteOffset) GetOffsets()
    {
        return (
            (uint)(_frameParity * Capacity),
            (uint)((1 - _frameParity) * Capacity)
        );
    }

    public void Dispose()
    {
        SurfelBuffer?.Dispose();
        AliveAtomicBuffer?.Dispose();
        PoolBuffer?.Dispose();
        PoolAllocAtomicBuffer?.Dispose();
        PoolMaxAtomicBuffer?.Dispose();
        MomentsBuffer?.Dispose();
        TouchedAtomicBuffer?.Dispose();
        GuidingBuffer?.Dispose();
        SurfelDepthBuffer?.Dispose();
    }
}
