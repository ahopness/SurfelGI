using System.Runtime.InteropServices;
using System.Numerics;
using Silk.NET.WebGPU;
using SurfelGI.Core;

namespace SurfelGI.GiPipeline;

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct AllocateUniforms
{
    public uint ReadOffset;
    public uint TileWidth;
    public uint Frame;
    public uint TileCount;
    public Vector4 CameraPosition;
}

public unsafe class SurfelAllocatePass : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    private readonly GpuBuffer<AllocateUniforms> _uniformBuffer;
    private readonly ComputePipeline* _mainPipeline;
    private readonly ComputePipeline* _syncPipeline;

    public SurfelAllocatePass(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx;
        _shaders = shaders;
        _uniformBuffer = new GpuBuffer<AllocateUniforms>(ctx, 1, BufferUsage.Uniform, "AllocateUniforms");
        _mainPipeline = _shaders.GetOrCreateComputePipeline("surfel_allocate.wgsl", "main");
        _syncPipeline = _shaders.GetOrCreateComputePipeline("surfel_allocate.wgsl", "sync_alive");
    }

    public void Run(
        CommandEncoder* encoder,
        SurfelPool pool,
        SurfelFindMissingPass find,
        int tileCount)
    {
        if (tileCount <= 0) return;

        var (readOffset, _) = pool.GetOffsets();
        var uData = new AllocateUniforms
        {
            ReadOffset = readOffset,
            TileWidth = (uint)find.TileWidth,
            CameraPosition = new Vector4(find.CameraPosition, 1f),
            Frame = _ctx.FrameIndex,
            TileCount = (uint)tileCount
        };
        _uniformBuffer.UploadSingle(in uData);

        Span<BindGroupEntry> mainEntries = stackalloc BindGroupEntry[12]
        {
            _uniformBuffer.AsBindGroupEntry(0),
            pool.PoolAllocAtomicBuffer.AsBindGroupEntry(1),
            pool.PoolMaxAtomicBuffer.AsBindGroupEntry(2),
            pool.PoolBuffer.AsBindGroupEntry(4),
            pool.SurfelBuffer.AsBindGroupEntry(5),
            pool.MomentsBuffer.AsBindGroupEntry(6),
            pool.GuidingBuffer.AsBindGroupEntry(7),
            pool.SurfelDepthBuffer.AsBindGroupEntry(8),
            find.TileAllocBuffer.AsBindGroupEntry(9),
            find.CandidatePackedBuffer.AsBindGroupEntry(10),
            find.TileIrradianceBuffer.AsBindGroupEntry(11),
            pool.TouchedAtomicBuffer.AsBindGroupEntry(12)
        };

        Span<BindGroupEntry> syncEntries = stackalloc BindGroupEntry[2]
        {
            pool.PoolAllocAtomicBuffer.AsBindGroupEntry(1),
            pool.AliveAtomicBuffer.AsBindGroupEntry(3)
        };

        BindGroup* bgMain = _shaders.CreateBindGroup(_mainPipeline, 0, mainEntries, "AllocateMainBG");
        BindGroup* bgSync = _shaders.CreateBindGroup(_syncPipeline, 0, syncEntries, "AllocateSyncBG");

        var passDesc = new ComputePassDescriptor();
        ComputePassEncoder* pass = _ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &passDesc);

        _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, _mainPipeline);
        _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bgMain, 0, null);
        uint groups = (uint)((tileCount + 63) / 64);
        _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, groups, 1, 1);

        _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, _syncPipeline);
        _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bgSync, 0, null);
        _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);

        _ctx.Wgpu.ComputePassEncoderEnd(pass);
        _ctx.Wgpu.ComputePassEncoderRelease(pass);

        _ctx.Wgpu.BindGroupRelease(bgMain);
        _ctx.Wgpu.BindGroupRelease(bgSync);
    }

    public void Dispose()
    {
        _uniformBuffer.Dispose();
    }
}
