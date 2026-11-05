using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.WebGPU;
using SurfelGI.Core;

namespace SurfelGI.GiPipeline;

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct AgeUniforms
{
    public Vector4 PrevCamPos;
    public Vector4 PrevGridOrigin;
}

public unsafe class SurfelAgePass : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    private readonly GpuBuffer<AgeUniforms> _uniformBuffer;
    private readonly ComputePipeline* _pipeline;

    public SurfelAgePass(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx;
        _shaders = shaders;
        _uniformBuffer = new GpuBuffer<AgeUniforms>(ctx, 1, BufferUsage.Uniform, "AgeUniforms");
        _pipeline = _shaders.GetOrCreateComputePipeline("surfel_age.wgsl", "main");
    }

    public void Run(
        CommandEncoder* encoder,
        SurfelPool pool,
        SurfelHashGrid grid,
        Vector3 prevCameraPos,
        GpuBuffer<uint> indirectArgs)
    {
        Vector3 prevGridOrigin = Constants.SnapToSurfelGridOrigin(prevCameraPos);
        var uData = new AgeUniforms
        {
            PrevCamPos = new Vector4(prevCameraPos, 1f),
            PrevGridOrigin = new Vector4(prevGridOrigin, 1f)
        };
        _uniformBuffer.UploadSingle(in uData);

        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[7]
        {
            _uniformBuffer.AsBindGroupEntry(0),
            pool.SurfelBuffer.AsBindGroupEntry(1),
            pool.PoolBuffer.AsBindGroupEntry(2),
            pool.PoolAllocAtomicBuffer.AsBindGroupEntry(3),
            pool.PoolMaxAtomicBuffer.AsBindGroupEntry(4),
            pool.TouchedAtomicBuffer.AsBindGroupEntry(5),
            grid.OffsetsAndListBuffer.AsBindGroupEntry(6)
        };

        BindGroup* bg = _shaders.CreateBindGroup(_pipeline, 0, entries, "SurfelAgeBG");

        var passDesc = new ComputePassDescriptor();
        ComputePassEncoder* pass = _ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &passDesc);
        _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, _pipeline);
        _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        _ctx.Wgpu.ComputePassEncoderDispatchWorkgroupsIndirect(pass, indirectArgs.Handle, 0);
        _ctx.Wgpu.ComputePassEncoderEnd(pass);

        _ctx.Wgpu.ComputePassEncoderRelease(pass);
        _ctx.Wgpu.BindGroupRelease(bg);
    }

    public void Dispose()
    {
        _uniformBuffer.Dispose();
    }
}
