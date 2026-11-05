using Silk.NET.WebGPU;
using SurfelGI.Core;

namespace SurfelGI.GiPipeline;

public unsafe class SurfelDispatchArgsPass : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    public GpuBuffer<uint> IndirectArgsBuffer { get; }
    private readonly ComputePipeline* _pipeline;

    public SurfelDispatchArgsPass(WebGpuContext ctx, ShaderManager shaders, string label = "SurfelDispatchArgs")
    {
        _ctx = ctx;
        _shaders = shaders;
        IndirectArgsBuffer = new GpuBuffer<uint>(
            ctx,
            new uint[] { 1, 1, 1 },
            BufferUsage.Storage | BufferUsage.Indirect,
            label);
        _pipeline = _shaders.GetOrCreateComputePipeline("surfel_dispatch_args.wgsl", "main");
    }

    public void Run(CommandEncoder* encoder, SurfelPool pool)
    {
        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[2]
        {
            pool.PoolMaxAtomicBuffer.AsBindGroupEntry(0),
            IndirectArgsBuffer.AsBindGroupEntry(1)
        };
        BindGroup* bg = _shaders.CreateBindGroup(_pipeline, 0, entries, "DispatchArgsBG");

        var passDesc = new ComputePassDescriptor();
        ComputePassEncoder* pass = _ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &passDesc);
        _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, _pipeline);
        _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        _ctx.Wgpu.ComputePassEncoderEnd(pass);

        _ctx.Wgpu.ComputePassEncoderRelease(pass);
        _ctx.Wgpu.BindGroupRelease(bg);
    }

    public void Dispose()
    {
        IndirectArgsBuffer.Dispose();
    }
}
