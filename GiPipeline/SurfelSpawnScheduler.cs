using Silk.NET.WebGPU;
using SurfelGI.Core;

namespace SurfelGI.GiPipeline;

public unsafe sealed class SurfelSpawnScheduler : IDisposable
{
    public const uint SpawnBudget = 1024;
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    private readonly GpuBuffer<uint> _uniform;
    private readonly ComputePipeline* _select;
    private readonly ComputePipeline* _finalize;
    public SurfelSpawnScheduler(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx; _shaders = shaders;
        _uniform = new GpuBuffer<uint>(ctx, 4, BufferUsage.Uniform, "SpawnScheduleUniforms");
        _select = shaders.GetOrCreateComputePipeline("surfel_schedule.wgsl", "select_candidates");
        _finalize = shaders.GetOrCreateComputePipeline("surfel_schedule.wgsl", "finalize_candidates");
    }
    public void Run(CommandEncoder* encoder, GpuBuffer<int> tiles, int tileCount, uint frame)
    {
        if (tileCount <= 0) return;
        _uniform.Upload(new uint[] { (uint)tileCount, frame, SpawnBudget, 0 });
        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[] { _uniform.AsBindGroupEntry(0), tiles.AsBindGroupEntry(1) };
        var desc = new ComputePassDescriptor();
        var pass = _ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &desc);
        Dispatch(pass, _select, entries, (SpawnBudget+63)/64);
        Dispatch(pass, _finalize, entries, ((uint)tileCount+63)/64);
        _ctx.Wgpu.ComputePassEncoderEnd(pass); _ctx.Wgpu.ComputePassEncoderRelease(pass);
    }
    private void Dispatch(ComputePassEncoder* pass, ComputePipeline* pipeline, ReadOnlySpan<BindGroupEntry> entries, uint groups)
    {
        var bg = _shaders.CreateBindGroup(pipeline, 0, entries, "SpawnScheduleBG");
        _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, pipeline);
        _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, groups, 1, 1);
        _ctx.Wgpu.BindGroupRelease(bg);
    }
    public void Dispose() => _uniform.Dispose();
}
