using Silk.NET.WebGPU;
using SurfelGI.Core;

namespace SurfelGI.GiPipeline;

public unsafe class SurfelPreparePass : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    private readonly GpuBuffer<int> _spawnCounterAtomic;
    private readonly GpuBuffer<int> _retiredCounterAtomic;

    private ComputePipeline* _initPipeline;
    private ComputePipeline* _resetPipeline;
    private bool _isInitialized;
    private int _currentCapacity;

    public SurfelPreparePass(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx;
        _shaders = shaders;
        _spawnCounterAtomic = new GpuBuffer<int>(ctx, new[] { 0 }, BufferUsage.Storage, "SpawnCounterAtomic");
        _retiredCounterAtomic = new GpuBuffer<int>(ctx, new[] { 0 }, BufferUsage.Storage, "RetiredCounterAtomic");

        _initPipeline = _shaders.GetOrCreateComputePipeline("surfel_prepare.wgsl", "init_pool");
        _resetPipeline = _shaders.GetOrCreateComputePipeline("surfel_prepare.wgsl", "reset_counters");
    }

    public void Run(CommandEncoder* encoder, SurfelPool pool, bool forceClear = false)
    {
        int capacity = pool.Capacity;

        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[6]
        {
            pool.SurfelBuffer.AsBindGroupEntry(0),
            pool.PoolBuffer.AsBindGroupEntry(1),
            pool.PoolAllocAtomicBuffer.AsBindGroupEntry(2),
            pool.PoolMaxAtomicBuffer.AsBindGroupEntry(3),
            _spawnCounterAtomic.AsBindGroupEntry(4),
            _retiredCounterAtomic.AsBindGroupEntry(5)
        };

        var passDesc = new ComputePassDescriptor();
        ComputePassEncoder* pass = _ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &passDesc);

        if (!_isInitialized || _currentCapacity != capacity || forceClear)
        {
            BindGroup* bgInit = _shaders.CreateBindGroup(_initPipeline, 0, entries, "PrepareInitBG");
            _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, _initPipeline);
            _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bgInit, 0, null);
            uint groups = (uint)((capacity + 63) / 64);
            _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, groups, 1, 1);
            _ctx.Wgpu.BindGroupRelease(bgInit);

            _isInitialized = true;
            _currentCapacity = capacity;
        }
        else
        {
            Span<BindGroupEntry> resetEntries = stackalloc BindGroupEntry[2]
            {
                _spawnCounterAtomic.AsBindGroupEntry(4),
                _retiredCounterAtomic.AsBindGroupEntry(5)
            };
            BindGroup* bgReset = _shaders.CreateBindGroup(_resetPipeline, 0, resetEntries, "PrepareResetBG");
            _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, _resetPipeline);
            _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bgReset, 0, null);
            _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
            _ctx.Wgpu.BindGroupRelease(bgReset);
        }

        _ctx.Wgpu.ComputePassEncoderEnd(pass);
        _ctx.Wgpu.ComputePassEncoderRelease(pass);
    }

    public void Dispose()
    {
        _spawnCounterAtomic.Dispose();
        _retiredCounterAtomic.Dispose();
    }
}
