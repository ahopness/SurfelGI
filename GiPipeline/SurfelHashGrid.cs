using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.Scene;

namespace SurfelGI.GiPipeline;

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct GridUniforms
{
    public Vector4 CamPos;
    public Vector4 GridOrigin;
    public int TotalCells;
    public int Segments;
    public int ListCapacity;
    public int Pad0;
}

public unsafe class SurfelHashGrid : IDisposable
{
    private const int WorkgroupSize = 512;
    private const int SegmentSize = WorkgroupSize * 2;

    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;

    public GpuBuffer<int> OffsetsAndListBuffer { get; }
    public GpuBuffer<int> SegmentSumBuffer { get; }
    private readonly GpuBuffer<GridUniforms> _uniformBuffer;
    private readonly int _segments;

    private readonly ComputePipeline* _clearPipeline;
    private readonly ComputePipeline* _countPipeline;
    private readonly ComputePipeline* _scanSegPipeline;
    private readonly ComputePipeline* _collectSegPipeline;
    private readonly ComputePipeline* _segPrefixPipeline;
    private readonly ComputePipeline* _mergePipeline;
    private readonly ComputePipeline* _slotPipeline;

    public SurfelHashGrid(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx;
        _shaders = shaders;

        int headerSize = Constants.TOTAL_CELLS + 1;
        int listSize = (Constants.TOTAL_CELLS + 1) * Constants.MAX_SURFELS_PER_CELL;
        int needed = headerSize + listSize;

        OffsetsAndListBuffer = new GpuBuffer<int>(ctx, needed, BufferUsage.Storage, "OffsetsAndList");
        _segments = Math.Max(1, (int)Math.Ceiling((Constants.TOTAL_CELLS + 1.0) / SegmentSize));
        SegmentSumBuffer = new GpuBuffer<int>(ctx, _segments, BufferUsage.Storage, "SegmentSums");
        _uniformBuffer = new GpuBuffer<GridUniforms>(ctx, 1, BufferUsage.Uniform, "GridUniforms");

        _clearPipeline = _shaders.GetOrCreateComputePipeline("surfel_grid_build.wgsl", "grid_clear");
        _countPipeline = _shaders.GetOrCreateComputePipeline("surfel_grid_build.wgsl", "grid_count");
        _scanSegPipeline = _shaders.GetOrCreateComputePipeline("surfel_grid_build.wgsl", "grid_scan_seg");
        _collectSegPipeline = _shaders.GetOrCreateComputePipeline("surfel_grid_build.wgsl", "grid_collect_seg");
        _segPrefixPipeline = _shaders.GetOrCreateComputePipeline("surfel_grid_build.wgsl", "grid_seg_prefix");
        _mergePipeline = _shaders.GetOrCreateComputePipeline("surfel_grid_build.wgsl", "grid_merge");
        _slotPipeline = _shaders.GetOrCreateComputePipeline("surfel_grid_build.wgsl", "grid_slot");
    }

    public void Build(CommandEncoder* encoder, SurfelPool pool, Camera camera, GpuBuffer<uint>? indirectArgs = null)
    {
        Vector3 gridOrigin = Constants.SnapToSurfelGridOrigin(camera.Position);
        var uData = new GridUniforms
        {
            CamPos = new Vector4(camera.Position, 1f),
            GridOrigin = new Vector4(gridOrigin, 1f),
            TotalCells = Constants.TOTAL_CELLS,
            Segments = _segments,
            ListCapacity = OffsetsAndListBuffer.Count
        };
        _uniformBuffer.UploadSingle(in uData);

        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[5]
        {
            _uniformBuffer.AsBindGroupEntry(0),
            pool.SurfelBuffer.AsBindGroupEntry(1),
            pool.PoolMaxAtomicBuffer.AsBindGroupEntry(2),
            OffsetsAndListBuffer.AsBindGroupEntry(3),
            SegmentSumBuffer.AsBindGroupEntry(4)
        };

        var passDesc = new ComputePassDescriptor();
        ComputePassEncoder* pass = _ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &passDesc);

        uint clearGroups = (uint)((Constants.TOTAL_CELLS + 1 + 255) / 256);
        uint surfelGroups = (uint)((pool.Capacity + 63) / 64);
        uint collectGroups = (uint)((_segments + 63) / 64);

        DispatchStep(pass, _clearPipeline, entries, clearGroups, 1, 1);
        if (indirectArgs != null)
            DispatchStepIndirect(pass, _countPipeline, entries, indirectArgs);
        else
            DispatchStep(pass, _countPipeline, entries, surfelGroups, 1, 1);
        DispatchStep(pass, _scanSegPipeline, entries, (uint)_segments, 1, 1);
        DispatchStep(pass, _collectSegPipeline, entries, collectGroups, 1, 1);
        DispatchStep(pass, _segPrefixPipeline, entries, 1, 1, 1);
        DispatchStep(pass, _mergePipeline, entries, clearGroups, 1, 1);
        if (indirectArgs != null)
            DispatchStepIndirect(pass, _slotPipeline, entries, indirectArgs);
        else
            DispatchStep(pass, _slotPipeline, entries, surfelGroups, 1, 1);

        _ctx.Wgpu.ComputePassEncoderEnd(pass);
        _ctx.Wgpu.ComputePassEncoderRelease(pass);
    }

    private void DispatchStep(ComputePassEncoder* pass, ComputePipeline* pipeline, ReadOnlySpan<BindGroupEntry> entries, uint gx, uint gy, uint gz)
    {
        BindGroup* bg = _shaders.CreateBindGroup(pipeline, 0, entries, "GridBuildBG");
        _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, pipeline);
        _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, gx, gy, gz);
        _ctx.Wgpu.BindGroupRelease(bg);
    }

    private void DispatchStepIndirect(ComputePassEncoder* pass, ComputePipeline* pipeline, ReadOnlySpan<BindGroupEntry> entries, GpuBuffer<uint> indirectArgs)
    {
        BindGroup* bg = _shaders.CreateBindGroup(pipeline, 0, entries, "GridBuildIndirectBG");
        _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, pipeline);
        _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        _ctx.Wgpu.ComputePassEncoderDispatchWorkgroupsIndirect(pass, indirectArgs.Handle, 0);
        _ctx.Wgpu.BindGroupRelease(bg);
    }

    public void Dispose()
    {
        _uniformBuffer.Dispose();
        SegmentSumBuffer.Dispose();
        OffsetsAndListBuffer.Dispose();
    }
}
