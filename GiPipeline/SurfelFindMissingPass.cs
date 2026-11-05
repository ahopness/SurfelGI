using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.Scene;

namespace SurfelGI.GiPipeline;

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct FindMissingUniforms
{
    public Matrix4x4 ProjInv;
    public Matrix4x4 CamWorld;
    public Vector4 CamPos;
    public Vector4 PrevCamPos;
    public Vector4 PrevGridOrigin;
    public Vector2 ScreenSize;
    public uint GridStride;
    public uint Frame;
    public uint ReadOffset;
    public uint Pad0;
    public uint Pad1;
    public uint Pad2;
}

public unsafe class SurfelFindMissingPass : IDisposable
{
    private const int GroupSizeX = 8;
    private const int GroupSizeY = 8;

    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    private readonly GpuBuffer<FindMissingUniforms> _uniformBuffer;
    private readonly ComputePipeline* _pipeline;
    private readonly SurfelSpawnScheduler _scheduler;

    public int TileWidth { get; private set; }
    public Vector3 CameraPosition { get; private set; }
    public GpuBuffer<int> TileAllocBuffer { get; }
    public GpuBuffer<Vector4> CandidatePackedBuffer { get; }
    public GpuBuffer<Vector4> TileIrradianceBuffer { get; }

    public SurfelFindMissingPass(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx;
        _scheduler = new SurfelSpawnScheduler(ctx, shaders);
        _shaders = shaders;
        _uniformBuffer = new GpuBuffer<FindMissingUniforms>(ctx, 1, BufferUsage.Uniform, "FindMissingUniforms");

        const int initialTiles = 4096;
        TileAllocBuffer = new GpuBuffer<int>(ctx, initialTiles * 2, BufferUsage.Storage, "TileAllocBuffer");
        CandidatePackedBuffer = new GpuBuffer<Vector4>(ctx, initialTiles, BufferUsage.Storage, "CandidatePackedBuffer");
        TileIrradianceBuffer = new GpuBuffer<Vector4>(ctx, initialTiles, BufferUsage.Storage, "TileIrradianceBuffer");

        _pipeline = _shaders.GetOrCreateComputePipeline("surfel_find_missing.wgsl", "main");
    }

    private void EnsureTileBuffers(int tileCount)
    {
        int needed = Math.Max(1, tileCount);
        TileAllocBuffer.EnsureCapacity(needed * 2);
        CandidatePackedBuffer.EnsureCapacity(needed);
        TileIrradianceBuffer.EnsureCapacity(needed);
    }

    public int Run(
        CommandEncoder* encoder,
        Camera camera,
        GBufferPass gbuffer,
        SurfelPool pool,
        SurfelHashGrid grid,
        Vector3 prevCameraPos)
    {
        int width = gbuffer.Width;
        int height = gbuffer.Height;
        int tileW = Math.Max(1, (width + GroupSizeX - 1) / GroupSizeX);
        int tileH = Math.Max(1, (height + GroupSizeY - 1) / GroupSizeY);
        int tileCount = tileW * tileH;

        TileWidth = tileW;
        CameraPosition = camera.Position;
        EnsureTileBuffers(tileCount);

        var (readOffset, _) = pool.GetOffsets();
        Vector3 prevGridOrigin = Constants.SnapToSurfelGridOrigin(prevCameraPos);

        var uData = new FindMissingUniforms
        {
            ProjInv = camera.ProjectionMatrixInverse,
            CamWorld = camera.CameraWorldMatrix,
            CamPos = new Vector4(camera.Position, 1f),
            PrevCamPos = new Vector4(prevCameraPos, 1f),
            PrevGridOrigin = new Vector4(prevGridOrigin, 1f),
            ScreenSize = new Vector2(width, height),
            GridStride = (uint)tileW,
            Frame = _ctx.FrameIndex,
            ReadOffset = readOffset
        };
        _uniformBuffer.UploadSingle(in uData);

        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[10]
        {
            _uniformBuffer.AsBindGroupEntry(0),
            gbuffer.DepthTexture.AsTextureEntry(1),
            gbuffer.NormalTexture.AsTextureEntry(2),
            pool.SurfelBuffer.AsBindGroupEntry(3),
            grid.OffsetsAndListBuffer.AsBindGroupEntry(4),
            pool.MomentsBuffer.AsBindGroupEntry(5),
            pool.TouchedAtomicBuffer.AsBindGroupEntry(6),
            TileAllocBuffer.AsBindGroupEntry(7),
            CandidatePackedBuffer.AsBindGroupEntry(8),
            TileIrradianceBuffer.AsBindGroupEntry(9)
        };

        BindGroup* bg = _shaders.CreateBindGroup(_pipeline, 0, entries, "FindMissingBG");

        var passDesc = new ComputePassDescriptor();
        ComputePassEncoder* pass = _ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &passDesc);
        _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, _pipeline);
        _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, (uint)tileW, (uint)tileH, 1);
        _ctx.Wgpu.ComputePassEncoderEnd(pass);

        _ctx.Wgpu.ComputePassEncoderRelease(pass);
        _ctx.Wgpu.BindGroupRelease(bg);

        _scheduler.Run(encoder, TileAllocBuffer, tileCount, _ctx.FrameIndex);
        return tileCount;
    }

    public void Dispose()
    {
        _scheduler.Dispose();
        TileIrradianceBuffer.Dispose();
        CandidatePackedBuffer.Dispose();
        TileAllocBuffer.Dispose();
        _uniformBuffer.Dispose();
    }
}
