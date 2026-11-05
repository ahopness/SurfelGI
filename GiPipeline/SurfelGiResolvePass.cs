using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.Scene;

namespace SurfelGI.GiPipeline;

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct ResolveUniforms
{
    public Matrix4x4 ProjInv;
    public Matrix4x4 CamWorld;
    public Vector4 CamPos;
    public Vector4 GridOrigin;
    public Vector4 OccParams;
    public uint Width;
    public uint Height;
    public uint ResolveOffset;
    public uint Frame;
    public uint SourceMode;
    public uint Pad0, Pad1, Pad2;
}

public unsafe class SurfelGiResolvePass : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    private readonly GpuBuffer<ResolveUniforms> _uniformBuffer;
    private readonly ComputePipeline* _pipeline;

    public GpuTexture OutputTexture { get; }

    public SurfelGiResolvePass(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx;
        _shaders = shaders;
        _uniformBuffer = new GpuBuffer<ResolveUniforms>(ctx, 1, BufferUsage.Uniform, "ResolveUniforms");
        OutputTexture = new GpuTexture(
            ctx,
            ctx.Width,
            ctx.Height,
            TextureFormat.Rgba16Float,
            TextureUsage.StorageBinding | TextureUsage.TextureBinding,
            filter: FilterMode.Linear,
            label: "SurfelGiOutput");

        _pipeline = _shaders.GetOrCreateComputePipeline("surfel_resolve.wgsl", "main");
    }

    public void Resize(int width, int height)
    {
        OutputTexture.Resize(width, height);
    }

    public void Run(
        CommandEncoder* encoder,
        Camera camera,
        GBufferPass gbuffer,
        SurfelPool pool,
        SurfelHashGrid grid,
        OcclusionSettings occlusion, ProbeFallbackPass fallback, bool halfResolution = true, GiSourceMode sourceMode = GiSourceMode.SurfelAndProbe)
    {
        int width = halfResolution ? (gbuffer.Width + 1) / 2 : gbuffer.Width;
        int height = halfResolution ? (gbuffer.Height + 1) / 2 : gbuffer.Height;
        OutputTexture.Resize(width, height);

        var (_, writeOffset) = pool.GetOffsets();
        Vector3 gridOrigin = Constants.SnapToSurfelGridOrigin(camera.Position);

        var uData = new ResolveUniforms
        {
            ProjInv = camera.ProjectionMatrixInverse,
            CamWorld = camera.CameraWorldMatrix,
            CamPos = new Vector4(camera.Position, 1f),
            GridOrigin = new Vector4(gridOrigin, 1f),
            OccParams = occlusion.ToVector4(),
            Width = (uint)width,
            Height = (uint)height,
            ResolveOffset = writeOffset,
            Frame = _ctx.FrameIndex,
            SourceMode = (uint)sourceMode
        };
        _uniformBuffer.UploadSingle(in uData);

        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[10]
        {
            _uniformBuffer.AsBindGroupEntry(0),
            gbuffer.DepthTexture.AsTextureEntry(1),
            gbuffer.NormalTexture.AsTextureEntry(2),
            pool.SurfelBuffer.AsBindGroupEntry(3),
            pool.MomentsBuffer.AsBindGroupEntry(4),
            grid.OffsetsAndListBuffer.AsBindGroupEntry(5),
            pool.TouchedAtomicBuffer.AsBindGroupEntry(6),
            pool.SurfelDepthBuffer.AsBindGroupEntry(7),
            OutputTexture.AsTextureEntry(8),
            fallback.Buffer.AsBindGroupEntry(9)
        };

        BindGroup* bg = _shaders.CreateBindGroup(_pipeline, 0, entries, "SurfelResolveBG");

        var passDesc = new ComputePassDescriptor();
        ComputePassEncoder* pass = _ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &passDesc);
        _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, _pipeline);
        _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        uint gx = (uint)((width + 7) / 8);
        uint gy = (uint)((height + 7) / 8);
        _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
        _ctx.Wgpu.ComputePassEncoderEnd(pass);

        _ctx.Wgpu.ComputePassEncoderRelease(pass);
        _ctx.Wgpu.BindGroupRelease(bg);
    }

    public void Dispose()
    {
        OutputTexture.Dispose();
        _uniformBuffer.Dispose();
    }
}
