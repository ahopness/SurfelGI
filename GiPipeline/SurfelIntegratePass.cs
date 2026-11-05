using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.WebGPU;
using SurfelGI.Bvh;
using SurfelGI.Core;
using SurfelGI.Scene;

namespace SurfelGI.GiPipeline;

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct IntegrateUniforms
{
    public Vector4 LightDir;
    public Vector4 LightColor;
    public Vector4 CamPos;
    public Vector4 GridOrigin;
    public Vector4 OccParams;
    public uint Frame;
    public uint ReadOffset;
    public uint WriteOffset;
    public uint BaseSampleCount;
    public float AlbedoBoost;
    public float GiFromDirect;
    public float GiFromIndirect;
    public float EnvIntensity;
    public float EnvLod;
    public float Pad0;
    public float Pad1;
    public float Pad2;
}

public unsafe class SurfelIntegratePass : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    private readonly GpuTexture _blueNoiseTex;
    private readonly GpuBuffer<IntegrateUniforms> _uniformBuffer;
    private readonly ComputePipeline* _pipeline;

    public SurfelIntegratePass(WebGpuContext ctx, ShaderManager shaders, GpuTexture blueNoiseTex)
    {
        _ctx = ctx;
        _shaders = shaders;
        _blueNoiseTex = blueNoiseTex;
        _uniformBuffer = new GpuBuffer<IntegrateUniforms>(ctx, 1, BufferUsage.Uniform, "IntegrateUniforms");
        _pipeline = _shaders.GetOrCreateComputePipeline("surfel_integrate.wgsl", "main");
    }

    public void Run(
        CommandEncoder* encoder,
        SurfelPool pool,
        SceneBvhBundle bvh,
        SurfelHashGrid grid,
        Camera camera,
        DirectionalLightData dirLight,
        GpuTexture envTex,
        GiTransportSettings transport,
        OcclusionSettings occlusion,
        GpuBuffer<uint> indirectArgs)
    {
        var (readOffset, writeOffset) = pool.GetOffsets();
        Vector3 lightDir = dirLight.GetLightDirectionToSource();
        Vector3 lightColor = dirLight.Color * dirLight.Intensity;
        Vector3 gridOrigin = Constants.SnapToSurfelGridOrigin(camera.Position);

        var uData = new IntegrateUniforms
        {
            LightDir = new Vector4(lightDir, 0f),
            LightColor = new Vector4(lightColor, 1f),
            CamPos = new Vector4(camera.Position, 1f),
            GridOrigin = new Vector4(gridOrigin, 1f),
            OccParams = occlusion.ToVector4(),
            Frame = _ctx.FrameIndex,
            ReadOffset = readOffset,
            WriteOffset = writeOffset,
            BaseSampleCount = (uint)Math.Max(1, transport.BaseSampleCount),
            AlbedoBoost = MathF.Max(0f, transport.AlbedoBoost),
            GiFromDirect = MathF.Max(0f, transport.GiFromDirect),
            GiFromIndirect = MathF.Max(0f, transport.GiFromIndirect),
            EnvIntensity = MathF.Max(0f, transport.EnvIntensity),
            Pad0 = transport.StaggerUpdates ? 1f : 0f,
            EnvLod = MathF.Max(0f, transport.EnvLod)
        };
        _uniformBuffer.UploadSingle(in uData);

        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[16]
        {
            _uniformBuffer.AsBindGroupEntry(0),
            bvh.BvhBuffer.AsBindGroupEntry(1),
            bvh.PositionBuffer.AsBindGroupEntry(2),
            bvh.IndexBuffer.AsBindGroupEntry(3),
            bvh.AttributeBuffer.AsBindGroupEntry(4),
            pool.SurfelBuffer.AsBindGroupEntry(5),
            pool.MomentsBuffer.AsBindGroupEntry(6),
            grid.OffsetsAndListBuffer.AsBindGroupEntry(7),
            pool.TouchedAtomicBuffer.AsBindGroupEntry(8),
            pool.GuidingBuffer.AsBindGroupEntry(9),
            pool.SurfelDepthBuffer.AsBindGroupEntry(10),
            bvh.DiffuseArrayTex.AsTextureEntry(11),
            bvh.DiffuseArrayTex.AsSamplerEntry(12),
            envTex.AsTextureEntry(13),
            envTex.AsSamplerEntry(14),
            _blueNoiseTex.AsTextureEntry(15)
        };

        BindGroup* bg = _shaders.CreateBindGroup(_pipeline, 0, entries, "SurfelIntegrateBG");

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
