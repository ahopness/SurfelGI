using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.WebGPU;
using SurfelGI.Bvh;
using SurfelGI.Core;
using SurfelGI.Scene;

namespace SurfelGI.GiPipeline;

[StructLayout(LayoutKind.Sequential)]
public struct ProbeFallbackUniforms
{
    public Vector4 Camera, LightDirection, LightColor, Transport;
    public uint Frame, Reset, Padding0, Padding1;
}

// Five 16^3 toroidal levels. Each probe occupies fourteen vec4s (224 bytes).
public unsafe sealed class ProbeFallbackPass : IDisposable
{
    public const int ProbeCount = 5 * 16 * 16 * 16;
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    private readonly GpuBuffer<ProbeFallbackUniforms> _uniforms;
    private readonly ComputePipeline* _pipeline;
    private uint _frame;
    public GpuBuffer<Vector4> Buffer { get; }

    public ProbeFallbackPass(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx; _shaders = shaders;
        Buffer = new(ctx, ProbeCount * 14, BufferUsage.Storage | BufferUsage.CopySrc, "ProbeFallbackCache");
        _uniforms = new(ctx, 1, BufferUsage.Uniform, "ProbeFallbackUniforms");
        _pipeline = shaders.GetOrCreateComputePipeline("probe_fallback.wgsl", "main");
    }

    public void Run(CommandEncoder* encoder, SceneBvhBundle bvh, Camera camera,
        DirectionalLightData light, GpuTexture skybox, GiTransportSettings transport, bool reset = false)
    {
        if (reset) _frame = 0;
        _uniforms.UploadSingle(new ProbeFallbackUniforms {
            Camera = new(camera.Position, 1),
            LightDirection = new(light.GetLightDirectionToSource(), 0),
            LightColor = new(light.Color * light.Intensity, 1),
            Transport = new(transport.EnvIntensity, transport.EnvLod, transport.GiFromDirect, transport.AlbedoBoost),
            Frame = _frame, Reset = _frame == 0 ? 1u : 0u
        });
        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[] {
            _uniforms.AsBindGroupEntry(0), bvh.BvhBuffer.AsBindGroupEntry(1),
            bvh.PositionBuffer.AsBindGroupEntry(2), bvh.IndexBuffer.AsBindGroupEntry(3),
            bvh.AttributeBuffer.AsBindGroupEntry(4), Buffer.AsBindGroupEntry(5),
            bvh.DiffuseArrayTex.AsTextureEntry(6), bvh.DiffuseArrayTex.AsSamplerEntry(7),
            skybox.AsTextureEntry(8), skybox.AsSamplerEntry(9)
        };
        var bg = _shaders.CreateBindGroup(_pipeline, 0, entries, "ProbeFallback");
        var desc = new ComputePassDescriptor();
        var pass = _ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &desc);
        _ctx.Wgpu.ComputePassEncoderSetPipeline(pass, _pipeline);
        _ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, ProbeCount / 64, 1, 1);
        _ctx.Wgpu.ComputePassEncoderEnd(pass);
        _ctx.Wgpu.ComputePassEncoderRelease(pass);
        _ctx.Wgpu.BindGroupRelease(bg);
        _frame++;
    }

    public void Dispose() { Buffer.Dispose(); _uniforms.Dispose(); }
}
