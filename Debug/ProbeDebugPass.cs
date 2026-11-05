using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Core.Native;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.GiPipeline;
using SurfelGI.Scene;

namespace SurfelGI.Debug;

[StructLayout(LayoutKind.Sequential)]
public struct ProbeDebugUniforms
{
    public Matrix4x4 ViewProjection, ProjInv, CamWorld;
    public Vector4 CamPos, Parameters;
    public uint Mode, Width, Height, Padding;
}

// Instanced camera-facing spheres; no probe readback or per-pixel search through the cache.
public unsafe sealed class ProbeDebugPass : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    private readonly GpuBuffer<ProbeDebugUniforms> _uniforms;
    private readonly RenderPipeline* _spheres;
    private readonly RenderPipeline* _surface;

    public ProbeDebugPass(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx; _shaders = shaders;
        _uniforms = new(ctx, 1, BufferUsage.Uniform, "ProbeDebugUniforms");
        _spheres = CreatePipeline("vs_probe", "fs_probe");
        _surface = CreatePipeline("vs_surface", "fs_surface");
    }

    private RenderPipeline* CreatePipeline(string vertex, string fragment)
    {
        var module = _shaders.GetOrCreateModule("probe_debug.wgsl");
        byte* vs = (byte*)SilkMarshal.StringToPtr(vertex);
        byte* fs = (byte*)SilkMarshal.StringToPtr(fragment);
        try {
            var blend = new BlendState {
                Color = new BlendComponent { SrcFactor = BlendFactor.SrcAlpha, DstFactor = BlendFactor.OneMinusSrcAlpha, Operation = BlendOperation.Add },
                Alpha = new BlendComponent { SrcFactor = BlendFactor.One, DstFactor = BlendFactor.OneMinusSrcAlpha, Operation = BlendOperation.Add }
            };
            var target = new ColorTargetState { Format = _ctx.SwapchainFormat, Blend = &blend, WriteMask = ColorWriteMask.All };
            var frag = new FragmentState { Module = module, EntryPoint = fs, TargetCount = 1, Targets = &target };
            var desc = new RenderPipelineDescriptor {
                Vertex = new VertexState { Module = module, EntryPoint = vs },
                Fragment = &frag,
                Primitive = new PrimitiveState { Topology = PrimitiveTopology.TriangleList, CullMode = CullMode.None },
                Multisample = new MultisampleState { Count = 1, Mask = ~0u }
            };
            return _ctx.Wgpu.DeviceCreateRenderPipeline(_ctx.Device, &desc);
        } finally { SilkMarshal.Free((nint)vs); SilkMarshal.Free((nint)fs); }
    }

    public void Render(CommandEncoder* encoder, TextureView* target, Camera camera,
        GBufferPass gbuffer, ProbeFallbackPass fallback, SurfelScreenDebugPass settings)
    {
        bool surface = settings.Mode == ScreenDebugMode.ProbeFallbackLighting;
        var pipeline = surface ? _surface : _spheres;
        _uniforms.UploadSingle(new ProbeDebugUniforms {
            ViewProjection = camera.ViewProjectionMatrix, ProjInv = camera.ProjectionMatrixInverse,
            CamWorld = camera.CameraWorldMatrix, CamPos = new(camera.Position, 1),
            Parameters = new(settings.ProbeLevel, settings.ProbeRadius, settings.ProbeXray ? 1 : 0, settings.ProbeExposure),
            Mode = (uint)settings.Mode, Width = (uint)gbuffer.Width, Height = (uint)gbuffer.Height
        });
        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[4] {
            _uniforms.AsBindGroupEntry(0), fallback.Buffer.AsBindGroupEntry(1),
            gbuffer.DepthTexture.AsTextureEntry(2), gbuffer.NormalTexture.AsTextureEntry(3)
        };
        var bg = _shaders.CreateBindGroup(pipeline, 0, surface ? entries : entries[..3], "ProbeDebug");
        var color = new RenderPassColorAttachment { View = target, LoadOp = LoadOp.Load, StoreOp = StoreOp.Store };
        var desc = new RenderPassDescriptor { ColorAttachmentCount = 1, ColorAttachments = &color };
        var pass = _ctx.Wgpu.CommandEncoderBeginRenderPass(encoder, &desc);
        _ctx.Wgpu.RenderPassEncoderSetPipeline(pass, pipeline);
        _ctx.Wgpu.RenderPassEncoderSetBindGroup(pass, 0, bg, 0, null);
        _ctx.Wgpu.RenderPassEncoderDraw(pass, surface ? 3u : 6u, surface ? 1u : 4096u, 0, 0);
        _ctx.Wgpu.RenderPassEncoderEnd(pass);
        _ctx.Wgpu.RenderPassEncoderRelease(pass);
        _ctx.Wgpu.BindGroupRelease(bg);
    }

    public void Dispose() {
        _ctx.Wgpu.RenderPipelineRelease(_spheres); _ctx.Wgpu.RenderPipelineRelease(_surface);
        _uniforms.Dispose();
    }
}
