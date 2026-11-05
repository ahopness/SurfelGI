using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Core.Native;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.Scene;

namespace SurfelGI.GiPipeline;

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct CompositeUniforms
{
    public Matrix4x4 ProjInv;
    public Matrix4x4 CamWorld;
    public Matrix4x4 ViewMat;
    public Matrix4x4 CsmLightViewProj0;
    public Matrix4x4 CsmLightViewProj1;
    public Matrix4x4 CsmLightViewProj2;
    public Matrix4x4 CsmLightViewProj3;
    public Vector4 CsmSplits;
    public Vector4 LightDir;
    public Vector4 LightColor;
    public Vector4 CamPos;
    public Vector2 ScreenSize;
    public uint GiMode;
    public float IndirectIntensity;
    public float ShadowBias;
    public float Exposure;
    public uint CsmEnabled;
    public uint Pad0;
}

public unsafe class CompositePass : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    private readonly GpuBuffer<CompositeUniforms> _uniformBuffer;
    private RenderPipeline* _pipeline;

    public CompositePass(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx;
        _shaders = shaders;
        _uniformBuffer = new GpuBuffer<CompositeUniforms>(ctx, 1, BufferUsage.Uniform, "CompositeUniforms");
        CreatePipeline();
    }

    private void CreatePipeline()
    {
        ShaderModule* module = _shaders.GetOrCreateModule("direct_csm_composite.wgsl");

        var colorTarget = new ColorTargetState
        {
            Format = _ctx.SwapchainFormat,
            WriteMask = ColorWriteMask.All
        };

        byte* vsEntry = (byte*)SilkMarshal.StringToPtr("vs_fullscreen");
        byte* fsEntry = (byte*)SilkMarshal.StringToPtr("fs_main");
        byte* labelPtr = (byte*)SilkMarshal.StringToPtr("CompositePipeline");

        try
        {
            var fragState = new FragmentState
            {
                Module = module,
                EntryPoint = fsEntry,
                TargetCount = 1,
                Targets = &colorTarget
            };

            var desc = new RenderPipelineDescriptor
            {
                Label = labelPtr,
                Vertex = new VertexState
                {
                    Module = module,
                    EntryPoint = vsEntry,
                    BufferCount = 0,
                    Buffers = null
                },
                Primitive = new PrimitiveState
                {
                    Topology = PrimitiveTopology.TriangleList,
                    CullMode = CullMode.None
                },
                Multisample = new MultisampleState
                {
                    Count = 1,
                    Mask = ~0u
                },
                Fragment = &fragState
            };

            _pipeline = _ctx.Wgpu.DeviceCreateRenderPipeline(_ctx.Device, &desc);
        }
        finally
        {
            SilkMarshal.Free((nint)vsEntry);
            SilkMarshal.Free((nint)fsEntry);
            SilkMarshal.Free((nint)labelPtr);
        }
    }

    public void Execute(
        CommandEncoder* encoder,
        TextureView* targetView,
        Camera camera,
        SceneGraph scene,
        GBufferPass gbuffer,
        SurfelGiResolvePass resolvePass,
        CsmShadowSystem csm,
        GpuTexture envTex,
        GiTransportSettings transport)
    {
        var light = scene.DirLight;
        Vector3 lightDir = light.GetLightDirectionToSource();
        Vector3 lightColor = light.Color * light.Intensity;

        var uData = new CompositeUniforms
        {
            ProjInv = camera.ProjectionMatrixInverse,
            CamWorld = camera.CameraWorldMatrix,
            ViewMat = camera.ViewMatrix,
            CsmLightViewProj0 = csm.CascadeViewProj[0],
            CsmLightViewProj1 = csm.CascadeViewProj[1],
            CsmLightViewProj2 = csm.CascadeViewProj[2],
            CsmLightViewProj3 = csm.CascadeViewProj[3],
            CsmSplits = csm.CascadeSplits,
            LightDir = new Vector4(lightDir, 0f),
            LightColor = new Vector4(lightColor, 1f),
            CamPos = new Vector4(camera.Position, 1f),
            ScreenSize = new Vector2(gbuffer.Width, gbuffer.Height),
            GiMode = (uint)transport.Mode,
            IndirectIntensity = transport.IndirectIntensity,
            ShadowBias = light.ShadowBias,
            Exposure = MathF.Max(0f, transport.Exposure),
            CsmEnabled = light.CastShadow ? 1u : 0u
        };
        _uniformBuffer.UploadSingle(in uData);

        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[8]
        {
            _uniformBuffer.AsBindGroupEntry(0),
            gbuffer.DepthTexture.AsTextureEntry(1),
            gbuffer.NormalTexture.AsTextureEntry(2),
            gbuffer.AlbedoTexture.AsTextureEntry(3),
            resolvePass.OutputTexture.AsTextureEntry(4),
            envTex.AsTextureEntry(5),
            envTex.AsSamplerEntry(6),
            csm.ShadowMapArray.AsTextureEntry(7)
        };

        BindGroup* bg = _shaders.CreateBindGroup(_pipeline, 0, entries, "CompositeBG");

        var colorAttach = new RenderPassColorAttachment
        {
            View = targetView,
            LoadOp = LoadOp.Clear,
            StoreOp = StoreOp.Store,
            ClearValue = new Color(0.12, 0.12, 0.15, 1.0)
        };

        var passDesc = new RenderPassDescriptor
        {
            ColorAttachmentCount = 1,
            ColorAttachments = &colorAttach,
            DepthStencilAttachment = null
        };

        RenderPassEncoder* pass = _ctx.Wgpu.CommandEncoderBeginRenderPass(encoder, &passDesc);
        _ctx.Wgpu.RenderPassEncoderSetPipeline(pass, _pipeline);
        _ctx.Wgpu.RenderPassEncoderSetBindGroup(pass, 0, bg, 0, null);
        _ctx.Wgpu.RenderPassEncoderDraw(pass, 3, 1, 0, 0);
        _ctx.Wgpu.RenderPassEncoderEnd(pass);

        _ctx.Wgpu.RenderPassEncoderRelease(pass);
        _ctx.Wgpu.BindGroupRelease(bg);
    }

    public void Dispose()
    {
        if (_pipeline != null) _ctx.Wgpu.RenderPipelineRelease(_pipeline);
        _uniformBuffer.Dispose();
    }
}
