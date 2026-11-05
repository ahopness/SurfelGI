using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Core.Native;
using Silk.NET.WebGPU;
using SurfelGI.Bvh;
using SurfelGI.Core;
using SurfelGI.GiPipeline;
using SurfelGI.Scene;

namespace SurfelGI.Debug;

public enum ScreenDebugMode : uint
{
    Off = 0,
    Surfels = 1,
    Irradiance = 2,
    Heatmap = 3,
    Cascades = 4,
    CellHash = 5,
    GridCoords = 6,
    WorldPos = 7,
    Depth = 8,
    Normals = 9,
    Albedo = 10,
    Coverage = 11,
    SLGGrid = 12,
    SpawnDespawn = 13,
    RadialOcclusion = 14,
    RadialDepthTile = 15,
    Variance = 16,
    AlbedoBVH = 17,
    AlbedoMatch = 18,
    ProbeValidity = 19,
    ProbeIrradiance = 20,
    ProbeDepth = 21,
    ProbeRelocation = 22,
    ProbeFallbackLighting = 23
}

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct DebugUniforms
{
    public Matrix4x4 ProjInv;
    public Matrix4x4 CamWorld;
    public Vector4 CamPos;
    public Vector4 PrevCamPos;
    public Vector4 GridOrigin;
    public Vector4 OccParams;
    public uint Width;
    public uint Height;
    public uint MomentsOffset;
    public uint Frame;
    public uint GridStride;
    public uint DebugMode;
    public float VarianceScale;
    public int MaxFetch;
}

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct BlitUniforms
{
    public float Width;
    public float Height;
    public float Pad0;
    public float Pad1;
}

public unsafe class SurfelScreenDebugPass : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly ProbeDebugPass _probeDebug;
    private readonly ShaderManager _shaders;
    private readonly GpuBuffer<DebugUniforms> _uniformBuffer;
    private readonly GpuBuffer<BlitUniforms> _blitUniformBuffer;
    public GpuBuffer<Vector4> OverlayBuffer { get; }

    private readonly ComputePipeline* _mainComputePipeline;
    private readonly ComputePipeline* _bvhComputePipeline;
    private RenderPipeline* _blitPipeline;

    public ScreenDebugMode Mode { get; set; } = ScreenDebugMode.Off;
    public int ProbeLevel { get; set; }
    public float ProbeRadius { get; set; } = 0.08f;
    public bool ProbeXray { get; set; }
    public float ProbeExposure { get; set; } = 1f;
    public float VarianceScale { get; set; } = 1.0f;
    public int MaxFetchPerPixel { get; set; } = 1024;

    public SurfelScreenDebugPass(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx;
        _shaders = shaders;
        _probeDebug = new ProbeDebugPass(ctx, shaders);
        _uniformBuffer = new GpuBuffer<DebugUniforms>(ctx, 1, BufferUsage.Uniform, "DebugUniforms");
        _blitUniformBuffer = new GpuBuffer<BlitUniforms>(ctx, 1, BufferUsage.Uniform, "BlitUniforms");
        OverlayBuffer = new GpuBuffer<Vector4>(ctx, Math.Max(1, ctx.Width * ctx.Height), BufferUsage.Storage, "DebugOverlayBuffer");

        _mainComputePipeline = _shaders.GetOrCreateComputePipeline("surfel_screen_debug.wgsl", "cs_main");
        _bvhComputePipeline = _shaders.GetOrCreateComputePipeline("surfel_screen_debug.wgsl", "cs_bvh");
        CreateBlitPipeline();
    }

    private void CreateBlitPipeline()
    {
        ShaderModule* module = _shaders.GetOrCreateModule("surfel_screen_debug.wgsl");

        var blendState = new BlendState
        {
            Color = new BlendComponent
            {
                SrcFactor = BlendFactor.One,
                DstFactor = BlendFactor.OneMinusSrcAlpha,
                Operation = BlendOperation.Add
            },
            Alpha = new BlendComponent
            {
                SrcFactor = BlendFactor.One,
                DstFactor = BlendFactor.OneMinusSrcAlpha,
                Operation = BlendOperation.Add
            }
        };

        var colorTarget = new ColorTargetState
        {
            Format = _ctx.SwapchainFormat,
            Blend = &blendState,
            WriteMask = ColorWriteMask.All
        };

        byte* vsEntry = (byte*)SilkMarshal.StringToPtr("vs_blit");
        byte* fsEntry = (byte*)SilkMarshal.StringToPtr("fs_blit");
        byte* labelPtr = (byte*)SilkMarshal.StringToPtr("DebugOverlayBlit");

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

            _blitPipeline = _ctx.Wgpu.DeviceCreateRenderPipeline(_ctx.Device, &desc);
        }
        finally
        {
            SilkMarshal.Free((nint)vsEntry);
            SilkMarshal.Free((nint)fsEntry);
            SilkMarshal.Free((nint)labelPtr);
        }
    }

    public void RunAndRenderOverlay(
        CommandEncoder* encoder,
        TextureView* targetView,
        Camera camera,
        GBufferPass gbuffer,
        SurfelPool pool,
        SurfelHashGrid grid,
        SurfelFindMissingPass find,
        SceneBvhBundle bvh,
        OcclusionSettings occlusion,
        Vector3 prevCameraPos,
        ProbeFallbackPass fallback)
    {
        if (Mode == ScreenDebugMode.Off)
            return;

        if (Mode >= ScreenDebugMode.ProbeValidity) {
            _probeDebug.Render(encoder, targetView, camera, gbuffer, fallback, this);
            return;
        }

        int width = gbuffer.Width;
        int height = gbuffer.Height;
        OverlayBuffer.EnsureCapacity(Math.Max(1, width * height));

        var (_, writeOffset) = pool.GetOffsets();
        Vector3 gridOrigin = Constants.SnapToSurfelGridOrigin(camera.Position);

        var uData = new DebugUniforms
        {
            ProjInv = camera.ProjectionMatrixInverse,
            CamWorld = camera.CameraWorldMatrix,
            CamPos = new Vector4(camera.Position, 1f),
            PrevCamPos = new Vector4(prevCameraPos, 1f),
            GridOrigin = new Vector4(gridOrigin, 1f),
            OccParams = occlusion.ToVector4(),
            Width = (uint)width,
            Height = (uint)height,
            MomentsOffset = writeOffset,
            Frame = _ctx.FrameIndex,
            GridStride = (uint)Math.Max(1, (width + 7) / 8),
            DebugMode = (uint)Mode,
            VarianceScale = VarianceScale,
            MaxFetch = MaxFetchPerPixel
        };
        _uniformBuffer.UploadSingle(in uData);

        bool isBvhMode = Mode == ScreenDebugMode.AlbedoBVH || Mode == ScreenDebugMode.AlbedoMatch;
        uint gx = (uint)((width + 7) / 8);
        uint gy = (uint)((height + 7) / 8);

        var passDesc = new ComputePassDescriptor();
        ComputePassEncoder* cPass = _ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &passDesc);

        if (isBvhMode)
        {
            Span<BindGroupEntry> bvhEntries = stackalloc BindGroupEntry[10]
            {
                _uniformBuffer.AsBindGroupEntry(0),
                gbuffer.DepthTexture.AsTextureEntry(1),
                gbuffer.AlbedoTexture.AsTextureEntry(3),
                OverlayBuffer.AsBindGroupEntry(10),
                bvh.BvhBuffer.AsBindGroupEntry(11),
                bvh.PositionBuffer.AsBindGroupEntry(12),
                bvh.IndexBuffer.AsBindGroupEntry(13),
                bvh.AttributeBuffer.AsBindGroupEntry(14),
                bvh.DiffuseArrayTex.AsTextureEntry(15),
                bvh.DiffuseArrayTex.AsSamplerEntry(16)
            };
            BindGroup* bgBvh = _shaders.CreateBindGroup(_bvhComputePipeline, 0, bvhEntries, "DebugBvhBG");
            _ctx.Wgpu.ComputePassEncoderSetPipeline(cPass, _bvhComputePipeline);
            _ctx.Wgpu.ComputePassEncoderSetBindGroup(cPass, 0, bgBvh, 0, null);
            _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(cPass, gx, gy, 1);
            _ctx.Wgpu.BindGroupRelease(bgBvh);
        }
        else
        {
            Span<BindGroupEntry> mainEntries = stackalloc BindGroupEntry[11]
            {
                _uniformBuffer.AsBindGroupEntry(0),
                gbuffer.DepthTexture.AsTextureEntry(1),
                gbuffer.NormalTexture.AsTextureEntry(2),
                gbuffer.AlbedoTexture.AsTextureEntry(3),
                pool.SurfelBuffer.AsBindGroupEntry(4),
                pool.MomentsBuffer.AsBindGroupEntry(5),
                grid.OffsetsAndListBuffer.AsBindGroupEntry(6),
                pool.GuidingBuffer.AsBindGroupEntry(7),
                find.TileAllocBuffer.AsBindGroupEntry(8),
                pool.SurfelDepthBuffer.AsBindGroupEntry(9),
                OverlayBuffer.AsBindGroupEntry(10)
            };
            BindGroup* bgMain = _shaders.CreateBindGroup(_mainComputePipeline, 0, mainEntries, "DebugMainBG");
            _ctx.Wgpu.ComputePassEncoderSetPipeline(cPass, _mainComputePipeline);
            _ctx.Wgpu.ComputePassEncoderSetBindGroup(cPass, 0, bgMain, 0, null);
            _ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(cPass, gx, gy, 1);
            _ctx.Wgpu.BindGroupRelease(bgMain);
        }

        _ctx.Wgpu.ComputePassEncoderEnd(cPass);
        _ctx.Wgpu.ComputePassEncoderRelease(cPass);

        // Blit overlay
        var blitU = new BlitUniforms { Width = width, Height = height };
        _blitUniformBuffer.UploadSingle(in blitU);

        Span<BindGroupEntry> blitEntries = stackalloc BindGroupEntry[2]
        {
            _blitUniformBuffer.AsBindGroupEntry(0),
            OverlayBuffer.AsBindGroupEntry(1)
        };
        BindGroup* bgBlit = _shaders.CreateBindGroup(_blitPipeline, 0, blitEntries, "DebugBlitBG");

        var colorAttach = new RenderPassColorAttachment
        {
            View = targetView,
            LoadOp = LoadOp.Load,
            StoreOp = StoreOp.Store
        };
        var rPassDesc = new RenderPassDescriptor
        {
            ColorAttachmentCount = 1,
            ColorAttachments = &colorAttach
        };

        RenderPassEncoder* rPass = _ctx.Wgpu.CommandEncoderBeginRenderPass(encoder, &rPassDesc);
        _ctx.Wgpu.RenderPassEncoderSetPipeline(rPass, _blitPipeline);
        _ctx.Wgpu.RenderPassEncoderSetBindGroup(rPass, 0, bgBlit, 0, null);
        _ctx.Wgpu.RenderPassEncoderDraw(rPass, 3, 1, 0, 0);
        _ctx.Wgpu.RenderPassEncoderEnd(rPass);

        _ctx.Wgpu.RenderPassEncoderRelease(rPass);
        _ctx.Wgpu.BindGroupRelease(bgBlit);
    }

    public void Dispose()
    {
        if (_blitPipeline != null) _ctx.Wgpu.RenderPipelineRelease(_blitPipeline);
        _probeDebug.Dispose();
        OverlayBuffer.Dispose();
        _blitUniformBuffer.Dispose();
        _uniformBuffer.Dispose();
    }
}
