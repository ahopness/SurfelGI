using System;
using System.Linq;
using System.Numerics;
using ImGuiNET;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.Scene;

namespace SurfelGI.Debug;

/// <summary>
/// Full UI parameter state matching webgiya's lil-gui controls:
/// Scene, Light, GI, Integrator, Occlusion, and Debug folders.
/// Renders an ImGui.NET overlay via a WebGPU render pipeline.
/// </summary>
public sealed unsafe class ImGuiController : IDisposable
{
    private readonly WebGpuContext _ctx;
    private IntPtr _imguiContext;
    private Texture* _fontTexture;
    private TextureView* _fontTextureView;
    private Sampler* _fontSampler;
    private ShaderModule* _shaderModule;
    private BindGroupLayout* _bindGroupLayout;
    private PipelineLayout* _pipelineLayout;
    private RenderPipeline* _pipeline;
    private BindGroup* _bindGroup;
    private GpuBuffer<ImDrawVert>? _vertexBuffer;
    private GpuBuffer<ushort>? _indexBuffer;
    private GpuBuffer<Matrix4x4>? _uniformBuffer;
    private int _vertexBufferCapacity = 0;
    private int _indexBufferCapacity = 0;

    // Scene state
    public int SelectedSceneIndex = 4; // Sponza (Heavy) default
    public Action<int>? OnSceneSelected;
    public Action? OnResetSurfels;

    // Telemetry
    public float Fps = 0f;
    public float FrameTimeMs = 0f;
    public int TriangleCount = 0;
    public int BvhNodeCount = 0;

    private static readonly string[] SceneLabels = ScenePresets.All.Select(p => p.Label).ToArray();

    private static readonly string[] GiModeLabels =
    {
        "Combined",
        "Direct Only",
        "Indirect Only"
    };

    private static readonly string[] GiSourceLabels =
    {
        "Off (Ambient Only)", "Probe Only", "Surfel Only", "Surfel + Probe"
    };

    private static readonly string[] DebugModeLabels =
    {
        "Off",
        "Surfels",
        "Irradiance",
        "Heatmap",
        "Cascades",
        "Cell Hash",
        "Grid Coords",
        "World Pos",
        "Depth",
        "Normals",
        "Albedo",
        "Coverage",
        "SLG Grid",
        "Spawn / Despawn",
        "Radial Occlusion",
        "Radial Depth Tile",
        "Variance",
        "Albedo BVH",
        "Albedo Match",
        "Probes: Validity",
        "Probes: Irradiance",
        "Probes: Depth",
        "Probes: Relocation",
        "Probes: Fallback Lighting"
    };

    private const string ImGuiWgsl = @"
struct Uniforms {
    mvp : mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var font_tex : texture_2d<f32>;
@group(0) @binding(2) var font_samp : sampler;

struct VsIn {
    @location(0) pos : vec2<f32>,
    @location(1) uv  : vec2<f32>,
    @location(2) col : u32,
}

struct VsOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) col : vec4<f32>,
}

@vertex
fn vs_main(in : VsIn) -> VsOut {
    var out : VsOut;
    out.clip_pos = uniforms.mvp * vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    let r = f32((in.col >>  0u) & 0xFFu) / 255.0;
    let g = f32((in.col >>  8u) & 0xFFu) / 255.0;
    let b = f32((in.col >> 16u) & 0xFFu) / 255.0;
    let a = f32((in.col >> 24u) & 0xFFu) / 255.0;
    out.col = vec4<f32>(r, g, b, a);
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    return in.col * textureSample(font_tex, font_samp, in.uv);
}
";

    public ImGuiController(WebGpuContext ctx)
    {
        _ctx = ctx;
        _imguiContext = ImGui.CreateContext();
        ImGui.SetCurrentContext(_imguiContext);

        var io = ImGui.GetIO();
        io.BackendFlags |= ImGuiBackendFlags.RendererHasVtxOffset;
        ImGui.StyleColorsDark();

        CreateDeviceObjects();
    }

    private void CreateDeviceObjects()
    {
        var io = ImGui.GetIO();
        io.Fonts.GetTexDataAsRGBA32(out byte* pixels, out int width, out int height, out int bytesPerPixel);

        var texDesc = new TextureDescriptor
        {
            Size = new Extent3D((uint)width, (uint)height, 1),
            MipLevelCount = 1,
            SampleCount = 1,
            Dimension = TextureDimension.Dimension2D,
            Format = TextureFormat.Rgba8Unorm,
            Usage = TextureUsage.TextureBinding | TextureUsage.CopyDst
        };
        _fontTexture = _ctx.Wgpu.DeviceCreateTexture(_ctx.Device, in texDesc);

        var viewDesc = new TextureViewDescriptor
        {
            Format = TextureFormat.Rgba8Unorm,
            Dimension = TextureViewDimension.Dimension2D,
            BaseMipLevel = 0,
            MipLevelCount = 1,
            BaseArrayLayer = 0,
            ArrayLayerCount = 1,
            Aspect = TextureAspect.All
        };
        _fontTextureView = _ctx.Wgpu.TextureCreateView(_fontTexture, in viewDesc);

        var copyTex = new ImageCopyTexture
        {
            Texture = _fontTexture,
            MipLevel = 0,
            Origin = new Origin3D(0, 0, 0),
            Aspect = TextureAspect.All
        };
        var layout = new TextureDataLayout
        {
            Offset = 0,
            BytesPerRow = (uint)(width * bytesPerPixel),
            RowsPerImage = (uint)height
        };
        var extent = new Extent3D((uint)width, (uint)height, 1);
        _ctx.Wgpu.QueueWriteTexture(_ctx.Queue, in copyTex, pixels, (nuint)(width * height * bytesPerPixel), in layout, in extent);
        io.Fonts.SetTexID((IntPtr)_fontTextureView);

        var sampDesc = new SamplerDescriptor
        {
            AddressModeU = AddressMode.ClampToEdge,
            AddressModeV = AddressMode.ClampToEdge,
            AddressModeW = AddressMode.ClampToEdge,
            MagFilter = FilterMode.Linear,
            MinFilter = FilterMode.Linear,
            MipmapFilter = MipmapFilterMode.Linear,
            MaxAnisotropy = 1
        };
        _fontSampler = _ctx.Wgpu.DeviceCreateSampler(_ctx.Device, in sampDesc);

        _uniformBuffer = new GpuBuffer<Matrix4x4>(_ctx, 1, BufferUsage.Uniform, "ImGui_Uniforms");

        byte[] wgslBytes = System.Text.Encoding.UTF8.GetBytes(ImGuiWgsl + "\0");
        fixed (byte* pCode = wgslBytes)
        {
            var wgslDesc = new ShaderModuleWGSLDescriptor
            {
                Chain = new ChainedStruct { SType = SType.ShaderSourceWGSL },
                Code = pCode
            };
            var smDesc = new ShaderModuleDescriptor { NextInChain = (ChainedStruct*)&wgslDesc };
            _shaderModule = _ctx.Wgpu.DeviceCreateShaderModule(_ctx.Device, in smDesc);
        }

        var bglEntries = stackalloc BindGroupLayoutEntry[3];
        bglEntries[0] = new BindGroupLayoutEntry
        {
            Binding = 0,
            Visibility = ShaderStage.Vertex,
            Buffer = new BufferBindingLayout { Type = BufferBindingType.Uniform, MinBindingSize = 64 }
        };
        bglEntries[1] = new BindGroupLayoutEntry
        {
            Binding = 1,
            Visibility = ShaderStage.Fragment,
            Texture = new TextureBindingLayout { SampleType = TextureSampleType.Float, ViewDimension = TextureViewDimension.Dimension2D }
        };
        bglEntries[2] = new BindGroupLayoutEntry
        {
            Binding = 2,
            Visibility = ShaderStage.Fragment,
            Sampler = new SamplerBindingLayout { Type = SamplerBindingType.Filtering }
        };

        var bglDesc = new BindGroupLayoutDescriptor { EntryCount = 3, Entries = bglEntries };
        _bindGroupLayout = _ctx.Wgpu.DeviceCreateBindGroupLayout(_ctx.Device, in bglDesc);

        var bglPtr = _bindGroupLayout;
        var plDesc = new PipelineLayoutDescriptor { BindGroupLayoutCount = 1, BindGroupLayouts = &bglPtr };
        _pipelineLayout = _ctx.Wgpu.DeviceCreatePipelineLayout(_ctx.Device, in plDesc);

        var attrs = stackalloc VertexAttribute[3];
        attrs[0] = new VertexAttribute { Format = VertexFormat.Float32x2, Offset = 0, ShaderLocation = 0 };
        attrs[1] = new VertexAttribute { Format = VertexFormat.Float32x2, Offset = 8, ShaderLocation = 1 };
        attrs[2] = new VertexAttribute { Format = VertexFormat.Uint32, Offset = 16, ShaderLocation = 2 };

        var vLayout = new VertexBufferLayout
        {
            ArrayStride = (ulong)sizeof(ImDrawVert),
            StepMode = VertexStepMode.Vertex,
            AttributeCount = 3,
            Attributes = attrs
        };

        var blend = new BlendState
        {
            Color = new BlendComponent
            {
                SrcFactor = BlendFactor.SrcAlpha,
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
            Blend = &blend,
            WriteMask = ColorWriteMask.All
        };

        fixed (byte* pVs = "vs_main\0"u8)
        fixed (byte* pFs = "fs_main\0"u8)
        {
            var fragState = new FragmentState
            {
                Module = _shaderModule,
                EntryPoint = pFs,
                TargetCount = 1,
                Targets = &colorTarget
            };

            var rpDesc = new RenderPipelineDescriptor
            {
                Layout = _pipelineLayout,
                Vertex = new VertexState
                {
                    Module = _shaderModule,
                    EntryPoint = pVs,
                    BufferCount = 1,
                    Buffers = &vLayout
                },
                Primitive = new PrimitiveState
                {
                    Topology = PrimitiveTopology.TriangleList,
                    FrontFace = FrontFace.Ccw,
                    CullMode = CullMode.None
                },
                Multisample = new MultisampleState { Count = 1, Mask = ~0u },
                Fragment = &fragState
            };
            _pipeline = _ctx.Wgpu.DeviceCreateRenderPipeline(_ctx.Device, in rpDesc);
        }

        var bgEntries = stackalloc BindGroupEntry[3];
        bgEntries[0] = _uniformBuffer.AsBindGroupEntry(0);
        bgEntries[1] = new BindGroupEntry { Binding = 1, TextureView = _fontTextureView };
        bgEntries[2] = new BindGroupEntry { Binding = 2, Sampler = _fontSampler };

        var bgDesc = new BindGroupDescriptor
        {
            Layout = _bindGroupLayout,
            EntryCount = 3,
            Entries = bgEntries
        };
        _bindGroup = _ctx.Wgpu.DeviceCreateBindGroup(_ctx.Device, in bgDesc);
    }

    public bool WantsCaptureMouse => ImGui.GetIO().WantCaptureMouse;
    public bool WantsCaptureKeyboard => ImGui.GetIO().WantCaptureKeyboard;

    public void UpdateInput(float dt, int width, int height, Vector2 mousePos, bool leftDown, bool rightDown, float wheelDelta)
    {
        ImGui.SetCurrentContext(_imguiContext);
        var io = ImGui.GetIO();
        io.DisplaySize = new Vector2(width, height);
        io.DeltaTime = dt > 0f ? dt : (1f / 60f);
        io.MousePos = mousePos;
        io.MouseDown[0] = leftDown;
        io.MouseDown[1] = rightDown;
        io.MouseWheel += wheelDelta;
    }

    public void BuildUi(
        LightingController lighting,
        GiTransportSettings gi,
        OcclusionSettings occlusion,
        SurfelScreenDebugPass debugPass)
    {
        ImGui.SetCurrentContext(_imguiContext);
        ImGui.NewFrame();

        ImGui.SetNextWindowPos(new Vector2(12, 12), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowSize(new Vector2(360, 680), ImGuiCond.FirstUseEver);

        if (ImGui.Begin("Surfel GI Controls"))
        {
            ImGui.Text($"FPS: {Fps:F1} ({FrameTimeMs:F2} ms)");
            ImGui.Text($"Triangles: {TriangleCount:N0} | BVH Nodes: {BvhNodeCount:N0}");
            ImGui.Separator();

            // 1. Scene Folder
            if (ImGui.CollapsingHeader("Scene", ImGuiTreeNodeFlags.DefaultOpen))
            {
                if (ImGui.Combo("Preset", ref SelectedSceneIndex, SceneLabels, SceneLabels.Length))
                {
                    OnSceneSelected?.Invoke(SelectedSceneIndex);
                }
                if (ImGui.Button("Reset Surfels"))
                {
                    OnResetSurfels?.Invoke();
                }
            }

            // 2. Light Folder
            if (ImGui.CollapsingHeader("Light", ImGuiTreeNodeFlags.DefaultOpen))
            {
                bool changed = false;
                changed |= ImGui.SliderFloat("Sun Azimuth", ref lighting.Config.AzimuthDeg, -180f, 180f);
                changed |= ImGui.SliderFloat("Sun Elevation", ref lighting.Config.ElevationDeg, 5f, 89f);
                changed |= ImGui.SliderFloat("Sun Intensity", ref lighting.Config.Intensity, 0f, 30f);
                ImGui.Checkbox("Animate Sun", ref lighting.Config.Animate);
                ImGui.SliderFloat("Sun Speed", ref lighting.Config.Speed, 0.05f, 3f);
                if (changed)
                    lighting.UpdateLightFromAngles();
            }

            // 3. GI Folder
            if (ImGui.CollapsingHeader("GI", ImGuiTreeNodeFlags.DefaultOpen))
            {
                int giMode = (int)gi.Mode;
                if (ImGui.Combo("Output Mode", ref giMode, GiModeLabels, GiModeLabels.Length))
                    gi.Mode = (GiOutputMode)giMode;

                ImGui.Checkbox("Half resolution GI", ref gi.HalfResolutionGi);
                ImGui.Checkbox("Stagger settled updates", ref gi.StaggerUpdates);
                ImGui.SliderFloat("Exposure", ref gi.Exposure, 0.05f, 8f);
                ImGui.SliderFloat("Indirect Scale", ref gi.IndirectIntensity, 0f, 5f);
                ImGui.SliderFloat("GI From Direct", ref gi.GiFromDirect, 0f, 5f);
                ImGui.SliderFloat("GI From Indirect", ref gi.GiFromIndirect, 0f, 5f);
                ImGui.SliderFloat("Albedo Boost", ref gi.AlbedoBoost, 0.1f, 3f);
            }

            // 4. Integrator Folder
            if (ImGui.CollapsingHeader("Integrator", ImGuiTreeNodeFlags.DefaultOpen))
            {
                ImGui.SliderInt("Base Ray Count", ref gi.BaseSampleCount, 1, 32);
                ImGui.SliderFloat("Env Intensity", ref gi.EnvIntensity, 0f, 5f);
                ImGui.SliderFloat("Env Mip LOD", ref gi.EnvLod, 0f, 8f);
            }

            // 5. Occlusion Folder
            if (ImGui.CollapsingHeader("Occlusion", ImGuiTreeNodeFlags.DefaultOpen))
            {
                ImGui.SliderFloat("Shadow Strength", ref occlusion.ShadowStrength, 0f, 3f);
                ImGui.SliderFloat("Bleed Reduction", ref occlusion.BleedReduction, 0f, 1f);
                ImGui.SliderFloat("Grazing Bias", ref occlusion.GrazingBiasScale, 0f, 1f);
                ImGui.SliderFloat("Variance Bleed", ref occlusion.VarianceBleedScale, 0f, 2f);
            }

            // 6. Debug Folder
            if (ImGui.CollapsingHeader("Debug", ImGuiTreeNodeFlags.DefaultOpen))
            {
                int sourceMode = (int)gi.SourceMode;
                if (ImGui.Combo("GI Mode", ref sourceMode, GiSourceLabels, GiSourceLabels.Length))
                    gi.SourceMode = (GiSourceMode)sourceMode;

                int dbgMode = (int)debugPass.Mode;
                if (ImGui.Combo("Debug Mode", ref dbgMode, DebugModeLabels, DebugModeLabels.Length))
                    debugPass.Mode = (ScreenDebugMode)dbgMode;

                if (debugPass.Mode >= ScreenDebugMode.ProbeValidity) {
                    if (debugPass.Mode != ScreenDebugMode.ProbeFallbackLighting) {
                        int level = debugPass.ProbeLevel;
                        if (ImGui.SliderInt("Probe Level", ref level, 0, 4)) debugPass.ProbeLevel = level;
                        ImGui.Text($"Spacing: {1 << level} world units");
                        float radius = debugPass.ProbeRadius;
                        if (ImGui.SliderFloat("Probe Radius", ref radius, 0.02f, 0.25f)) debugPass.ProbeRadius = radius;
                        bool xray = debugPass.ProbeXray;
                        if (ImGui.Checkbox("Probe X-Ray", ref xray)) debugPass.ProbeXray = xray;
                    }
                    float exposure = debugPass.ProbeExposure;
                    if (ImGui.SliderFloat("Probe Exposure", ref exposure, 0.1f, 10f)) debugPass.ProbeExposure = exposure;
                    if (debugPass.Mode == ScreenDebugMode.ProbeValidity) ImGui.TextWrapped("Red: invalid. Amber: newly initialized. Green: accumulated updates.");
                    if (debugPass.Mode == ScreenDebugMode.ProbeDepth) ImGui.TextWrapped("Directional mean distance: black = nearby, white = 8 probe spacings.");
                    if (debugPass.Mode == ScreenDebugMode.ProbeRelocation) ImGui.TextWrapped("RGB = absolute X/Y/Z offset. Black = unmoved. Red also marks invalid unmoved probes.");
                    if (debugPass.Mode == ScreenDebugMode.ProbeFallbackLighting) ImGui.TextWrapped("Probe estimate on surfaces, before albedo and surfel blending. Magenta = no usable probes. Uses automatic level selection.");
                }

                float varScale = debugPass.VarianceScale;
                if (ImGui.SliderFloat("Variance Scale", ref varScale, 0.1f, 10f))
                    debugPass.VarianceScale = varScale;
            }
        }
        ImGui.End();
        ImGui.Render();
    }

    public void Render(CommandEncoder* encoder, TextureView* targetView, int width, int height)
    {
        var drawData = ImGui.GetDrawData();
        if (drawData.CmdListsCount == 0 || width <= 0 || height <= 0)
            return;

        int totalVtx = drawData.TotalVtxCount;
        int totalIdx = drawData.TotalIdxCount;
        if (totalVtx == 0 || totalIdx == 0)
            return;

        if (_vertexBuffer == null || totalVtx > _vertexBufferCapacity)
        {
            _vertexBuffer?.Dispose();
            _vertexBufferCapacity = Math.Max(totalVtx * 2, 4096);
            _vertexBuffer = new GpuBuffer<ImDrawVert>(_ctx, _vertexBufferCapacity, BufferUsage.Vertex, "ImGui_VB");
        }

        if (_indexBuffer == null || totalIdx + 2 > _indexBufferCapacity)
        {
            _indexBuffer?.Dispose();
            _indexBufferCapacity = (Math.Max(totalIdx * 2, 8192) + 1) & ~1;
            _indexBuffer = new GpuBuffer<ushort>(_ctx, _indexBufferCapacity, BufferUsage.Index, "ImGui_IB");
        }

        int alignedIdxCount = (totalIdx + 1) & ~1;
        ushort[] idxStaging = new ushort[alignedIdxCount];
        ulong vtxOffset = 0;
        int idxCursor = 0;
        for (int n = 0; n < drawData.CmdListsCount; n++)
        {
            var cmdList = drawData.CmdLists[n];
            ulong vBytes = (ulong)(cmdList.VtxBuffer.Size * sizeof(ImDrawVert));
            _ctx.Wgpu.QueueWriteBuffer(_ctx.Queue, _vertexBuffer.Handle, vtxOffset, (void*)cmdList.VtxBuffer.Data, (nuint)vBytes);
            vtxOffset += vBytes;

            new ReadOnlySpan<ushort>((void*)cmdList.IdxBuffer.Data, cmdList.IdxBuffer.Size)
                .CopyTo(idxStaging.AsSpan(idxCursor, cmdList.IdxBuffer.Size));
            idxCursor += cmdList.IdxBuffer.Size;
        }
        fixed (ushort* pIdx = idxStaging)
        {
            _ctx.Wgpu.QueueWriteBuffer(_ctx.Queue, _indexBuffer.Handle, 0, pIdx, (nuint)(alignedIdxCount * sizeof(ushort)));
        }

        float L = drawData.DisplayPos.X;
        float R = drawData.DisplayPos.X + drawData.DisplaySize.X;
        float T = drawData.DisplayPos.Y;
        float B = drawData.DisplayPos.Y + drawData.DisplaySize.Y;
        Matrix4x4 mvp = new(
            2.0f / (R - L), 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f / (T - B), 0.0f, 0.0f,
            0.0f, 0.0f, 0.5f, 0.0f,
            (R + L) / (L - R), (T + B) / (B - T), 0.5f, 1.0f
        );
        _uniformBuffer!.UploadSingle(in mvp);

        var colorAttach = new RenderPassColorAttachment
        {
            View = targetView,
            LoadOp = LoadOp.Load,
            StoreOp = StoreOp.Store
        };
        var rpDesc = new RenderPassDescriptor
        {
            ColorAttachmentCount = 1,
            ColorAttachments = &colorAttach
        };

        var pass = _ctx.Wgpu.CommandEncoderBeginRenderPass(encoder, in rpDesc);
        _ctx.Wgpu.RenderPassEncoderSetPipeline(pass, _pipeline);
        _ctx.Wgpu.RenderPassEncoderSetBindGroup(pass, 0, _bindGroup, 0, null);
        _ctx.Wgpu.RenderPassEncoderSetVertexBuffer(pass, 0, _vertexBuffer.Handle, 0, _vertexBuffer.SizeInBytes);
        _ctx.Wgpu.RenderPassEncoderSetIndexBuffer(pass, _indexBuffer.Handle, IndexFormat.Uint16, 0, _indexBuffer.SizeInBytes);
        _ctx.Wgpu.RenderPassEncoderSetViewport(pass, 0, 0, width, height, 0, 1);

        int globalVtxOffset = 0;
        int globalIdxOffset = 0;
        Vector2 clipOff = drawData.DisplayPos;

        for (int n = 0; n < drawData.CmdListsCount; n++)
        {
            var cmdList = drawData.CmdLists[n];
            for (int i = 0; i < cmdList.CmdBuffer.Size; i++)
            {
                var pcmd = cmdList.CmdBuffer[i];
                var clipMin = new Vector2(Math.Max(0, pcmd.ClipRect.X - clipOff.X), Math.Max(0, pcmd.ClipRect.Y - clipOff.Y));
                var clipMax = new Vector2(Math.Min(width, pcmd.ClipRect.Z - clipOff.X), Math.Min(height, pcmd.ClipRect.W - clipOff.Y));
                if (clipMax.X <= clipMin.X || clipMax.Y <= clipMin.Y)
                    continue;

                _ctx.Wgpu.RenderPassEncoderSetScissorRect(pass,
                    (uint)clipMin.X, (uint)clipMin.Y,
                    (uint)(clipMax.X - clipMin.X), (uint)(clipMax.Y - clipMin.Y));
                _ctx.Wgpu.RenderPassEncoderDrawIndexed(pass, pcmd.ElemCount, 1, pcmd.IdxOffset + (uint)globalIdxOffset, (int)pcmd.VtxOffset + globalVtxOffset, 0);
            }
            globalIdxOffset += cmdList.IdxBuffer.Size;
            globalVtxOffset += cmdList.VtxBuffer.Size;
        }

        _ctx.Wgpu.RenderPassEncoderEnd(pass);
        _ctx.Wgpu.RenderPassEncoderRelease(pass);
    }

    public void Dispose()
    {
        _vertexBuffer?.Dispose();
        _indexBuffer?.Dispose();
        _uniformBuffer?.Dispose();
        if (_bindGroup != null) _ctx.Wgpu.BindGroupRelease(_bindGroup);
        if (_pipeline != null) _ctx.Wgpu.RenderPipelineRelease(_pipeline);
        if (_pipelineLayout != null) _ctx.Wgpu.PipelineLayoutRelease(_pipelineLayout);
        if (_bindGroupLayout != null) _ctx.Wgpu.BindGroupLayoutRelease(_bindGroupLayout);
        if (_shaderModule != null) _ctx.Wgpu.ShaderModuleRelease(_shaderModule);
        if (_fontSampler != null) _ctx.Wgpu.SamplerRelease(_fontSampler);
        if (_fontTextureView != null) _ctx.Wgpu.TextureViewRelease(_fontTextureView);
        if (_fontTexture != null) _ctx.Wgpu.TextureDestroy(_fontTexture);
        if (_imguiContext != IntPtr.Zero) ImGui.DestroyContext(_imguiContext);
    }
}
