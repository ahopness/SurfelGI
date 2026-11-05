using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Core.Native;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.Scene;

namespace SurfelGI.GiPipeline;

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct GBufferUniforms
{
    public Matrix4x4 ViewProj;
    public Matrix4x4 Model;
    public Matrix4x4 NormalMat;
    public Vector4 BaseColor;
    public uint HasTexture;
    public float Roughness;
    public float Metalness;
    public float Pad0;
}

public unsafe class GBufferPass : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;

    public GpuTexture NormalTexture { get; private set; }
    public GpuTexture AlbedoTexture { get; private set; }
    public GpuTexture DepthTexture { get; private set; }
    public GpuTexture WhiteFallbackTexture { get; }

    private RenderPipeline* _pipelineCullBack;
    private RenderPipeline* _pipelineDoubleSided;

    private readonly List<GpuBuffer<GBufferUniforms>> _uniformPool = new();

    public GBufferPass(WebGpuContext ctx, ShaderManager shaders)
    {
        _ctx = ctx;
        _shaders = shaders;

        NormalTexture = new GpuTexture(
            ctx, ctx.Width, ctx.Height,
            TextureFormat.Rgba16Float,
            TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
            filter: FilterMode.Nearest,
            label: "GBuffer_Normal");

        AlbedoTexture = new GpuTexture(
            ctx, ctx.Width, ctx.Height,
            TextureFormat.Rgba16Float,
            TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
            filter: FilterMode.Nearest,
            label: "GBuffer_Albedo");

        DepthTexture = new GpuTexture(
            ctx, ctx.Width, ctx.Height,
            TextureFormat.Depth32Float,
            TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
            filter: FilterMode.Nearest,
            label: "GBuffer_Depth");

        WhiteFallbackTexture = new GpuTexture(
            ctx, 1, 1,
            TextureFormat.Rgba8UnormSrgb,
            TextureUsage.TextureBinding | TextureUsage.CopyDst,
            filter: FilterMode.Linear,
            addressMode: AddressMode.Repeat,
            label: "WhiteFallback");
        WhiteFallbackTexture.UploadLayer<byte>(new byte[] { 255, 255, 255, 255 }, 4);

        CreatePipelines();
    }

    public int Width => NormalTexture.Width;
    public int Height => NormalTexture.Height;

    public void Resize(int width, int height)
    {
        NormalTexture.Resize(width, height);
        AlbedoTexture.Resize(width, height);
        DepthTexture.Resize(width, height);
    }

    private void CreatePipelines()
    {
        ShaderModule* module = _shaders.GetOrCreateModule("gbuffer.wgsl");
        _pipelineCullBack = CreatePipelineVariant(module, CullMode.Back, "GBuffer_CullBack");
        _pipelineDoubleSided = CreatePipelineVariant(module, CullMode.None, "GBuffer_DoubleSided");
    }

    private RenderPipeline* CreatePipelineVariant(ShaderModule* module, CullMode cullMode, string label)
    {
        VertexAttribute* attrs = stackalloc VertexAttribute[3];
        attrs[0] = new VertexAttribute { Format = VertexFormat.Float32x3, Offset = 0, ShaderLocation = 0 };
        attrs[1] = new VertexAttribute { Format = VertexFormat.Float32x3, Offset = 12, ShaderLocation = 1 };
        attrs[2] = new VertexAttribute { Format = VertexFormat.Float32x2, Offset = 24, ShaderLocation = 2 };

        var vbLayout = new VertexBufferLayout
        {
            ArrayStride = (ulong)sizeof(MeshVertex),
            StepMode = VertexStepMode.Vertex,
            AttributeCount = 3,
            Attributes = attrs
        };

        ColorTargetState* targets = stackalloc ColorTargetState[2];
        targets[0] = new ColorTargetState { Format = TextureFormat.Rgba16Float, WriteMask = ColorWriteMask.All };
        targets[1] = new ColorTargetState { Format = TextureFormat.Rgba16Float, WriteMask = ColorWriteMask.All };

        byte* vsEntry = (byte*)SilkMarshal.StringToPtr("vs_main");
        byte* fsEntry = (byte*)SilkMarshal.StringToPtr("fs_main");
        byte* labelPtr = (byte*)SilkMarshal.StringToPtr(label);

        try
        {
            var fragState = new FragmentState
            {
                Module = module,
                EntryPoint = fsEntry,
                TargetCount = 2,
                Targets = targets
            };

            var depthStencil = new DepthStencilState
            {
                Format = TextureFormat.Depth32Float,
                DepthWriteEnabled = true,
                DepthCompare = CompareFunction.LessEqual
            };

            var desc = new RenderPipelineDescriptor
            {
                Label = labelPtr,
                Vertex = new VertexState
                {
                    Module = module,
                    EntryPoint = vsEntry,
                    BufferCount = 1,
                    Buffers = &vbLayout
                },
                Primitive = new PrimitiveState
                {
                    Topology = PrimitiveTopology.TriangleList,
                    FrontFace = FrontFace.Ccw,
                    CullMode = cullMode
                },
                DepthStencil = &depthStencil,
                Multisample = new MultisampleState
                {
                    Count = 1,
                    Mask = ~0u
                },
                Fragment = &fragState
            };

            return _ctx.Wgpu.DeviceCreateRenderPipeline(_ctx.Device, &desc);
        }
        finally
        {
            SilkMarshal.Free((nint)vsEntry);
            SilkMarshal.Free((nint)fsEntry);
            SilkMarshal.Free((nint)labelPtr);
        }
    }

    private GpuBuffer<GBufferUniforms> GetUniformBuffer(int index)
    {
        while (_uniformPool.Count <= index)
        {
            _uniformPool.Add(new GpuBuffer<GBufferUniforms>(_ctx, 1, BufferUsage.Uniform, $"GBuf_UB_{_uniformPool.Count}"));
        }
        return _uniformPool[index];
    }

    private GpuTexture EnsureMaterialTexture(MaterialData mat)
    {
        if (mat.DiffuseMap == null || mat.DiffuseMap.Rgba8Data == null)
            return WhiteFallbackTexture;

        if (mat.GpuDiffuseTexture == null)
        {
            mat.GpuDiffuseTexture = new GpuTexture(
                _ctx,
                mat.DiffuseMap.Width,
                mat.DiffuseMap.Height,
                TextureFormat.Rgba8UnormSrgb,
                TextureUsage.TextureBinding | TextureUsage.CopyDst,
                filter: FilterMode.Linear,
                addressMode: AddressMode.Repeat,
                label: mat.DiffuseMap.Name);
            mat.GpuDiffuseTexture.UploadLayer<byte>(mat.DiffuseMap.Rgba8Data, 4);
        }
        return mat.GpuDiffuseTexture;
    }

    public void Execute(CommandEncoder* encoder, SceneGraph scene, Camera camera)
    {
        RenderPassColorAttachment* colorAttachments = stackalloc RenderPassColorAttachment[2];
        colorAttachments[0] = new RenderPassColorAttachment
        {
            View = NormalTexture.View,
            LoadOp = LoadOp.Clear,
            StoreOp = StoreOp.Store,
            ClearValue = new Color(0.5, 0.5, 1.0, 1.0)
        };
        colorAttachments[1] = new RenderPassColorAttachment
        {
            View = AlbedoTexture.View,
            LoadOp = LoadOp.Clear,
            StoreOp = StoreOp.Store,
            ClearValue = new Color(0.0, 0.0, 0.0, 1.0)
        };

        var depthAttachment = new RenderPassDepthStencilAttachment
        {
            View = DepthTexture.View,
            DepthLoadOp = LoadOp.Clear,
            DepthStoreOp = StoreOp.Store,
            DepthClearValue = 1.0f
        };

        var passDesc = new RenderPassDescriptor
        {
            ColorAttachmentCount = 2,
            ColorAttachments = colorAttachments,
            DepthStencilAttachment = &depthAttachment
        };

        RenderPassEncoder* pass = _ctx.Wgpu.CommandEncoderBeginRenderPass(encoder, &passDesc);

        int drawIndex = 0;
        var tempBindGroups = new List<IntPtr>();
        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[3];

        foreach (var inst in scene.Meshes)
        {
            if (!inst.Visible || inst.GpuBuffers == null)
                continue;

            Matrix4x4.Invert(inst.WorldMatrix, out Matrix4x4 invWorld);
            Matrix4x4 normalMat = Matrix4x4.Transpose(invWorld);

            _ctx.Wgpu.RenderPassEncoderSetVertexBuffer(pass, 0, inst.GpuBuffers.VertexBuffer.Handle, 0, inst.GpuBuffers.VertexBuffer.SizeInBytes);
            _ctx.Wgpu.RenderPassEncoderSetIndexBuffer(pass, inst.GpuBuffers.IndexBuffer.Handle, IndexFormat.Uint32, 0, inst.GpuBuffers.IndexBuffer.SizeInBytes);

            foreach (var grp in inst.Mesh.Groups)
            {
                if (grp.IndexCount <= 0) continue;
                var mat = grp.MaterialIndex < inst.Materials.Count ? inst.Materials[grp.MaterialIndex] : new MaterialData();
                var tex = EnsureMaterialTexture(mat);

                var ub = GetUniformBuffer(drawIndex++);
                var uData = new GBufferUniforms
                {
                    ViewProj = camera.ViewProjectionMatrix,
                    Model = inst.WorldMatrix,
                    NormalMat = normalMat,
                    BaseColor = new Vector4(mat.BaseColorLinear, 1.0f),
                    HasTexture = mat.DiffuseMap != null ? 1u : 0u,
                    Roughness = mat.Roughness,
                    Metalness = mat.Metalness
                };
                ub.UploadSingle(in uData);

                RenderPipeline* pipe = mat.DoubleSided ? _pipelineDoubleSided : _pipelineCullBack;
                _ctx.Wgpu.RenderPassEncoderSetPipeline(pass, pipe);

                entries[0] = ub.AsBindGroupEntry(0);
                entries[1] = tex.AsTextureEntry(1);
                entries[2] = tex.AsSamplerEntry(2);
                BindGroup* bg = _shaders.CreateBindGroup(pipe, 0, entries, "GBuf_DrawBG");
                tempBindGroups.Add((IntPtr)bg);

                _ctx.Wgpu.RenderPassEncoderSetBindGroup(pass, 0, bg, 0, null);
                _ctx.Wgpu.RenderPassEncoderDrawIndexed(pass, (uint)grp.IndexCount, 1, (uint)grp.StartIndex, 0, 0);
            }
        }

        _ctx.Wgpu.RenderPassEncoderEnd(pass);
        _ctx.Wgpu.RenderPassEncoderRelease(pass);

        foreach (IntPtr bg in tempBindGroups)
            _ctx.Wgpu.BindGroupRelease((BindGroup*)bg);
    }

    public void Dispose()
    {
        foreach (var ub in _uniformPool) ub.Dispose();
        _uniformPool.Clear();
        if (_pipelineCullBack != null) _ctx.Wgpu.RenderPipelineRelease(_pipelineCullBack);
        if (_pipelineDoubleSided != null) _ctx.Wgpu.RenderPipelineRelease(_pipelineDoubleSided);
        WhiteFallbackTexture.Dispose();
        DepthTexture.Dispose();
        AlbedoTexture.Dispose();
        NormalTexture.Dispose();
    }
}
