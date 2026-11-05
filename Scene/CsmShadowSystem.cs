using System.Numerics;
using Silk.NET.Core.Native;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.GiPipeline;

namespace SurfelGI.Scene;

public unsafe class CsmShadowSystem : IDisposable
{
    public const int CascadeCount = 4;

    private readonly WebGpuContext _ctx;
    private readonly ShaderManager _shaders;
    public GpuTexture ShadowMapArray { get; private set; }
    private RenderPipeline* _shadowPipeline;
    private readonly List<GpuBuffer<GBufferUniforms>> _uniformPool = new();
    private readonly Dictionary<CpuMesh, (Vector3 Min, Vector3 Max)> _meshBounds = new();
    private (float Near, float Far) CasterDepth(SceneGraph scene, Matrix4x4 lightView)
    {
        float minZ = float.PositiveInfinity, maxZ = float.NegativeInfinity;
        foreach (var inst in scene.Meshes) {
            if (!inst.Visible || !inst.CastShadow) continue;
            if (!_meshBounds.TryGetValue(inst.Mesh, out var bounds)) {
                Vector3 lo = new(float.PositiveInfinity), hi = new(float.NegativeInfinity);
                foreach (var v in inst.Mesh.Vertices) { lo = Vector3.Min(lo, v.Position); hi = Vector3.Max(hi, v.Position); }
                bounds = (lo, hi); _meshBounds[inst.Mesh] = bounds;
            }
            var transform = inst.WorldMatrix * lightView;
            for (int i=0;i<8;i++) {
                Vector3 corner = new((i&1)==0?bounds.Min.X:bounds.Max.X, (i&2)==0?bounds.Min.Y:bounds.Max.Y, (i&4)==0?bounds.Min.Z:bounds.Max.Z);
                float z = Vector3.Transform(corner, transform).Z;
                minZ = MathF.Min(minZ,z); maxZ = MathF.Max(maxZ,z);
            }
        }
        return float.IsFinite(minZ) ? (-maxZ - 1f, MathF.Max(-maxZ, -minZ) + 1f) : (0.1f,150f);
    }

    public Matrix4x4[] CascadeViewProj { get; } = new Matrix4x4[CascadeCount];
    public Vector4 CascadeSplits { get; private set; }

    public CsmShadowSystem(WebGpuContext ctx, ShaderManager shaders, int mapSize = 2048)
    {
        _ctx = ctx;
        _shaders = shaders;

        ShadowMapArray = new GpuTexture(
            ctx,
            mapSize,
            mapSize,
            TextureFormat.Depth32Float,
            TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
            depthOrArrayLayers: CascadeCount,
            viewDimension: TextureViewDimension.Dimension2DArray,
            filter: FilterMode.Nearest,
            label: "CSM_ShadowMapArray");

        CreatePipeline();
    }

    private void CreatePipeline()
    {
        ShaderModule* module = _shaders.GetOrCreateModule("gbuffer.wgsl");

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

        var depthStencil = new DepthStencilState
        {
            Format = TextureFormat.Depth32Float,
            DepthWriteEnabled = true,
            DepthCompare = CompareFunction.LessEqual,
            DepthBias = 2,
            DepthBiasSlopeScale = 2.0f,
            DepthBiasClamp = 0.0f
        };

        byte* vsEntry = (byte*)SilkMarshal.StringToPtr("vs_shadow");
        byte* labelPtr = (byte*)SilkMarshal.StringToPtr("ShadowDepthPipeline");
        try
        {
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
                    CullMode = CullMode.None
                },
                DepthStencil = &depthStencil,
                Multisample = new MultisampleState
                {
                    Count = 1,
                    Mask = ~0u
                },
                Fragment = null
            };
            _shadowPipeline = _ctx.Wgpu.DeviceCreateRenderPipeline(_ctx.Device, &desc);
        }
        finally
        {
            SilkMarshal.Free((nint)vsEntry);
            SilkMarshal.Free((nint)labelPtr);
        }
    }

    private GpuBuffer<GBufferUniforms> GetUniformBuffer(int idx)
    {
        while (_uniformPool.Count <= idx)
            _uniformPool.Add(new GpuBuffer<GBufferUniforms>(_ctx, 1, BufferUsage.Uniform, $"CSM_UB_{_uniformPool.Count}"));
        return _uniformPool[idx];
    }

    public void UpdateCascades(Camera camera, DirectionalLightData light, SceneGraph scene)
    {
        Vector3 lightDir = light.GetLightDirectionToSource();
        Vector3 up = MathF.Abs(lightDir.Y) > 0.99f ? Vector3.UnitZ : Vector3.UnitY;

        if (!light.UseCsm)
        {
            float ext = light.OrthoExtent;
            Matrix4x4 lightView = Matrix4x4.CreateLookAt(light.Target + lightDir * 50f, light.Target, up);
            var depth = CasterDepth(scene, lightView);
            Matrix4x4 lightProj = Camera.CreateOrthographicOffCenterZeroToOne(-ext, ext, -ext, ext, depth.Near, depth.Far);
            Matrix4x4 vp = lightView * lightProj;
            for (int c = 0; c < CascadeCount; c++)
                CascadeViewProj[c] = vp;
            CascadeSplits = new Vector4(1e20f);
            return;
        }

        float near = camera.Near;
        float maxFar = MathF.Max(near + 1f, light.CsmMaxFar);
        const float lambda = 0.5f;

        Span<float> splits = stackalloc float[CascadeCount + 1];
        splits[0] = near;
        for (int i = 1; i <= CascadeCount; i++)
        {
            float p = (float)i / CascadeCount;
            float logSplit = near * MathF.Pow(maxFar / near, p);
            float uniSplit = near + (maxFar - near) * p;
            splits[i] = lambda * logSplit + (1f - lambda) * uniSplit;
        }
        CascadeSplits = new Vector4(splits[1], splits[2], splits[3], splits[4]);

        float fovRad = camera.FovDegrees * (MathF.PI / 180f);
        float tanHalfFovY = MathF.Tan(fovRad * 0.5f);
        float tanHalfFovX = tanHalfFovY * camera.Aspect;
        Span<Vector3> frustumCorners = stackalloc Vector3[8];

        for (int c = 0; c < CascadeCount; c++)
        {
            float cNear = c == 0 ? splits[c] : splits[c] - (splits[c] - splits[c - 1]) * 0.1f;
            float cFar = splits[c + 1];

            frustumCorners[0] = new(-tanHalfFovX * cNear,  tanHalfFovY * cNear, -cNear);
            frustumCorners[1] = new( tanHalfFovX * cNear,  tanHalfFovY * cNear, -cNear);
            frustumCorners[2] = new( tanHalfFovX * cNear, -tanHalfFovY * cNear, -cNear);
            frustumCorners[3] = new(-tanHalfFovX * cNear, -tanHalfFovY * cNear, -cNear);
            frustumCorners[4] = new(-tanHalfFovX * cFar,   tanHalfFovY * cFar,  -cFar);
            frustumCorners[5] = new( tanHalfFovX * cFar,   tanHalfFovY * cFar,  -cFar);
            frustumCorners[6] = new( tanHalfFovX * cFar,  -tanHalfFovY * cFar,  -cFar);
            frustumCorners[7] = new(-tanHalfFovX * cFar,  -tanHalfFovY * cFar,  -cFar);

            Vector3 center = Vector3.Zero;
            for (int i = 0; i < 8; i++)
            {
                frustumCorners[i] = Vector3.Transform(frustumCorners[i], camera.CameraWorldMatrix);
                center += frustumCorners[i];
            }
            center /= 8f;

            float radius = 0f;
            for (int i = 0; i < 8; i++)
                radius = MathF.Max(radius, (frustumCorners[i] - center).Length());
            radius = MathF.Ceiling(radius * 16f) / 16f;

            float margin = MathF.Max(30f, light.CsmLightMargin);
            Matrix4x4 lightView = Matrix4x4.CreateLookAt(center + lightDir * (radius + margin), center, up);
            var depth = CasterDepth(scene, lightView);
            Matrix4x4 lightProj = Camera.CreateOrthographicOffCenterZeroToOne(
                -radius, radius, -radius, radius, depth.Near, depth.Far);

            // Snap the projection translation in light space to a shadow texel.
            Matrix4x4 vp = lightView * lightProj;
            Vector3 origin = Vector3.Transform(Vector3.Zero, vp);
            float halfMap = ShadowMapArray.Width * 0.5f;
            vp.M41 += (MathF.Round(origin.X * halfMap) - origin.X * halfMap) / halfMap;
            vp.M42 += (MathF.Round(origin.Y * halfMap) - origin.Y * halfMap) / halfMap;
            CascadeViewProj[c] = vp;
        }
    }

    public void Execute(CommandEncoder* encoder, SceneGraph scene, Camera camera)
    {
        int requestedSize = Math.Clamp(scene.DirLight.ShadowMapSize, 512, 4096);
        int requestedLayers = scene.DirLight.UseCsm ? CascadeCount : 1;
        if (ShadowMapArray.DepthOrArrayLayers != requestedLayers) {
            ShadowMapArray.Dispose();
            ShadowMapArray = new GpuTexture(_ctx, requestedSize, requestedSize, TextureFormat.Depth32Float,
                TextureUsage.RenderAttachment | TextureUsage.TextureBinding, depthOrArrayLayers: requestedLayers,
                viewDimension: TextureViewDimension.Dimension2DArray, filter: FilterMode.Nearest, label: "CSM_ShadowMapArray");
        } else if (ShadowMapArray.Width != requestedSize) ShadowMapArray.Resize(requestedSize, requestedSize);
        UpdateCascades(camera, scene.DirLight, scene);

        int activeCascades = scene.DirLight.UseCsm ? CascadeCount : 1;
        int drawIdx = 0;
        var tempBindGroups = new List<IntPtr>();
        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[1];

        for (uint c = 0; c < (uint)activeCascades; c++)
        {
            TextureView* layerView = ShadowMapArray.CreateLayerView(c);
            var depthAttach = new RenderPassDepthStencilAttachment
            {
                View = layerView,
                DepthLoadOp = LoadOp.Clear,
                DepthStoreOp = StoreOp.Store,
                DepthClearValue = 1.0f
            };
            var passDesc = new RenderPassDescriptor
            {
                ColorAttachmentCount = 0,
                ColorAttachments = null,
                DepthStencilAttachment = &depthAttach
            };

            RenderPassEncoder* pass = _ctx.Wgpu.CommandEncoderBeginRenderPass(encoder, &passDesc);
            _ctx.Wgpu.RenderPassEncoderSetPipeline(pass, _shadowPipeline);

            foreach (var inst in scene.Meshes)
            {
                if (!inst.Visible || !inst.CastShadow || inst.GpuBuffers == null)
                    continue;

                var ub = GetUniformBuffer(drawIdx++);
                var uData = new GBufferUniforms
                {
                    ViewProj = CascadeViewProj[c],
                    Model = inst.WorldMatrix,
                    NormalMat = Matrix4x4.Identity,
                    BaseColor = Vector4.One
                };
                ub.UploadSingle(in uData);

                entries[0] = ub.AsBindGroupEntry(0);
                BindGroup* bg = _shaders.CreateBindGroup(_shadowPipeline, 0, entries, "ShadowDrawBG");
                tempBindGroups.Add((IntPtr)bg);

                _ctx.Wgpu.RenderPassEncoderSetBindGroup(pass, 0, bg, 0, null);
                _ctx.Wgpu.RenderPassEncoderSetVertexBuffer(pass, 0, inst.GpuBuffers.VertexBuffer.Handle, 0, inst.GpuBuffers.VertexBuffer.SizeInBytes);
                _ctx.Wgpu.RenderPassEncoderSetIndexBuffer(pass, inst.GpuBuffers.IndexBuffer.Handle, IndexFormat.Uint32, 0, inst.GpuBuffers.IndexBuffer.SizeInBytes);
                _ctx.Wgpu.RenderPassEncoderDrawIndexed(pass, (uint)inst.GpuBuffers.IndexCount, 1, 0, 0, 0);
            }

            _ctx.Wgpu.RenderPassEncoderEnd(pass);
            _ctx.Wgpu.RenderPassEncoderRelease(pass);
            _ctx.Wgpu.TextureViewRelease(layerView);
        }

        foreach (IntPtr bg in tempBindGroups)
            _ctx.Wgpu.BindGroupRelease((BindGroup*)bg);
    }

    public void Dispose()
    {
        foreach (var ub in _uniformPool) ub.Dispose();
        _uniformPool.Clear();
        if (_shadowPipeline != null) _ctx.Wgpu.RenderPipelineRelease(_shadowPipeline);
        ShadowMapArray.Dispose();
    }
}
