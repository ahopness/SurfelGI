using System.Numerics;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.GiPipeline;

namespace SurfelGI.Debug;

// Exercise the production resolve shader with a crowded cell. Only one surfel
// faces the receiver; permuting its slot must never make it disappear.
public static unsafe class SurfelResolveValidation
{
    public static void Run(WebGpuContext ctx, ShaderManager shaders)
    {
        const int count = 160;
        const int usefulSid = count - 1;
        using var depth = new GpuTexture(ctx, 1, 1, TextureFormat.Depth32Float,
            TextureUsage.RenderAttachment | TextureUsage.TextureBinding);
        using var normal = new GpuTexture(ctx, 1, 1, TextureFormat.Rgba16Float,
            TextureUsage.RenderAttachment | TextureUsage.TextureBinding);
        using var output = new GpuTexture(ctx, 1, 1, TextureFormat.Rgba16Float,
            TextureUsage.StorageBinding | TextureUsage.TextureBinding);

        var surfels = new SurfelPacked[count];
        var history = new SurfelMoments[count * 2];
        for (int i = 0; i < count; i++)
        {
            surfels[i] = new SurfelPacked { PosB = new Vector4(0, 0, 0.5f, 0), Normal = -Vector3.UnitZ };
            history[i].Irradiance = history[i + count].Irradiance = new Vector4(0.25f, 0.5f, 1, 32);
        }
        surfels[usefulSid].Normal = Vector3.UnitZ;
        using var surfelBuffer = new GpuBuffer<SurfelPacked>(ctx, surfels, BufferUsage.Storage);
        using var moments = new GpuBuffer<SurfelMoments>(ctx, history, BufferUsage.Storage);
        using var grid = new GpuBuffer<int>(ctx, Constants.OFFSETS_AND_LIST_START + count, BufferUsage.Storage);
        using var touched = new GpuBuffer<int>(ctx, count, BufferUsage.Storage | BufferUsage.CopySrc);
        using var radialDepth = new GpuBuffer<Vector4>(ctx,
            count * Constants.SURFEL_DEPTH_TEXELS * Constants.SURFEL_DEPTH_TEXELS, BufferUsage.Storage);
        using var uniforms = new GpuBuffer<ResolveUniforms>(ctx, 1, BufferUsage.Uniform);
        // Identity reconstruction maps the pixel to (0,0,0.5), in cascade zero.
        int half = Constants.SURFEL_CS / 2;
        int cell = half + half * Constants.SURFEL_CS + (half + 2) * Constants.SURFEL_CS * Constants.SURFEL_CS;
        grid.Upload(new[] { 0, count }, (ulong)cell);
        using var fallback = new ProbeFallbackPass(ctx, shaders);
        var probeData = new Vector4[ProbeFallbackPass.ProbeCount * 14];
        for (int level = 0; level < 5; level++)
        for (int z = 0; z <= 1; z++) for (int y = 0; y <= 1; y++) for (int x = 0; x <= 1; x++)
            ProbeFallbackValidation.SetProbe(probeData, x, y, z, level, 0.2f, 8 * (1 << level));
        fallback.Buffer.Upload(probeData);
        // Read the production resolve texture back through a separate compute pass.
        shaders.CreateModuleFromSource("ResolvePixelReadback", """
            @group(0) @binding(0) var input: texture_2d<f32>;
            @group(0) @binding(1) var<storage, read_write> result: array<vec4f>;
            @compute @workgroup_size(1) fn main() { result[0] = textureLoad(input, vec2i(0), 0); }
            """);
        var reader = shaders.GetOrCreateComputePipeline("ResolvePixelReadback");
        using var pixelResult = new GpuBuffer<Vector4>(ctx, 1, BufferUsage.Storage | BufferUsage.CopySrc);
        Span<BindGroupEntry> readerEntries = stackalloc BindGroupEntry[] { output.AsTextureEntry(0), pixelResult.AsBindGroupEntry(1) };
        var readerBg = shaders.CreateBindGroup(reader, 0, readerEntries, "ResolvePixelReadback");
        var pipeline = shaders.GetOrCreateComputePipeline("surfel_resolve.wgsl", "main");
        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[]
        {
            uniforms.AsBindGroupEntry(0), depth.AsTextureEntry(1), normal.AsTextureEntry(2),
            surfelBuffer.AsBindGroupEntry(3), moments.AsBindGroupEntry(4), grid.AsBindGroupEntry(5),
            touched.AsBindGroupEntry(6), radialDepth.AsBindGroupEntry(7), output.AsTextureEntry(8), fallback.Buffer.AsBindGroupEntry(9)
        };
        var bg = shaders.CreateBindGroup(pipeline, 0, entries, "ResolveValidation");
        try
        {
            foreach (int slot in new[] { 0, 31, 32, 63, 64, 127, 128, count - 1 })
            foreach (uint bank in new[] { 0u, (uint)count })
            foreach (uint sourceMode in new[] { 0u, 1u, 2u, 3u })
            foreach (float sampleCount in new[] { 32f, 1f, 0f })
            {
                if (sampleCount != 32 && (slot != 0 || bank != 0)) continue;
                for (int i = 0; i < history.Length; i++) history[i].Irradiance.W = sampleCount;
                moments.Upload(history);
                int[] list = Enumerable.Range(0, count).ToArray();
                (list[slot], list[usefulSid]) = (list[usefulSid], list[slot]);
                grid.Upload(list, Constants.OFFSETS_AND_LIST_START);
                touched.Upload(new int[count]);
                uniforms.UploadSingle(new ResolveUniforms
                {
                    ProjInv = Matrix4x4.Identity, CamWorld = Matrix4x4.Identity,
                    Width = 1, Height = 1, Frame = 64, ResolveOffset = bank, SourceMode = sourceMode
                });

                var desc = new CommandEncoderDescriptor();
                var encoder = ctx.Wgpu.DeviceCreateCommandEncoder(ctx.Device, in desc);
                var color = new RenderPassColorAttachment
                {
                    View = normal.View, LoadOp = LoadOp.Clear, StoreOp = StoreOp.Store,
                    ClearValue = new Color(0.5, 0.5, 1, 1)
                };
                var depthAttachment = new RenderPassDepthStencilAttachment
                {
                    View = depth.View, DepthLoadOp = LoadOp.Clear, DepthStoreOp = StoreOp.Store, DepthClearValue = 0.5f
                };
                var renderDesc = new RenderPassDescriptor
                {
                    ColorAttachmentCount = 1, ColorAttachments = &color, DepthStencilAttachment = &depthAttachment
                };
                var render = ctx.Wgpu.CommandEncoderBeginRenderPass(encoder, &renderDesc);
                ctx.Wgpu.RenderPassEncoderEnd(render);
                ctx.Wgpu.RenderPassEncoderRelease(render);
                var computeDesc = new ComputePassDescriptor();
                var compute = ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &computeDesc);
                ctx.Wgpu.ComputePassEncoderSetPipeline(compute, pipeline);
                ctx.Wgpu.ComputePassEncoderSetBindGroup(compute, 0, bg, 0, null);
                ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(compute, 1, 1, 1);
                ctx.Wgpu.ComputePassEncoderEnd(compute);
                ctx.Wgpu.ComputePassEncoderRelease(compute);
                var readPass = ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &computeDesc);
                ctx.Wgpu.ComputePassEncoderSetPipeline(readPass, reader);
                ctx.Wgpu.ComputePassEncoderSetBindGroup(readPass, 0, readerBg, 0, null);
                ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(readPass, 1, 1, 1);
                ctx.Wgpu.ComputePassEncoderEnd(readPass); ctx.Wgpu.ComputePassEncoderRelease(readPass);
                var cbDesc = new CommandBufferDescriptor();
                var command = ctx.Wgpu.CommandEncoderFinish(encoder, in cbDesc);
                ctx.Wgpu.QueueSubmit(ctx.Queue, 1, &command);
                ctx.Wgpu.CommandBufferRelease(command);
                ctx.Wgpu.CommandEncoderRelease(encoder);
                var result = FrameCapture.ReadBuffer(ctx, touched);
                if (ctx.LastError != null) throw new InvalidOperationException(ctx.LastError);
                var pixel = FrameCapture.ReadBuffer(ctx, pixelResult)[0];
                var surfelLight = sampleCount > 0 ? new Vector3(0.25f, 0.5f, 1) : Vector3.Zero;
                var expected = sourceMode switch {
                    0 => Vector3.Zero,
                    1 => new Vector3(0.2f),
                    2 => surfelLight,
                    _ => Vector3.Lerp(new Vector3(0.2f), surfelLight, Math.Clamp(sampleCount / 32, 0, 1))
                };
                if (Vector3.Distance(new Vector3(pixel.X, pixel.Y, pixel.Z), expected) > 0.002f)
                    throw new InvalidOperationException($"Resolve source {sourceMode}, samples {sampleCount}: expected {expected}, got {pixel}.");
                if (sourceMode < 2 || sampleCount == 0) {
                    if (result.Any(v => v != 0)) throw new InvalidOperationException("Ambient/probe-only resolve accessed contributing surfels.");
                    continue;
                }
                if (result[usefulSid] <= 0 || result.Take(usefulSid).Any(v => v != 0))
                    throw new InvalidOperationException($"Resolve lost the only contributing surfel at cell slot {slot}, moments bank {bank}.");
            }
        }
        finally { ctx.Wgpu.BindGroupRelease(bg); ctx.Wgpu.BindGroupRelease(readerBg); }
        Console.WriteLine("[Resolve] GPU checks passed: crowded cell permutations across lookup limits, both moments banks, GI source isolation, empty coverage, and young-surfel blending.");
    }
}
