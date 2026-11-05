using System.Numerics;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.GiPipeline;

namespace SurfelGI.Debug;

// Controlled inputs to the production gather: rendering scenes alone can hide
// normalization errors behind surfel coverage and tone mapping.
public static unsafe class ProbeFallbackValidation
{
    public static void Run(WebGpuContext ctx, ShaderManager shaders)
    {
        string source = shaders.PreprocessWgsl("Common/probe_fallback.wgsl") + """
            struct FixtureUniforms { camPos: vec4f, position: vec4f };
            @group(0) @binding(0) var<uniform> u: FixtureUniforms;
            @group(0) @binding(1) var<storage, read> probes: array<FallbackProbe>;
            @group(0) @binding(2) var<storage, read_write> result: array<vec4f>;
            """ + shaders.PreprocessWgsl("Common/probe_gather.wgsl")
            + shaders.PreprocessWgsl("Common/gi_resolve_helpers.wgsl")
            + shaders.PreprocessWgsl("Common/gi_radiance.wgsl") + """
            @compute @workgroup_size(1)
            fn validate() {
                result[0] = gatherProbeFallback(u.position.xyz, vec3f(0.0, 1.0, 0.0));
                result[1] = vec4f(surfel_resolve_confidence(10.0, 0.5),
                    surfel_resolve_confidence(1.0, 1.0),
                    surfel_resolve_confidence(0.0, 0.0), 0.0);
                result[2] = vec4f(gi_environment_radiance(vec3f(65000.0)), 1.0);
                result[3] = blend_surfel_probe(vec3f(0.4), 0.1, vec4f(0.0));
                result[4] = blend_surfel_probe(vec3f(0.4), 0.0, vec4f(0.2, 0.2, 0.2, 1.0));
                result[5] = blend_surfel_probe(vec3f(0.4), 0.0, vec4f(0.04, 0.04, 0.04, 0.2));
            }
            """;
        shaders.CreateModuleFromSource("ProbeGatherValidation", source);
        var pipeline = shaders.GetOrCreateComputePipeline("ProbeGatherValidation", "validate");
        using var uniforms = new GpuBuffer<Vector4>(ctx, 2, BufferUsage.Uniform);
        using var probes = new GpuBuffer<Vector4>(ctx, ProbeFallbackPass.ProbeCount * 14, BufferUsage.Storage);
        using var result = new GpuBuffer<Vector4>(ctx, 6, BufferUsage.Storage | BufferUsage.CopySrc);
        Span<BindGroupEntry> entries = stackalloc BindGroupEntry[] {
            uniforms.AsBindGroupEntry(0), probes.AsBindGroupEntry(1), result.AsBindGroupEntry(2)
        };
        var bg = shaders.CreateBindGroup(pipeline, 0, entries, "ProbeGatherValidation");
        try {
            foreach (string scenario in new[] { "constant", "empty", "weak-bright-corner", "occluded-bright-level", "stale-cell" }) {
                var data = new Vector4[ProbeFallbackPass.ProbeCount * 14];
                if (scenario != "empty") {
                    for (int level = scenario == "constant" ? 0 : 1; level < 5; level++)
                    for (int z = 0; z <= 1; z++) for (int y = 0; y <= 1; y++) for (int x = 0; x <= 1; x++)
                        SetProbe(data, x, y, z, level, 0.2f, 8 * (1 << level));
                }
                var position = new Vector4(0.5f, 0.5f, 0.5f, 0);
                if (scenario == "weak-bright-corner") {
                    position = new Vector4(0.05f, 0.05f, 0.05f, 0);
                    SetProbe(data, 1, 1, 1, 0, 100, 8);
                }
                if (scenario == "occluded-bright-level")
                    for (int z = 0; z <= 1; z++) for (int y = 0; y <= 1; y++) for (int x = 0; x <= 1; x++)
                        SetProbe(data, x, y, z, 0, 8, 0);
                if (scenario == "stale-cell") SetProbe(data, 16, 0, 0, 0, 100, 8);
                probes.Upload(data);
                uniforms.Upload(new[] { Vector4.Zero, position });
                var desc = new CommandEncoderDescriptor();
                var encoder = ctx.Wgpu.DeviceCreateCommandEncoder(ctx.Device, in desc);
                var pd = new ComputePassDescriptor();
                var pass = ctx.Wgpu.CommandEncoderBeginComputePass(encoder, &pd);
                ctx.Wgpu.ComputePassEncoderSetPipeline(pass, pipeline);
                ctx.Wgpu.ComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
                ctx.Wgpu.ComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
                ctx.Wgpu.ComputePassEncoderEnd(pass); ctx.Wgpu.ComputePassEncoderRelease(pass);
                var cd = new CommandBufferDescriptor();
                var command = ctx.Wgpu.CommandEncoderFinish(encoder, in cd);
                ctx.Wgpu.QueueSubmit(ctx.Queue, 1, &command);
                ctx.Wgpu.CommandBufferRelease(command); ctx.Wgpu.CommandEncoderRelease(encoder);
                var values = FrameCapture.ReadBuffer(ctx, result);
                var value = values[0];
                if (!float.IsFinite(value.X) || !float.IsFinite(value.W)) throw new InvalidOperationException($"Non-finite fallback: {scenario}");
                if (scenario == "empty") {
                    if (value != Vector4.Zero) throw new InvalidOperationException("Empty probes produced lighting.");
                } else if (MathF.Abs(value.X - 0.2f) > 0.03f || MathF.Abs(value.Y - value.X) > 1e-4f || MathF.Abs(value.Z - value.X) > 1e-4f) {
                    throw new InvalidOperationException($"Probe fallback {scenario}: expected coarse/constant lighting near 0.2, got {value}.");
                }
                if (MathF.Abs(values[1].X - 0.05f) > 1e-4f || values[1].Y != 1 || values[1].Z != 0)
                    throw new InvalidOperationException("Overlapping immature surfels suppressed the fallback.");
                if (MathF.Abs(values[3].X - 0.4f) > 1e-4f || MathF.Abs(values[4].X - 0.2f) > 1e-4f || MathF.Abs(values[5].X - 0.36f) > 1e-4f)
                    throw new InvalidOperationException("Unavailable probe support darkened provisional surfel lighting.");
                if (values[2].X > 15.001f || values[2].X < 14.9f)
                    throw new InvalidOperationException("HDR sky outlier was not compressed before cache integration.");
            }
        } finally { ctx.Wgpu.BindGroupRelease(bg); }
        Console.WriteLine("[Probe gather] GPU checks passed: sparse/occluded bright samples, coarse fill, stale cells, surfel confidence, HDR outliers.");
    }

    internal static void SetProbe(Vector4[] data, int x, int y, int z, int level, float light, float distance)
    {
        int slot = level * 4096 + (x & 15) + (y & 15) * 16 + (z & 15) * 256;
        int offset = slot * 14;
        data[offset] = new Vector4(BitConverter.Int32BitsToSingle(x), BitConverter.Int32BitsToSingle(y), BitConverter.Int32BitsToSingle(z), BitConverter.Int32BitsToSingle(level));
        data[offset + 1] = new Vector4(0, 0, 0, 4);
        for (int lobe = 0; lobe < 6; lobe++) {
            data[offset + 2 + lobe] = new Vector4(light, light, light, 0);
            data[offset + 8 + lobe] = new Vector4(distance, distance * distance, 0, 0);
        }
    }
}
