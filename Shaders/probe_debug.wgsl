#include "Common/constants.wgsl"
#include "Common/grid_helpers.wgsl"
#include "Common/probe_fallback.wgsl"
struct ProbeDebugUniforms {
    viewProjection: mat4x4f, projInv: mat4x4f, camWorld: mat4x4f,
    camPos: vec4f,
    params: vec4f, // level, radius as fraction of spacing, xray, exposure
    mode: u32, width: u32, height: u32, pad: u32,
};
@group(0) @binding(0) var<uniform> u: ProbeDebugUniforms;
@group(0) @binding(1) var<storage, read> probes: array<FallbackProbe>;
@group(0) @binding(2) var sceneDepth: texture_depth_2d;
@group(0) @binding(3) var sceneNormal: texture_2d<f32>;
#include "Common/probe_gather.wgsl"
struct SphereOut {
    @builtin(position) position: vec4f,
    @location(0) local: vec2f,
    @location(1) @interpolate(flat) slot: u32,
    @location(2) @interpolate(flat) center: vec3f,
};
@vertex
fn vs_probe(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> SphereOut {
    let level = u32(u.params.x);
    let spacing = probe_spacing(level);
    let slot = level * PROBES_PER_LEVEL + instance;
    let p = probes[slot];
    let center = vec3f(p.key.xyz) * spacing + p.offset.xyz;
    let corners = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1));
    let local = corners[vertex];
    let world = center + (u.camWorld[0].xyz * local.x + u.camWorld[1].xyz * local.y) * spacing * u.params.y;
    var out: SphereOut;
    out.position = u.viewProjection * vec4f(world, 1.0);
    out.local = local; out.slot = slot; out.center = center;
    return out;
}
@fragment
fn fs_probe(in: SphereOut) -> @location(0) vec4f {
    let r2 = dot(in.local, in.local);
    if (r2 > 1.0) { discard; }
    let p = probes[in.slot];
    let level = u32(u.params.x);
    let spacing = probe_spacing(level);
    let origin = vec3i(floor(u.camPos.xyz / spacing)) - vec3i(8);
    if (any(p.key.xyz < origin) || any(p.key.xyz >= origin + vec3i(16)) || p.key.w != i32(level)) { discard; }
    let normal = normalize(u.camWorld[0].xyz * in.local.x + u.camWorld[1].xyz * in.local.y + u.camWorld[2].xyz * sqrt(1.0 - r2));
    let surface = in.center + normal * spacing * u.params.y;
    let projected = u.viewProjection * vec4f(surface, 1.0);
    let depth = textureLoad(sceneDepth, vec2i(in.position.xy), 0);
    let hidden = projected.z / projected.w > depth + 0.00001;
    if (hidden && u.params.z < 0.5) { discard; }
    let valid = p.offset.w > 0.0;
    var color = vec3f(1.0, 0.05, 0.1);
    if (valid) {
        if (u.mode == 19u) {
            color = mix(vec3f(1.0, 0.6, 0.05), vec3f(0.1, 1.0, 0.25), clamp((p.offset.w - 1.0) / 3.0, 0.0, 1.0));
        } else if (u.mode == 20u) {
            let n2 = normal * normal;
            var light = p.light[select(0u,1u,normal.x < 0.0)].rgb * n2.x;
            light += p.light[select(2u,3u,normal.y < 0.0)].rgb * n2.y;
            light += p.light[select(4u,5u,normal.z < 0.0)].rgb * n2.z;
            light *= u.params.w;
            color = light / (vec3f(1.0) + light);
        } else if (u.mode == 21u) {
            let mean = p.depth[probe_direction_lobe(normal)].x / (spacing * 8.0);
            color = vec3f(clamp(mean, 0.0, 1.0));
        } else {
            color = abs(p.offset.xyz) / (spacing * 0.45);
        }
    }
    // Keep invalid relocated probes visible in relocation mode too.
    if (u.mode == 22u && length(p.offset.xyz) > 0.0) { color = abs(p.offset.xyz) / (spacing * 0.45); }
    if (u.mode != 20u && u.mode != 21u) { color *= 0.45 + 0.55 * sqrt(1.0 - r2); }
    return vec4f(color, select(1.0, 0.2, hidden));
}
struct FullscreenOut { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@vertex
fn vs_surface(@builtin(vertex_index) vertex: u32) -> FullscreenOut {
    let xy = vec2f(f32((vertex << 1u) & 2u), f32(vertex & 2u));
    return FullscreenOut(vec4f(xy.x * 2.0 - 1.0, 1.0 - xy.y * 2.0, 0.0, 1.0), xy);
}
@fragment
fn fs_surface(in: FullscreenOut) -> @location(0) vec4f {
    let pixel = vec2i(in.position.xy);
    let depth = textureLoad(sceneDepth, pixel, 0);
    if (depth >= 1.0 || depth <= 0.0) { discard; }
    let view = reconstruct_view_position(in.uv, depth, u.projInv);
    let position = (u.camWorld * vec4f(view, 1.0)).xyz;
    let normal = normalize(textureLoad(sceneNormal, pixel, 0).xyz * 2.0 - 1.0);
    let fallback = gatherProbeFallback(position, normal);
    if (fallback.w == 0.0) { return vec4f(1.0, 0.0, 1.0, 1.0); }
    let light = fallback.rgb * u.params.w;
    return vec4f(light / (vec3f(1.0) + light), 1.0);
}
