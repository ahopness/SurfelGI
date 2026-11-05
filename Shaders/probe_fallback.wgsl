#include "Common/constants.wgsl"
#include "Common/probe_fallback.wgsl"
#include "Common/hash_utils.wgsl"
#include "Common/gi_radiance.wgsl"
#include "Common/update_schedule.wgsl"
struct ProbeUniforms {
    camera: vec4f, lightDirection: vec4f, lightColor: vec4f, transport: vec4f,
    frame: u32, reset: u32, pad0: u32, pad1: u32,
};
@group(0) @binding(0) var<uniform> u: ProbeUniforms;
@group(0) @binding(1) var<storage, read> bvh: array<BVHNode>;
@group(0) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(0) @binding(3) var<storage, read> bvh_index: array<vec4u>;
@group(0) @binding(4) var<storage, read> bvh_attribute: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> probes: array<FallbackProbe>;
@group(0) @binding(6) var diffuseTex: texture_2d_array<f32>;
@group(0) @binding(7) var diffuseSampler: sampler;
@group(0) @binding(8) var skybox: texture_2d<f32>;
@group(0) @binding(9) var skySampler: sampler;
#include "Common/bvh_traverse.wgsl"

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let slot = id.x;
    if (slot >= PROBES_PER_LEVEL * PROBE_LEVELS) { return; }
    let level = slot / PROBES_PER_LEVEL;
    let local = slot % PROBES_PER_LEVEL;
    let spacing = probe_spacing(level);
    let origin = vec3i(floor(u.camera.xyz / spacing)) - vec3i(8);
    let wrapped = vec3i(i32(local % 16u), i32((local / 16u) % 16u), i32(local / 256u));
    let cell = origin + ((wrapped - origin) & vec3i(15));
    var p = probes[slot];
    if (u.reset != 0u || any(p.key.xyz != cell)) {
        p = FallbackProbe();
        p.key = vec4i(cell, i32(level));
        probes[slot] = p;
    }
    // A spatial permutation distributes each 1/16 update cohort across every level.
    if ((permute_update_index(local, PROBES_PER_LEVEL) & 15u) != (u.frame & 15u)) {
        return;
    }
    let position = vec3f(cell) * spacing + p.offset.xyz;
    var lights: array<vec3f, 6>;
    var depths: array<vec2f, 6>;
    var weights: array<f32, 6>;
    var backfaces = 0u;
    var escape = vec3f(0.0);
    var nearestBack = spacing;
    // Rotated spherical samples accumulate angular coverage across refreshes.
    let rotation = uintToU01Float(hash1(slot ^ (u.frame / 16u) * 747796405u)) * 2.0 * PI;
    for (var r = 0u; r < 8u; r++) {
        let z = 1.0 - 2.0 * (f32(r) + 0.5) / 8.0;
        let phi = f32(r) * 2.39996323 + rotation;
        let radial = sqrt(max(0.0, 1.0 - z * z));
        let direction = vec3f(cos(phi) * radial, z, sin(phi) * radial);
        let hit = bvhIntersectFirstHit(Ray(position, direction));
        var distance = spacing * 8.0;
        var radiance = vec3f(0.0);
        if (hit.didHit) {
            distance = min(hit.dist, distance);
            if (hit.side < 0.0) {
                backfaces++;
                if (hit.dist < nearestBack) { nearestBack = hit.dist; escape = direction * (hit.dist + spacing * 0.05); }
            } else {
                let point = position + direction * hit.dist;
                let hitAttribute = getVertexAttribute(hit.barycoord, hit.indices.xyz);
                var albedo = textureSampleLevel(diffuseTex, diffuseSampler, hitAttribute.xy, i32(round(hitAttribute.z)), 0.0).rgb;
                let lum = max(1e-4, dot(albedo, vec3f(0.2126, 0.7152, 0.0722)));
                albedo = clamp(albedo * ((1.0 - pow(1.0 - lum, max(0.0, u.transport.w))) / lum), vec3f(0.0), vec3f(1.0));
                let ndotl = max(0.0, dot(hit.normal, u.lightDirection.xyz));
                if (ndotl > 0.0 && u.transport.z > 0.0) {
                    if (!bvhOccluded(Ray(point + hit.normal * 0.005, u.lightDirection.xyz))) {
                        radiance = albedo * u.lightColor.rgb * ndotl * (max(0.0, u.transport.z) / PI);
                    }
                }
            }
        } else {
            let uv = vec2f(atan2(direction.z, direction.x) * (0.5 / PI) + 0.5, acos(clamp(direction.y, -1.0, 1.0)) / PI);
            radiance = gi_environment_radiance(textureSampleLevel(skybox, skySampler, uv, max(0.0, u.transport.y)).rgb) * max(0.0, u.transport.x);
        }
        for (var l = 0u; l < 6u; l++) {
            let w = max(0.0, dot(direction, probe_axis(l)));
            lights[l] += radiance * w;
            depths[l] += vec2f(distance, distance * distance) * w;
            weights[l] += w;
        }
    }
    if (backfaces >= 5u) {
        // Best-effort bounded escape. Never expose lighting measured inside solid geometry.
        p.offset = vec4f(clamp(p.offset.xyz + escape, vec3f(-spacing * 0.45), vec3f(spacing * 0.45)), 0.0);
        probes[slot] = p;
        return;
    }
    let alpha = select(0.25, 1.0, p.offset.w == 0.0);
    for (var l = 0u; l < 6u; l++) {
        let inv = 1.0 / max(weights[l], 1e-5);
        p.light[l] = vec4f(mix(p.light[l].rgb, lights[l] * inv, alpha), 0.0);
        p.depth[l] = vec4f(mix(p.depth[l].xy, depths[l] * inv, alpha), 0.0, 0.0);
    }
    p.offset.w = min(p.offset.w + 1.0, 64.0);
    probes[slot] = p;
}
