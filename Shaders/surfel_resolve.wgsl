#include "Common/constants.wgsl"
#include "Common/grid_helpers.wgsl"
#include "Common/probe_fallback.wgsl"

struct ResolveUniforms {
    projInv: mat4x4f,
    camWorld: mat4x4f,
    camPos: vec4f,
    gridOrigin: vec4f,
    occParams: vec4f,
    width: u32,
    height: u32,
    resolveOffset: u32,
    frame: u32,
    sourceMode: u32,
    pad0: u32, pad1: u32, pad2: u32,
};

@group(0) @binding(0) var<uniform> u: ResolveUniforms;
@group(0) @binding(1) var texDepth: texture_depth_2d;
@group(0) @binding(2) var texNormal: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> surfels: array<SurfelPacked>;
@group(0) @binding(4) var<storage, read> moments: array<SurfelMoments>;
@group(0) @binding(5) var<storage, read> offsetsAndList: array<i32>;
@group(0) @binding(6) var<storage, read_write> touched: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read> surfelDepth: array<vec4f>;
@group(0) @binding(8) var outputTex: texture_storage_2d<rgba16float, write>;

@group(0) @binding(9) var<storage, read> probes: array<FallbackProbe>;

#include "Common/probe_gather.wgsl"
#include "Common/gi_resolve_helpers.wgsl"

fn load_radial_depth(index: u32) -> vec4f { return surfelDepth[index]; }
#include "Common/radial_depth.wgsl"

struct SurfelResolveSample {
    weightedLight: vec3f,
    weight: f32,
    geometryWeight: f32,
};

fn resolveIrradiance(
    sid: i32,
    worldPos: vec3f,
    pixNormal: vec3f
) -> SurfelResolveSample {
    let s = surfels[sid];
    let sNor = s.normal;
    let weightDir = dot(sNor, pixNormal);
    if (weightDir <= 0.0) {
        return SurfelResolveSample();
    }

    let momentsIndex = u32(sid) + u.resolveOffset;
    let m = moments[momentsIndex];
    let samples = m.irradiance.w;
    if (samples <= 0.0) {
        return SurfelResolveSample();
    }

    let sPos = s.posb.xyz;
    let sRad = surfel_radius_for_pos(sPos, u.camPos.xyz) * SURFEL_RADIUS_OVERSCALE;
    let eps = radius_based_epsilon(sRad);
    let sPosWithOffset = sPos + sNor * eps;
    let dV = worldPos - sPosWithOffset;
    let distSq = dot(dV, dV);
    if (distSq >= sRad * sRad) {
        return SurfelResolveSample();
    }
    let distLen = sqrt(distSq);

    let dotN = abs(dot(dV, sNor));
    let squishFactor = 1.0 + dotN * SURFEL_NORMAL_DIRECTION_SQUISH;
    let mahalanobisDist = distLen * squishFactor;

    let weightGeom = (1.0 - smoothstep(0.0, sRad, mahalanobisDist));
    var geomW = weightGeom * weightDir;
    if (geomW <= 0.0) {
        return SurfelResolveSample();
    }

    let sIrr = m.irradiance.xyz;
    let birthFrame = s.posb.w;
    let sinceBirth = f32(u.frame) - birthFrame;

    let dirWS = dV / max(1e-6, distLen);
    let occ = surfel_radial_occlusion(u32(sid), dirWS, sNor, distLen, u.occParams);
    geomW *= occ;

    let fade = clamp((sinceBirth + 1.0) / FADE_FRAMES, 0.0, 1.0);
    let conf = clamp(samples / TARGET_SAMPLE_COUNT, 0.0, 1.0);
    let wFinal = geomW * fade * conf;

    return SurfelResolveSample(sIrr * wFinal, wFinal, geomW);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;
    if (x >= u.width || y >= u.height) {
        return;
    }

    let uv = vec2f((f32(x) + 0.5) / f32(u.width), (f32(y) + 0.5) / f32(u.height));
    let gbPixel = vec2i(uv * vec2f(textureDimensions(texDepth)));
    let depth = textureLoad(texDepth, gbPixel, 0);
    let valid = depth < 1.0 && depth > 0.0;

    var outColor = vec4f(0.0);

    if (valid) {
        let encN = textureLoad(texNormal, gbPixel, 0).xyz;
        let pixNormal = normalize(encN * 2.0 - 1.0);

        let viewPos = reconstruct_view_position(uv, depth, u.projInv);
        let worldPos = (u.camWorld * vec4f(viewPos, 1.0)).xyz;

        // Ambient is evaluated from the skybox in the composite, without cache reads.
        if (u.sourceMode == 0u) {
            textureStore(outputTex, vec2i(id.xy), vec4f(0.0, 0.0, 0.0, 1.0));
            return;
        }
        if (u.sourceMode == 1u) {
            let fallback = gatherProbeFallback(worldPos, pixNormal);
            textureStore(outputTex, vec2i(id.xy), vec4f(fallback.rgb, fallback.w));
            return;
        }

        let pRel = worldPos - u.gridOrigin.xyz;
        let gridCoord = surfel_pos_to_grid_coord(pRel);
        let c4 = surfel_grid_coord_to_c4(gridCoord);
        let rawHash = surfel_grid_c4_to_hash(c4);
        let cellIdx = i32(rawHash);

        let start = offsetsAndList[cellIdx];
        let end = offsetsAndList[cellIdx + 1];
        // Grid slots are filled atomically in a different order each frame.
        // A prefix cap can omit every surfel covering this pixel and blink black.
        let listEnd = min(end, i32(arrayLength(&offsetsAndList)) - OFFSETS_AND_LIST_START);
        let count = max(listEnd - start, 0);

        var sumLight = vec3f(0.0);
        var sumWeight = 0.0;
        var sumGeometryWeight = 0.0;
        var bestContrib = 0.0;
        var bestSid: i32 = -1;
        let capacity = i32(arrayLength(&surfels));

        for (var i: i32 = 0; i < count; i = i + 1) {
            let sid = offsetsAndList[OFFSETS_AND_LIST_START + start + i];
            if (sid >= 0 && sid < capacity && surfels[sid].age < SURFEL_TTL) {
                let irr = resolveIrradiance(sid, worldPos, pixNormal);
                if (irr.weight > bestContrib) {
                    bestContrib = irr.weight;
                    bestSid = sid;
                }
                sumLight += irr.weightedLight;
                sumWeight += irr.weight;
                sumGeometryWeight += irr.geometryWeight;
            }
        }

        let light = sumLight / max(sumWeight, 1e-5);
        let coverage = surfel_resolve_confidence(sumGeometryWeight, sumWeight);
        var resolved = light;
        var resolvedCoverage = coverage;
        if (u.sourceMode == 3u && coverage < 0.999) {
            let fallback = gatherProbeFallback(worldPos, pixNormal);
            if (fallback.w > 0.0) {
                let blended = blend_surfel_probe(light, coverage, fallback);
                resolved = blended.rgb;
                resolvedCoverage = blended.w;
            }
        }
        outColor = vec4f(resolved, resolvedCoverage);

        let dominance = bestContrib / (sumWeight + 1e-5);
        if (bestSid >= 0 && dominance > 0.5) {
            let impRange = f32(SURFEL_IMPORTANCE_DIRECT_MAX - SURFEL_IMPORTANCE_INDIRECT_MAX);
            let base = f32(SURFEL_IMPORTANCE_INDIRECT_MAX);
            let importance = clamp(i32(base + dominance * impRange), 0, 100);
            atomicMax(&touched[bestSid], importance);
        }

        if (bestSid >= 0 && bestContrib > 0.05) {
            atomicMax(&touched[bestSid], 2);
        }
    }

    textureStore(outputTex, vec2i(i32(x), i32(y)), outColor);
}
