#include "Common/constants.wgsl"
#include "Common/grid_helpers.wgsl"

const MODE_OFF: u32 = 0u;
const MODE_SURFELS: u32 = 1u;
const MODE_IRRADIANCE: u32 = 2u;
const MODE_HEATMAP: u32 = 3u;
const MODE_CASCADES: u32 = 4u;
const MODE_CELL_HASH: u32 = 5u;
const MODE_GRID: u32 = 6u;
const MODE_WORLD_POS: u32 = 7u;
const MODE_DEPTH: u32 = 8u;
const MODE_NORMALS: u32 = 9u;
const MODE_ALBEDO: u32 = 10u;
const MODE_COVERAGE: u32 = 11u;
const MODE_SLG: u32 = 12u;
const MODE_SPAWN_DESPAWN: u32 = 13u;
const MODE_RADIAL_OCC: u32 = 14u;
const MODE_RADIAL_TILE: u32 = 15u;
const MODE_VARIANCE: u32 = 16u;
const MODE_ALBEDO_BVH: u32 = 17u;
const MODE_ALBEDO_MATCH: u32 = 18u;

struct DebugUniforms {
    projInv: mat4x4f,
    camWorld: mat4x4f,
    camPos: vec4f,
    prevCamPos: vec4f,
    gridOrigin: vec4f,
    occParams: vec4f,
    width: u32,
    height: u32,
    momentsOffset: u32,
    frame: u32,
    gridStride: u32,
    debugMode: u32,
    varianceScale: f32,
    maxFetch: i32,
};

@group(0) @binding(0) var<uniform> u: DebugUniforms;
@group(0) @binding(1) var texDepth: texture_depth_2d;
@group(0) @binding(2) var texNormal: texture_2d<f32>;
@group(0) @binding(3) var texAlbedo: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> surfels: array<SurfelPacked>;
@group(0) @binding(5) var<storage, read> moments: array<SurfelMoments>;
@group(0) @binding(6) var<storage, read> offsetsAndList: array<i32>;
@group(0) @binding(7) var<storage, read> guiding: array<f32>;
@group(0) @binding(8) var<storage, read> tileAlloc: array<i32>;
@group(0) @binding(9) var<storage, read> surfelDepth: array<vec4f>;
@group(0) @binding(10) var<storage, read_write> overlay: array<vec4f>;

fn load_radial_depth(index: u32) -> vec4f { return surfelDepth[index]; }
#include "Common/radial_depth.wgsl"

fn resolveIrradianceDebug(sid: i32, worldPos: vec3f, pixNormal: vec3f) -> vec4f {
    let s = surfels[sid];
    let sPos = s.posb.xyz;
    let sNor = s.normal;
    let momentsIndex = u32(sid) + u.momentsOffset;

    let birthFrame = s.posb.w;
    let sinceBirth = f32(u.frame) - birthFrame;

    let m = moments[momentsIndex];
    let sIrr = m.irradiance.xyz;
    let samples = m.irradiance.w;

    let sRad = surfel_radius_for_pos(sPos, u.camPos.xyz) * SURFEL_RADIUS_OVERSCALE;
    let eps = radius_based_epsilon(sRad);
    let sPosWithOffset = sPos + sNor * eps;
    let dV = worldPos - sPosWithOffset;
    let distLen = length(dV);

    let weightDir = max(0.0, dot(sNor, pixNormal));
    let dotN = abs(dot(dV, sNor));
    let squishFactor = 1.0 + dotN * SURFEL_NORMAL_DIRECTION_SQUISH;
    let mahalanobisDist = distLen * squishFactor;

    let weightGeom = (1.0 - smoothstep(0.0, sRad, mahalanobisDist));
    var wFinal = weightGeom * weightDir;

    let fade = clamp((sinceBirth + 1.0) / FADE_FRAMES, 0.0, 1.0);
    let conf = clamp(samples / TARGET_SAMPLE_COUNT, 0.0, 1.0);
    wFinal *= fade * conf;

    if (wFinal > 0.0) {
        let dirWS = dV / max(1e-6, distLen);
        let occ = surfel_radial_occlusion(u32(sid), dirWS, sNor, distLen, u.occParams);
        wFinal *= occ;
    }

    return vec4f(sIrr * wFinal, wFinal);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    _ = arrayLength(&guiding);
    let x = i32(id.x);
    let y = i32(id.y);
    let W = i32(u.width);
    let H = i32(u.height);
    if (x >= W || y >= H) {
        return;
    }

    let uvCoord = vec2f((f32(x) + 0.5) / f32(W), (f32(y) + 0.5) / f32(H));
    let depth = textureLoad(texDepth, vec2i(x, y), 0);
    let encN = textureLoad(texNormal, vec2i(x, y), 0).xyz;
    let pixelNormal = normalize(encN * 2.0 - 1.0);
    let validDepth = depth > 1e-6 && depth < 1.0;

    let viewPos = reconstruct_view_position(uvCoord, depth, u.projInv);
    let worldPos = (u.camWorld * vec4f(viewPos, 1.0)).xyz;
    let pRel = worldPos - u.gridOrigin.xyz;

    let gridCoord = surfel_pos_to_grid_coord(pRel);
    let c4 = surfel_grid_coord_to_c4(gridCoord);
    let rawHash = surfel_grid_c4_to_hash(c4);
    let cellIdx = i32(rawHash);

    let start = offsetsAndList[cellIdx];
    let end = offsetsAndList[cellIdx + 1];
    let count = max(end - start, 0);
    let capped = min(u.maxFetch, count);
    let capacity = i32(arrayLength(&surfels));

    var outC = vec4f(0.0);

    if (u.debugMode == MODE_SURFELS) {
        var sumR = 0.0;
        var sumG = 0.0;
        var sumB = 0.0;
        var sumW = 0.0;
        for (var i: i32 = 0; i < capped; i = i + 1) {
            let sid = offsetsAndList[OFFSETS_AND_LIST_START + start + i];
            if (sid >= 0 && sid < capacity) {
                let s = surfels[sid];
                let sPos = s.posb.xyz;
                let nRaw = s.normal;
                let sRad = surfel_radius_for_pos(sPos, u.camPos.xyz) * SURFEL_RADIUS_OVERSCALE;
                let pos_offset = worldPos - sPos;
                let directional_weight = max(0.0, dot(nRaw, pixelNormal));
                let dist = length(pos_offset);
                let mahalanobis_dist = dist * (1.0 + abs(dot(pos_offset, nRaw)) * SURFEL_NORMAL_DIRECTION_SQUISH);
                let weight = (1.0 - smoothstep(0.0, sRad, mahalanobis_dist)) * directional_weight;

                let f = f32(sid);
                let r = fract(sin(f * 12.9898)) * 0.75 + 0.25;
                let g = fract(r * 1.3) * 0.75 + 0.25;
                let b = fract(r * 1.7) * 0.75 + 0.25;

                sumR += r * weight;
                sumG += g * weight;
                sumB += b * weight;
                sumW += weight;
            }
        }
        let alpha = clamp(sumW, 0.0, 1.0);
        let invW = select(0.0, 1.0 / sumW, sumW > 0.0);
        outC = vec4f(sumR * invW, sumG * invW, sumB * invW, alpha);
    } else if (u.debugMode == MODE_IRRADIANCE) {
        var sumW = 0.0;
        var sumLight = vec3f(0.0);
        for (var i: i32 = 0; i < capped; i = i + 1) {
            let sid = offsetsAndList[OFFSETS_AND_LIST_START + start + i];
            if (sid >= 0 && sid < capacity) {
                let irr = resolveIrradianceDebug(sid, worldPos, pixelNormal);
                sumW += irr.w;
                sumLight += irr.xyz;
            }
        }
        let alpha = clamp(sumW, 0.0, 1.0);
        let invW = select(0.0, 1.0 / sumW, sumW > 0.0);
        outC = vec4f(sumLight * invW, alpha);
    } else if (u.debugMode == MODE_VARIANCE) {
        var sumW = 0.0;
        var sumVar = 0.0;
        for (var i: i32 = 0; i < capped; i = i + 1) {
            let sid = offsetsAndList[OFFSETS_AND_LIST_START + start + i];
            if (sid >= 0 && sid < capacity) {
                let s = surfels[sid];
                let sPos = s.posb.xyz;
                let sNor = s.normal;
                let sRad = surfel_radius_for_pos(sPos, u.camPos.xyz) * SURFEL_RADIUS_OVERSCALE;
                let posOffset = worldPos - sPos;
                let dist = length(posOffset);
                let alignPenalty = abs(dot(posOffset, sNor)) * SURFEL_NORMAL_DIRECTION_SQUISH;
                let mahal = dist * (1.0 + alignPenalty);
                let dotN = max(0.0, dot(sNor, pixelNormal));
                let weight = (1.0 - smoothstep(0.0, sRad, mahal)) * dotN;

                let momentsIndex = u32(sid) + u.momentsOffset;
                let varianceVec = moments[momentsIndex].msmeData1.xyz;
                let variance = dot(varianceVec, vec3f(0.2126, 0.7152, 0.0722));
                sumVar += variance * weight;
                sumW += weight;
            }
        }
        let invW = select(0.0, 1.0 / sumW, sumW > 0.0);
        let v = clamp(sumVar * invW * u.varianceScale, 0.0, 1.0);
        let r = clamp(1.5 - abs(v * 4.0 - 3.0), 0.0, 1.0);
        let g = clamp(1.5 - abs(v * 4.0 - 2.0), 0.0, 1.0);
        let b = clamp(1.5 - abs(v * 4.0 - 1.0), 0.0, 1.0);
        outC = vec4f(r, g, b, clamp(sumW, 0.0, 1.0));
    } else if (u.debugMode == MODE_HEATMAP) {
        let heat = f32(count) / 64.0;
        outC = select(vec4f(0.0, 0.0, 1.0, 1.0), vec4f(heat, 1.0 - heat, 0.0, 1.0), count > 0);
    } else if (u.debugMode == MODE_CASCADES) {
        let cascade = f32(c4.w) + 1.0;
        outC = vec4f(fract(cascade * 0.456), fract(cascade * 0.789), fract(cascade * 0.123), 1.0);
    } else if (u.debugMode == MODE_CELL_HASH) {
        let cf = f32(cellIdx);
        outC = vec4f(fract(cf * 0.123), fract(cf * 0.456), fract(cf * 0.789), 1.0);
    } else if (u.debugMode == MODE_GRID) {
        outC = vec4f(fract(f32(gridCoord.x) * 0.1), fract(f32(gridCoord.y) * 0.1), fract(f32(gridCoord.z) * 0.1), 1.0);
    } else if (u.debugMode == MODE_WORLD_POS) {
        outC = vec4f(worldPos, 1.0);
    } else if (u.debugMode == MODE_DEPTH) {
        let d = clamp(depth, 0.0, 1.0);
        outC = vec4f(d, d, d, 1.0);
    } else if (u.debugMode == MODE_NORMALS) {
        outC = vec4f(pixelNormal * 0.5 + 0.5, 1.0);
    } else if (u.debugMode == MODE_SPAWN_DESPAWN) {
        let gx = u32(x) / 8u;
        let gy = u32(y) / 8u;
        let tIdx = gy * u.gridStride + gx;
        let flag = tileAlloc[tIdx * 2u];
        let bestParentSelected = select(0.0, 1.0, tileAlloc[tIdx * 2u + 1u] > 0);
        outC = vec4f(f32(flag), -f32(flag), bestParentSelected, 1.0);
    } else if (u.debugMode == MODE_ALBEDO) {
        let albedo = textureLoad(texAlbedo, vec2i(x, y), 0).xyz;
        outC = vec4f(albedo, 1.0);
    } else if (u.debugMode == MODE_COVERAGE) {
        var aliveCount: i32 = 0;
        for (var i: i32 = 0; i < capped; i = i + 1) {
            let sid = offsetsAndList[OFFSETS_AND_LIST_START + start + i];
            if (sid >= 0 && sid < capacity) {
                if (surfels[sid].age < SURFEL_LIFE_RECYCLE) {
                    aliveCount += 1;
                }
            }
        }
        let deadCount = max(0, count - aliveCount);
        let aliveN = f32(aliveCount) / 64.0;
        let deadN = f32(deadCount) / 64.0;
        outC = select(vec4f(0.0, 0.0, 1.0, 1.0), vec4f(deadN, aliveN, 0.0, 1.0), count > 0);
    } else if (u.debugMode == MODE_SLG) {
        let tileX = x / 8;
        let tileY = y / 8;
        let localX = x % 8;
        let localY = y % 8;
        let centerX = min(W - 1, tileX * 8 + 4);
        let centerY = min(H - 1, tileY * 8 + 4);
        let centerUv = vec2f((f32(centerX) + 0.5) / f32(W), (f32(centerY) + 0.5) / f32(H));
        let depthC = textureLoad(texDepth, vec2i(centerX, centerY), 0);

        if (depthC > 1e-6 && depthC < 1.0) {
            let viewPosC = reconstruct_view_position(centerUv, depthC, u.projInv);
            let worldPosC = (u.camWorld * vec4f(viewPosC, 1.0)).xyz;
            let pRelC = worldPosC - u.gridOrigin.xyz;
            let gridCoordC = surfel_pos_to_grid_coord(pRelC);
            let c4C = surfel_grid_coord_to_c4(gridCoordC);
            let cellIdxC = i32(surfel_grid_c4_to_hash(c4C));

            let startC = offsetsAndList[cellIdxC];
            let endC = offsetsAndList[cellIdxC + 1];
            let countC = min(max(endC - startC, 0), 64);

            var bestSid: i32 = -1;
            var maxWeight: f32 = -1.0;
            for (var i: i32 = 0; i < countC; i = i + 1) {
                let sid = offsetsAndList[OFFSETS_AND_LIST_START + startC + i];
                if (sid >= 0 && sid < capacity) {
                    let s = surfels[sid];
                    let sPos = s.posb.xyz;
                    let sNor = s.normal;
                    let dist = length(worldPosC - sPos);
                    let dotN = max(0.0, dot(sNor, pixelNormal));
                    let sRad = surfel_radius_for_pos(sPos, u.camPos.xyz);
                    let weight = dotN * max(0.0, 1.0 - dist / sRad);
                    if (weight > maxWeight) {
                        maxWeight = weight;
                        bestSid = sid;
                    }
                }
            }

            if (bestSid >= 0) {
                let s = surfels[bestSid];
                let momentsIndex = u32(bestSid) + u.momentsOffset;
                let m = moments[momentsIndex];
                outC = m.guiding;

                let age = f32(u.frame) - s.posb.w;
                if (localX == 0 || localY == 0) {
                    outC = select(vec4f(1.0, 0.0, 0.0, 0.2), vec4f(0.0, 1.0, 0.0, 0.2), age > 10.0);
                }
            } else {
                outC = vec4f(0.1, 0.0, 0.0, 0.5);
            }
        }
    } else if (u.debugMode == MODE_RADIAL_OCC) {
        var bestSid: i32 = -1;
        var bestW: f32 = 0.0;
        for (var i: i32 = 0; i < capped; i = i + 1) {
            let sid = offsetsAndList[OFFSETS_AND_LIST_START + start + i];
            if (sid >= 0 && sid < capacity) {
                let s = surfels[sid];
                let sPos = s.posb.xyz;
                let sNor = s.normal;
                let sRad = surfel_radius_for_pos(sPos, u.camPos.xyz) * SURFEL_RADIUS_OVERSCALE;
                let dV = worldPos - sPos;
                let distLen = length(dV);
                let wDir = max(0.0, dot(sNor, pixelNormal));
                let dotN = abs(dot(dV, sNor));
                let mDist = distLen * (1.0 + dotN * SURFEL_NORMAL_DIRECTION_SQUISH);
                let w = (1.0 - smoothstep(0.0, sRad, mDist)) * wDir;
                if (w > bestW) {
                    bestW = w;
                    bestSid = sid;
                }
            }
        }

        if (bestSid >= 0 && bestW > 1e-6) {
            let s = surfels[bestSid];
            let sPos = s.posb.xyz;
            let sNor = s.normal;
            let dV = worldPos - sPos;
            let distLen = length(dV);
            let dirWS = dV / max(1e-6, distLen);

            let n = normalize(sNor);
            let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
            let t = normalize(cross(up, n));
            let b = cross(n, t);
            let hemi = vec3f(dot(dirWS, t), dot(dirWS, b), max(0.0, dot(dirWS, n)));
            let uv = hemiOctSquareEncode(normalize(hemi));
            let m = point_sample_radial_depth(u32(bestSid), uv);
            let occ = surfel_radial_occlusion(u32(bestSid), dirWS, sNor, distLen, u.occParams);
            let blocked = clamp(1.0 - occ, 0.0, 1.0);
            let relDist = distLen / max(0.001, m.x) - blocked;
            outC = vec4f(blocked, relDist, relDist, relDist);
        }
    } else if (u.debugMode == MODE_RADIAL_TILE) {
        let tileX = x / 8;
        let tileY = y / 8;
        let localX = x % 8;
        let localY = y % 8;
        let centerX = min(W - 1, tileX * 8 + 4);
        let centerY = min(H - 1, tileY * 8 + 4);
        let centerUv = vec2f((f32(centerX) + 0.5) / f32(W), (f32(centerY) + 0.5) / f32(H));
        let depthC = textureLoad(texDepth, vec2i(centerX, centerY), 0);

        if (depthC > 1e-6 && depthC < 1.0) {
            let encNC = textureLoad(texNormal, vec2i(centerX, centerY), 0).xyz;
            let nC = normalize(encNC * 2.0 - 1.0);
            let viewPosC = reconstruct_view_position(centerUv, depthC, u.projInv);
            let worldPosC = (u.camWorld * vec4f(viewPosC, 1.0)).xyz;
            let pRelC = worldPosC - u.gridOrigin.xyz;
            let gridCoordC = surfel_pos_to_grid_coord(pRelC);
            let c4C = surfel_grid_coord_to_c4(gridCoordC);
            let cellIdxC = i32(surfel_grid_c4_to_hash(c4C));

            let startC = offsetsAndList[cellIdxC];
            let endC = offsetsAndList[cellIdxC + 1];
            let countC = min(max(endC - startC, 0), 64);

            var bestSid: i32 = -1;
            var bestW: f32 = -1.0;
            for (var i: i32 = 0; i < countC; i = i + 1) {
                let sid = offsetsAndList[OFFSETS_AND_LIST_START + startC + i];
                if (sid >= 0 && sid < capacity) {
                    let sPos = surfels[sid].posb.xyz;
                    let sNor = surfels[sid].normal;
                    let dist = length(worldPosC - sPos);
                    let sRad = surfel_radius_for_pos(sPos, u.camPos.xyz) * SURFEL_RADIUS_OVERSCALE;
                    let w = max(0.0, dot(sNor, nC)) * max(0.0, 1.0 - dist / sRad);
                    if (w > bestW) {
                        bestW = w;
                        bestSid = sid;
                    }
                }
            }

            if (bestSid >= 0) {
                let tx = localX / 2;
                let ty = localY / 2;
                let uvRD = vec2f(
                    (f32(tx) + 0.5) / f32(SURFEL_DEPTH_TEXELS),
                    (f32(ty) + 0.5) / f32(SURFEL_DEPTH_TEXELS)
                );
                let m = point_sample_radial_depth(u32(bestSid), uvRD);
                let sPos = surfels[bestSid].posb.xyz;
                let sRad = surfel_radius_for_pos(sPos, u.camPos.xyz) * SURFEL_RADIUS_OVERSCALE;
                let maxDepth = sRad * 2.0;
                let meanN = clamp(m.x / max(1e-6, maxDepth), 0.0, 1.0);
                outC = vec4f(meanN, meanN, m.w / 255.0, 0.75);
            }
        }
    }

    let doWrite = validDepth || (u.debugMode == MODE_SLG);
    let flatPix = y * W + x;
    overlay[flatPix] = select(vec4f(0.0), outC, doWrite);
}

// -----------------------------------------------------------------------------
// BVH Albedo / Match Debug Entry Point (uses separate bind group layout)
// -----------------------------------------------------------------------------
@group(0) @binding(11) var<storage, read> bvh: array<BVHNode>;
@group(0) @binding(12) var<storage, read> bvh_position: array<vec4f>;
@group(0) @binding(13) var<storage, read> bvh_index: array<vec4u>;
@group(0) @binding(14) var<storage, read> bvh_attribute: array<vec4f>;
@group(0) @binding(15) var diffuseTex: texture_2d_array<f32>;
@group(0) @binding(16) var diffuseTexSampler: sampler;

#include "Common/bvh_traverse.wgsl"

@compute @workgroup_size(8, 8, 1)
fn cs_bvh(@builtin(global_invocation_id) id: vec3u) {
    let x = i32(id.x);
    let y = i32(id.y);
    let W = i32(u.width);
    let H = i32(u.height);
    if (x >= W || y >= H) {
        return;
    }

    let uvCoord = vec2f((f32(x) + 0.5) / f32(W), (f32(y) + 0.5) / f32(H));
    let depth = textureLoad(texDepth, vec2i(x, y), 0);
    let validDepth = depth > 1e-6 && depth < 1.0;

    let viewPos = reconstruct_view_position(uvCoord, depth, u.projInv);
    let worldPos = (u.camWorld * vec4f(viewPos, 1.0)).xyz;

    var outC = vec4f(0.0);
    if (validDepth) {
        var ray: Ray;
        ray.origin = u.camPos.xyz;
        ray.direction = normalize(worldPos - u.camPos.xyz);

        let hit = bvhIntersectFirstHit(ray);
        if (hit.didHit) {
            let uvMat = getVertexAttribute(hit.barycoord, hit.indices.xyz);
            let matId = i32(round(uvMat.z));
            let layerCount = i32(textureNumLayers(diffuseTex));
            let layer = clamp(matId, 0, max(0, layerCount - 1));
            let arrayAlbedo = textureSampleLevel(diffuseTex, diffuseTexSampler, uvMat.xy, layer, 0.0).rgb;
            let gbufferAlbedo = textureLoad(texAlbedo, vec2i(x, y), 0).rgb;
            let err = abs(arrayAlbedo - gbufferAlbedo);
            let color = select(arrayAlbedo, err, u.debugMode == MODE_ALBEDO_MATCH);
            outC = vec4f(color, 1.0);
        } else {
            outC = vec4f(0.0, 0.0, 0.0, 1.0);
        }
    }

    let flatPix = y * W + x;
    overlay[flatPix] = select(vec4f(0.0), outC, validDepth);
}

// -----------------------------------------------------------------------------
// Overlay Blit Shader
// -----------------------------------------------------------------------------
struct BlitUniforms {
    width: f32,
    height: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> uBlit: BlitUniforms;
@group(0) @binding(1) var<storage, read> overlayRead: array<vec4f>;

struct BlitVSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_blit(@builtin(vertex_index) vid: u32) -> BlitVSOut {
    var out: BlitVSOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.pos = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2f(x, y);
    return out;
}

@fragment
fn fs_blit(in: BlitVSOut) -> @location(0) vec4f {
    let x = clamp(i32(floor(in.uv.x * uBlit.width)), 0, i32(uBlit.width) - 1);
    let y = clamp(i32(floor(in.uv.y * uBlit.height)), 0, i32(uBlit.height) - 1);
    let flat = y * i32(uBlit.width) + x;
    let val = overlayRead[flat];
    return vec4f(val.rgb * val.a, val.a);
}


