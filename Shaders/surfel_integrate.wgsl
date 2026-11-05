#include "Common/constants.wgsl"
#include "Common/grid_helpers.wgsl"

struct IntegrateUniforms {
    lightDir: vec4f,
    lightColor: vec4f,
    camPos: vec4f,
    gridOrigin: vec4f,
    occParams: vec4f,
    frame: u32,
    readOffset: u32,
    writeOffset: u32,
    baseSampleCount: u32,
    albedoBoost: f32,
    giFromDirect: f32,
    giFromIndirect: f32,
    envIntensity: f32,
    envLod: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: IntegrateUniforms;
@group(0) @binding(1) var<storage, read> bvh: array<BVHNode>;
@group(0) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(0) @binding(3) var<storage, read> bvh_index: array<vec4u>;
@group(0) @binding(4) var<storage, read> bvh_attribute: array<vec4f>;
@group(0) @binding(5) var<storage, read> surfels: array<SurfelPacked>;
@group(0) @binding(6) var<storage, read_write> moments: array<SurfelMoments>;
@group(0) @binding(7) var<storage, read> offsetsAndList: array<i32>;
@group(0) @binding(8) var<storage, read_write> touched: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> guidingBuffer: array<f32>;
@group(0) @binding(10) var<storage, read_write> surfelDepth: array<vec4f>;
@group(0) @binding(11) var diffuseTex: texture_2d_array<f32>;
@group(0) @binding(12) var diffuseTexSampler: sampler;
@group(0) @binding(13) var envTexture: texture_2d<f32>;
@group(0) @binding(14) var envSampler: sampler;
@group(0) @binding(15) var blueNoiseTex: texture_2d<f32>;

#include "Common/bvh_traverse.wgsl"
fn load_radial_depth(index: u32) -> vec4f { return surfelDepth[index]; }
#include "Common/radial_depth.wgsl"

fn update_surfel_depth2(
    surfelIndex: u32,
    uvIn: vec2f,
    dist: f32,
    maxDepth: f32,
    dirLocal: vec3f,
) {
    let uv = clamp(uvIn, vec2f(0.0), vec2f(0.999));
    let tu = u32(SURFEL_DEPTH_TEXELS);
    let base = surfel_depth_base_index(surfelIndex);

    let p = uv * f32(SURFEL_DEPTH_TEXELS) - vec2f(0.5);
    var pxOffsetI = vec2i(floor(p + vec2f(0.5)));
    let maxI = i32(SURFEL_DEPTH_TEXELS - 1u);
    pxOffsetI = clamp(pxOffsetI, vec2i(0), vec2i(maxI));
    let pxOffset = vec2u(u32(pxOffsetI.x), u32(pxOffsetI.y));

    let idx = base + pxOffset.y * tu + pxOffset.x;
    let prev = surfelDepth[idx];
    var next: vec4f;

    let d = max(dist, 1e-4);
    let d2 = d * d;
    let d3 = d2 * d;
    let d4 = d2 * d2;
    let sample = vec4f(d, d2, d3, d4);

    if (prev.w == 0.0) {
        let m0 = max(maxDepth, d);
        let m2 = m0 * m0;
        let prior = vec4f(m0, m2, m2 * m0, m2 * m2);
        let alpha0 = select(0.05, 1.0, dist >= maxDepth * 0.99);
        next = prior + alpha0 * (sample - prior);
        next.y = max(next.y, next.x * next.x);
        next.w = max(next.w, next.y * next.y);
    } else {
        let kHit = 0.05;
        let kMiss = 0.01;
        let alpha = select(kHit, kMiss, dist > prev.x);
        next = prev + alpha * (sample - prev);

        next.y = max(next.y, next.x * next.x);
        next.w = max(next.w, next.y * next.y);
    }

    surfelDepth[idx] = next;
}

fn envEquirectUV(dirW: vec3f) -> vec2f {
    let d = normalize(dirW);
    let uCoord = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
    let vCoord = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2f(uCoord, vCoord);
}

fn sampleEnvEquirect(dirW: vec3f, lod: f32) -> vec3f {
    let uv = envEquirectUV(dirW);
    let hdr = textureSampleLevel(envTexture, envSampler, uv, lod).rgb;
    let lum = dot(hdr, vec3f(0.2126, 0.7152, 0.0722));
    let knee = 5.0;
    let maxVal = 15.0;
    if (lum <= knee) { return hdr; }
    let compressed = knee + (maxVal - knee) * (1.0 - exp(-(lum - knee) / (maxVal - knee)));
    return hdr * (compressed / lum);
}

fn calculate_luma(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn sampleDiffuseArray(uvIn: vec2f, layerIn: i32) -> vec3f {
    let layerCount = i32(textureNumLayers(diffuseTex));
    let layer = clamp(layerIn, 0, max(0, layerCount - 1));
    return textureSampleLevel(diffuseTex, diffuseTexSampler, uvIn, layer, 0.0).rgb;
}

fn blueNoise4(surfelIndex: u32, sampleIndex: u32, purpose: u32) -> vec4f {
    let combined = sampleIndex + purpose * 0x10000u;
    let seq = (combined * 4099u + surfelIndex * 7919u) & BLUE_NOISE_MASK;
    let x = seq & (BLUE_NOISE_SIZE - 1u);
    let y = seq >> 10u;
    let dims = textureDimensions(blueNoiseTex, 0);
    return textureLoad(blueNoiseTex, vec2u(x % dims.x, y % dims.y), 0);
}

fn getTangentBasis(normal: vec3f) -> mat3x3f {
    let nNormal = normalize(normal);
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(nNormal.z) < 0.999);
    let t = normalize(cross(up, nNormal));
    let b = cross(nNormal, t);
    return mat3x3f(t, b, nNormal);
}

fn sampleCosineHemisphereLocal(uRand: vec2f) -> vec3f {
    let r = sqrt(uRand.x);
    let theta = 2.0 * PI * uRand.y;
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - uRand.x));
    return vec3f(x, y, z);
}

fn slgSafeU01(x: f32) -> f32 {
    return clamp(x, 1e-6, 1.0 - 1e-6);
}

fn pdfSLG(surfelIndex: u32, uv: vec2f, slgMass: f32) -> f32 {
    if (slgMass <= 1e-6) { return 0.0; }
    let cx = min(u32(floor(uv.x * f32(SLG_DIM))), SLG_DIM - 1u);
    let cy = min(u32(floor(uv.y * f32(SLG_DIM))), SLG_DIM - 1u);
    let idx = cy * SLG_DIM + cx;
    let baseIdx = surfelIndex * SLG_TOTAL_FLOATS;
    let w = max(0.0, guidingBuffer[baseIdx + idx]);
    let J = hemiOctJacobian(uv);
    return (w / slgMass) * f32(SLG_LOBE_COUNT) / J;
}

fn slgGetLobeAxisLocal(idx: u32) -> vec3f {
    let x = f32(idx % SLG_DIM);
    let y = f32(idx / SLG_DIM);
    let uv = vec2f((x + 0.5) / f32(SLG_DIM), (y + 0.5) / f32(SLG_DIM));
    return hemiOctSquareDecode(uv);
}

struct MSMEData {
    mean: vec3f,
    shortMean: vec3f,
    vbbr: f32,
    variance: vec3f,
    inconsistency: f32,
};

fn runMSME(y: vec3f, dataIn: MSMEData, shortWindowBlend: f32) -> MSMEData {
    var data = dataIn;
    let dev = sqrt(max(vec3f(1e-5), data.variance));
    let highThreshold = vec3f(0.1) + data.shortMean + dev * 8.0;
    let yClamped = min(y, highThreshold);

    let delta = yClamped - data.shortMean;
    data.shortMean = mix(data.shortMean, yClamped, shortWindowBlend);
    let delta2 = yClamped - data.shortMean;

    let varianceBlend = shortWindowBlend * 0.5;
    data.variance = mix(data.variance, delta * delta2, varianceBlend);
    data.variance = max(data.variance, vec3f(0.0));

    let devNew = sqrt(max(vec3f(1e-5), data.variance));
    let shortDiff = data.mean - data.shortMean;
    let relativeDiff = dot(vec3f(0.299, 0.587, 0.114), abs(shortDiff) / max(vec3f(1e-5), devNew));
    data.inconsistency = mix(data.inconsistency, relativeDiff, 0.08);

    let term = (0.5 * data.shortMean) / max(vec3f(1e-5), devNew);
    let varianceBasedBlendReduction = clamp(dot(vec3f(0.299, 0.587, 0.114), term), 1.0 / 32.0, 1.0);

    let catchUpFactor = smoothstep(0.0, 1.0, relativeDiff * max(0.02, data.inconsistency - 0.2));
    var catchUpBlend = clamp(catchUpFactor, 1.0 / 256.0, 1.0);

    catchUpBlend *= data.vbbr;
    data.vbbr = mix(data.vbbr, varianceBasedBlendReduction, 0.1);
    data.mean = mix(data.mean, yClamped, clamp(catchUpBlend, 0.0, 1.0));
    return data;
}

fn slgSampleLobeIndex(surfelIndex: u32, uVal: f32, totalIn: f32) -> i32 {
    if (totalIn <= 1e-6) { return -1; }
    let baseIdx = surfelIndex * SLG_TOTAL_FLOATS;
    let rowSumOffset = baseIdx + SLG_LOBE_COUNT;

    var targ = clamp(uVal * totalIn, 0.0, totalIn);
    var row: u32 = 0u;
    for (var r: u32 = 0u; r < SLG_DIM; r = r + 1u) {
        let w = max(0.0, guidingBuffer[rowSumOffset + r]);
        if (targ <= w || r == SLG_DIM - 1u) { row = r; break; }
        targ -= w;
    }

    let rowStart = baseIdx + row * SLG_DIM;
    var col: u32 = 0u;
    for (var c: u32 = 0u; c < SLG_DIM; c = c + 1u) {
        let w = max(0.0, guidingBuffer[rowStart + c]);
        if (targ <= w || c == SLG_DIM - 1u) { col = c; break; }
        targ -= w;
    }

    return i32(row * SLG_DIM + col);
}

fn slgUpdateFromSample(surfelIndex: u32, uv: vec2f, lum: f32) -> f32 {
    let gridPos = uv * f32(SLG_DIM) - 0.5;
    let basePos = floor(gridPos);
    let f = fract(gridPos);

    let baseIdx = surfelIndex * SLG_TOTAL_FLOATS;
    let rowSumOffset = baseIdx + SLG_LOBE_COUNT;
    let eta = LEARNING_RATE;
    var massDiff = 0.0;

    for (var dy: i32 = 0; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = 0; dx <= 1; dx = dx + 1) {
            let cx = i32(basePos.x) + dx;
            let cy = i32(basePos.y) + dy;
            if (cx >= 0 && cx < i32(SLG_DIM) && cy >= 0 && cy < i32(SLG_DIM)) {
                let wx = select(1.0 - f.x, f.x, dx == 1);
                let wy = select(1.0 - f.y, f.y, dy == 1);
                let w = wx * wy;
                let targ = lum * w;

                let idx = u32(cy) * SLG_DIM + u32(cx);
                let oldVal = guidingBuffer[baseIdx + idx];
                let newVal = mix(oldVal, targ, eta);
                guidingBuffer[baseIdx + idx] = newVal;

                let diff = newVal - oldVal;
                guidingBuffer[rowSumOffset + u32(cy)] += diff;
                massDiff += diff;
            }
        }
    }
    return massDiff;
}

fn sampleGuidedDirection(surfelIndex: u32, slgMass: f32, pGuide: f32, uRand: vec4f) -> SLGSample {
    let ux = slgSafeU01(uRand.x);
    let uy = slgSafeU01(uRand.y);
    let uz = slgSafeU01(uRand.z);
    let uw = slgSafeU01(uRand.w);

    var out: SLGSample;
    if (slgMass > 1e-6 && uw < pGuide) {
        let chosen = slgSampleLobeIndex(surfelIndex, uz, slgMass);
        if (chosen >= 0) {
            let col = u32(chosen) % SLG_DIM;
            let row = u32(chosen) / SLG_DIM;
            let cellUV = vec2f(
                (f32(col) + ux) / f32(SLG_DIM),
                (f32(row) + uy) / f32(SLG_DIM)
            );
            out.uv = cellUV;
            out.dirLocal = hemiOctSquareDecode(cellUV);
            return out;
        }
    }

    let dirLocal = sampleCosineHemisphereLocal(vec2f(ux, uy));
    out.dirLocal = dirLocal;
    out.uv = hemiOctSquareEncode(dirLocal);
    return out;
}

fn slgGetTotalMass(surfelIndex: u32) -> f32 {
    let baseIdx = surfelIndex * SLG_TOTAL_FLOATS;
    let rowSumOffset = baseIdx + SLG_LOBE_COUNT;
    var total = 0.0;
    for (var r: u32 = 0u; r < SLG_DIM; r = r + 1u) {
        total += max(0.0, guidingBuffer[rowSumOffset + r]);
    }
    return total;
}

fn lookupSurfelGI(
    pt_ws: vec3f,
    normal_ws: vec3f,
    cam_pos: vec3f,
    grid_origin: vec3f,
    readOffset: u32,
    occParams: vec4f
) -> vec3f {
    let pRel = pt_ws - grid_origin;
    let gridCoord = surfel_pos_to_grid_coord(pRel);
    let c4 = surfel_grid_coord_to_c4(gridCoord);
    let hash = surfel_grid_c4_to_hash(c4);
    let cellIdx = i32(hash % TOTAL_CELLS);

    let start = offsetsAndList[cellIdx];
    let end = offsetsAndList[cellIdx + 1];
    // Visit the full stored list: atomic insertion order is not a relevance order.
    let listEnd = min(end, i32(arrayLength(&offsetsAndList)) - OFFSETS_AND_LIST_START);
    let count = max(listEnd - start, 0);
    if (count <= 0) {
        return vec3f(0.0);
    }

    var totalColor = vec3f(0.0);
    var totalWeight = 0.0;
    var bestContrib = 0.0;
    var bestSid: i32 = -1;

    for (var i: i32 = 0; i < count; i = i + 1) {
        let sid = offsetsAndList[OFFSETS_AND_LIST_START + start + i];
        if (sid < 0 || u32(sid) >= arrayLength(&surfels)) {
            continue;
        }

        let surfel = surfels[u32(sid)];
        if (surfel.age >= SURFEL_TTL) { continue; }
        let sPos = surfel.posb.xyz;
        let sNor = normalize(surfel.normal);

        let directional = dot(sNor, normal_ws);
        if (directional <= 0.0) {
            continue;
        }

        let sRad = surfel_radius_for_pos(sPos, cam_pos) * SURFEL_RADIUS_OVERSCALE;
        let eps = radius_based_epsilon(sRad);
        let sPosOff = sPos + sNor * eps;

        let dV = pt_ws - sPosOff;
        let distSq = dot(dV, dV);
        if (distSq >= sRad * sRad) {
            continue;
        }
        let dist = sqrt(distSq);
        let dirWS = dV / max(1e-6, dist);

        let align = abs(dot(dV, sNor));
        let mahal = dist * (1.0 + align * SURFEL_NORMAL_DIRECTION_SQUISH);
        var weight = (1.0 - smoothstep(0.0, sRad, mahal)) * directional;

        if (weight <= 0.0) {
            continue;
        }

        if (weight > 0.02) {
            let vis = surfel_radial_occlusion_rw(u32(sid), dirWS, sNor, dist, occParams);
            weight *= vis;
            if (weight <= 0.0) { continue; }
        }

        let readSid = u32(sid) + readOffset;
        let mRead = moments[readSid].irradiance;
        if (mRead.w <= 0.0) {
            continue;
        }
        let sIrr = mRead.xyz;
        weight *= clamp(mRead.w / TARGET_SAMPLE_COUNT, 0.05, 1.0);
        let contrib = sIrr * weight;
        let lenContrib = length(contrib);
        if (lenContrib > bestContrib) {
            bestContrib = lenContrib;
            bestSid = sid;
        }

        totalWeight += weight;
        totalColor += contrib;
    }

    if (bestSid >= 0 && bestContrib > 0.01) {
        let impMax = f32(SURFEL_IMPORTANCE_INDIRECT_MAX);
        let importance = clamp(i32(bestContrib * impMax), 0, 50);
        atomicMax(&touched[bestSid], importance);
    }

    if (totalWeight < 1e-5) { return vec3f(0.0); }
    return totalColor / totalWeight;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&surfels)) {
        return;
    }

    let s = surfels[index];
    if (s.age >= SURFEL_TTL) {
        return;
    }

    let basis = getTangentBasis(s.normal);
    let nW = basis[2];

    let readIdx = index + u.readOffset;
    let mPrev = moments[readIdx];

    var msmeState: MSMEData;
    msmeState.mean = mPrev.irradiance.xyz;
    msmeState.shortMean = mPrev.msmeData0.xyz;
    msmeState.vbbr = clamp(mPrev.msmeData0.w, 1.0 / 32.0, 1.0);
    msmeState.variance = max(mPrev.msmeData1.xyz, vec3f(0.0));
    msmeState.inconsistency = clamp(mPrev.msmeData1.w, 0.0, 10.0);

    let prevCount = mPrev.irradiance.w;
    let birthFrame = u32(s.posb.w);
    let sinceBirth = u.frame - birthFrame;

    // Preserve history for settled surfels between staggered updates.
    if (u._pad0 > 0.0 && sinceBirth > 32u && msmeState.inconsistency < 0.3 && ((index + u.frame) & 3u) != 0u) {
        moments[index + u.writeOffset] = mPrev;
        return;
    }
    var slgMass = slgGetTotalMass(index);
    let hasGrid = slgMass > 1e-5;
    var pGuide = select(0.0, 0.9, hasGrid);
    let guideRamp = clamp(f32(sinceBirth) / 16.0, 0.0, 1.0);
    pGuide = min(pGuide * guideRamp, 0.9);

    var diffuseGI = vec3f(0.0);
    var validSamples = 0.0;

    let baseCount = max(1u, u.baseSampleCount);
    let boostCount = select(0u, 12u, msmeState.inconsistency > 0.3);
    var sampleCount = baseCount + boostCount;
    if (sinceBirth <= 4u) {
        sampleCount = max(sampleCount, select(16u, 32u, sinceBirth <= 2u));
    }

    var hitPos0 = s.posb.xyz;
    var debugFlag = 0.0;

    let DEPTH_PROBE_STRIDE = 4u;
    let doProbe = (((index ^ u.frame) & (DEPTH_PROBE_STRIDE - 1u)) == 0u);

    var ray: Ray;
    let sRad = surfel_radius_for_pos(s.posb.xyz, u.camPos.xyz);
    let eps = radius_based_epsilon(sRad);
    ray.origin = s.posb.xyz + nW * eps;

    if (doProbe) {
        let binCount = SURFEL_DEPTH_TEXELS * SURFEL_DEPTH_TEXELS;
        let bin = (u.frame + index * 13u) % binCount;
        let bx = bin % SURFEL_DEPTH_TEXELS;
        let by = bin / SURFEL_DEPTH_TEXELS;

        let n4 = blueNoise4(index, u.frame, 0u);
        let uvDepth = vec2f(f32(bx) + n4.x, f32(by) + n4.y) / f32(SURFEL_DEPTH_TEXELS);
        let dirLocalDepth = hemiOctSquareDecode(uvDepth);
        ray.direction = basis * dirLocalDepth;

        let hitD = bvhTrace(ray, sRad * 2.0, false);
        let maxDepth = sRad * 2.0;
        let dHit = clamp(hitD.dist, 0.0, maxDepth);
        let dLearn = select(maxDepth, dHit, hitD.didHit);
        update_surfel_depth2(index, uvDepth, dLearn, maxDepth, dirLocalDepth);
    }

    for (var i = 0u; i < sampleCount; i = i + 1u) {
        let u4 = blueNoise4(index, u.frame * BLUE_NOISE_STRIDE + i, 1u);
        let slgS = sampleGuidedDirection(index, slgMass, pGuide, u4);
        let dirLocal = slgS.dirLocal;
        let uv = slgS.uv;

        let cosTerm = max(0.0, dirLocal.z);
        let pdfCos = cosTerm / PI;

        var pdfGuide = 0.0;
        if (pGuide > 0.0) {
            pdfGuide = pdfSLG(index, uv, slgMass);
        }

        let mixPdf = (1.0 - pGuide) * pdfCos + pGuide * pdfGuide;
        if (mixPdf > 1e-6 && cosTerm > 0.0) {
            debugFlag = 1.0;
            let rayDir = basis * dirLocal;
            ray.direction = rayDir;
            let hit = bvhIntersectFirstHit(ray);
            var bounceLi = vec3f(0.0);

            let maxDepth = sRad * 2.0;
            let dHit = clamp(hit.dist, 0.0, maxDepth);
            let dLearn = select(maxDepth, dHit, hit.didHit);
            update_surfel_depth2(index, uv, dLearn, maxDepth, dirLocal);

            if (hit.didHit) {
                let hitPoint = ray.origin + ray.direction * hit.dist;
                let hitNormal = normalize(hit.normal);
                if (i == 0u) { hitPos0 = hitPoint; }

                let uvMat = getVertexAttribute(hit.barycoord, hit.indices.xyz);
                let matId = i32(round(uvMat.z));
                let hitUv = uvMat.xy;
                var hitAlbedo = sampleDiffuseArray(hitUv, matId);
                let y = max(1e-4, dot(hitAlbedo, vec3f(0.2126, 0.7152, 0.0722)));
                let y2 = 1.0 - pow(1.0 - y, max(0.0, u.albedoBoost));
                let scale = y2 / y;
                hitAlbedo = clamp(hitAlbedo * scale, vec3f(0.0), vec3f(1.0));

                var shadowRay2: Ray;
                shadowRay2.origin = hitPoint + hitNormal * eps;
                shadowRay2.direction = u.lightDir.xyz;
                let NdotL2 = max(0.0, dot(hitNormal, u.lightDir.xyz));
                if (NdotL2 > 0.0 && u.giFromDirect > 0.0 && !bvhOccluded(shadowRay2)) {
                    bounceLi += u.lightColor.xyz * hitAlbedo * NdotL2 * (1.0 / PI) * u.giFromDirect;
                }
                let gi = lookupSurfelGI(hitPoint, hitNormal, u.camPos.xyz, u.gridOrigin.xyz, u.readOffset, u.occParams);
                bounceLi += gi * hitAlbedo * u.giFromIndirect;
            } else {
                bounceLi = sampleEnvEquirect(rayDir, u.envLod) * u.envIntensity;
            }

            diffuseGI += bounceLi * (cosTerm / PI) / mixPdf;
            let lum = calculate_luma(bounceLi) * cosTerm;
            let slgMassDiff = slgUpdateFromSample(index, uv, lum);
            slgMass += slgMassDiff;
            validSamples += 1.0;
        }
    }

    slgMass = max(0.0, slgMass);
    let newAvg = diffuseGI / max(1.0, validSamples);

    if (validSamples > 0.0) {
        let tau = 12.5;
        let shortWindowBlend = clamp(1.0 - exp(-validSamples / tau), 0.01, 0.10);

        if (prevCount < 32.0) {
            let blend = 1.0 / (1.0 + prevCount);
            msmeState.mean = mix(msmeState.mean, newAvg, blend);
            msmeState.shortMean = mix(msmeState.shortMean, newAvg, blend);
            msmeState.variance = mix(msmeState.variance, vec3f(1.0), blend);
            msmeState.vbbr = max(msmeState.vbbr, 1.0);
            msmeState.inconsistency = max(msmeState.inconsistency, 1.0);
        } else {
            msmeState = runMSME(newAvg, msmeState, shortWindowBlend);
        }
    }

    var totalCount = prevCount;
    if (validSamples > 0.0 && (dot(msmeState.mean, vec3f(1.0)) > 1e-5 || sinceBirth > 2u)) {
        totalCount = min(prevCount + 1.0, MAX_TEMPORAL_M);
    }

    var meanLocalN = vec3f(0.0, 0.0, 1.0);
    if (slgMass > 1e-5) {
        let baseIdx = index * SLG_TOTAL_FLOATS;
        var meanLocal = vec3f(0.0);
        for (var li: u32 = 0u; li < SLG_LOBE_COUNT; li = li + 1u) {
            let w = guidingBuffer[baseIdx + li];
            if (w <= 0.0) { continue; }
            let axisL = slgGetLobeAxisLocal(li);
            meanLocal += axisL * w;
        }
        let meanLen = length(meanLocal);
        if (meanLen > 1e-6) {
            meanLocalN = meanLocal / meanLen;
        }
    }
    let meanWorld = normalize(basis * meanLocalN);

    let outIdx = index + u.writeOffset;
    moments[outIdx].irradiance = vec4f(msmeState.mean, totalCount);
    moments[outIdx].msmeData0 = vec4f(msmeState.shortMean, msmeState.vbbr);
    moments[outIdx].msmeData1 = vec4f(msmeState.variance, msmeState.inconsistency);
    moments[outIdx].hit = vec4f(hitPos0, debugFlag);
    moments[outIdx].guiding = vec4f(meanWorld, slgMass);
}
