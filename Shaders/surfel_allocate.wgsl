#include "Common/constants.wgsl"
#include "Common/vertex_packed.wgsl"
#include "Common/grid_helpers.wgsl"
#include "Common/hash_utils.wgsl"

struct AllocateUniforms {
    readOffset: u32,
    tileWidth: u32,
    frame: u32,
    tileCount: u32,
    cameraPosition: vec4f,
};

@group(0) @binding(0) var<uniform> u: AllocateUniforms;
@group(0) @binding(1) var<storage, read_write> poolAlloc: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> poolMax: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> aliveAtomic: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read> poolBuf: array<i32>;
@group(0) @binding(5) var<storage, read_write> surfels: array<SurfelPacked>;
@group(0) @binding(6) var<storage, read_write> moments: array<SurfelMoments>;
@group(0) @binding(7) var<storage, read_write> guiding: array<f32>;
@group(0) @binding(8) var<storage, read_write> surfelDepth: array<vec4f>;
@group(0) @binding(9) var<storage, read> tileAlloc: array<i32>;
@group(0) @binding(10) var<storage, read> candPacked: array<VertexPacked>;
@group(0) @binding(11) var<storage, read> tileIrradiance: array<vec4f>;

@group(0) @binding(12) var<storage,read_write> touched: array<atomic<i32>>;

// Candidates all saw the previous grid. Resolve local conflicts before claiming
// a pool slot so adjacent tiles do not plant multiple surfels on the same patch.
fn overlaps_preferred_candidate(idx: u32, v: Vertex) -> bool {
    let width = i32(u.tileWidth);
    let tile = vec2i(i32(idx) % width, i32(idx) / width);
    let priority = hash1(idx ^ u.frame);
    let radius = surfel_radius_for_pos(v.position, u.cameraPosition.xyz);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let t = tile + vec2i(dx, dy);
            if (t.x < 0 || t.y < 0 || t.x >= width) { continue; }
            let j = u32(t.y * width + t.x);
            if (j >= u.tileCount || tileAlloc[j * 2u] != 1) { continue; }
            let otherPriority = hash1(j ^ u.frame);
            if (otherPriority > priority || (otherPriority == priority && j > idx)) { continue; }
            let other = unpack_vertex(candPacked[j]);
            let offset = other.position - v.position;
            if (dot(v.normal, other.normal) > 0.96 && length(offset) < radius * 0.30 &&
                abs(dot(offset, v.normal)) < radius * 0.04) { return true; }
        }
    }
    return false;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= u.tileCount) {
        return;
    }

    let capacity = i32(arrayLength(&surfels));
    let base = idx * 2u;
    let spawnFlag = tileAlloc[base];

    if (spawnFlag == 1) {
        let candidate = unpack_vertex(candPacked[idx]);
        if (overlaps_preferred_candidate(idx, candidate)) { return; }
        let prevAlloc = atomicAdd(&poolAlloc[0], 1);
        if (prevAlloc < capacity) {
            let surfelIdx = poolBuf[prevAlloc];
            atomicMax(&poolMax[0], surfelIdx + 1);

            let norm = normalize(candidate.normal);

            surfels[surfelIdx].posb = vec4f(candidate.position, f32(u.frame));
            surfels[surfelIdx].normal = norm;
            surfels[surfelIdx].age = 0;
            atomicStore(&touched[surfelIdx], 0);

            let tileTexels = SURFEL_DEPTH_TEXELS * SURFEL_DEPTH_TEXELS;
            let depthBase = u32(surfelIdx) * tileTexels;
            for (var i = 0u; i < tileTexels; i = i + 1u) {
                surfelDepth[depthBase + i] = vec4f(0.0);
            }

            let parentSid = tileAlloc[base + 1u];
            let childBase = u32(surfelIdx) * SLG_TOTAL_FLOATS;
            let tileGI = tileIrradiance[idx];
            let historySlot = u32(surfelIdx) + u.readOffset;

            var transplanted = false;
            if (parentSid >= 0 && parentSid < capacity && parentSid != surfelIdx) {
                let parentSurfel = surfels[parentSid];
                if (parentSurfel.age < SURFEL_TTL && dot(parentSurfel.normal, norm) > 0.95) {
                    transplanted = true;
                    let parentBase = u32(parentSid) * SLG_TOTAL_FLOATS;
                    for (var i = 0u; i < SLG_TOTAL_FLOATS; i = i + 1u) {
                        guiding[childBase + i] = guiding[parentBase + i];
                    }

                    let parentHistorySlot = u32(parentSid) + u.readOffset;
                    let parentIrr = moments[parentHistorySlot].irradiance;
                    let seedColor = select(parentIrr.xyz, tileGI.xyz, tileGI.w > 1e-5);
                    let priorCount = clamp(parentIrr.w, 1.0, 4.0);

                    moments[historySlot].irradiance = vec4f(seedColor, priorCount);
                    moments[historySlot].msmeData0 = vec4f(seedColor, 1.0);
                    moments[historySlot].msmeData1 = vec4f(1.0);
                }
            }

            if (!transplanted) {
                for (var i = 0u; i < SLG_TOTAL_FLOATS; i = i + 1u) {
                    guiding[childBase + i] = 0.0;
                }
                let priorCount = select(0.0, 2.0, tileGI.w > 1e-5);
                moments[historySlot].irradiance = vec4f(tileGI.xyz, priorCount);
                moments[historySlot].msmeData0 = vec4f(tileGI.xyz, 1.0);
                moments[historySlot].msmeData1 = vec4f(1.0);
            }
        } else {
            atomicAdd(&poolAlloc[0], -1);
        }
    }
}

@compute @workgroup_size(1, 1, 1)
fn sync_alive() {
    let allocNow = atomicLoad(&poolAlloc[0]);
    atomicStore(&aliveAtomic[0], allocNow);
}
