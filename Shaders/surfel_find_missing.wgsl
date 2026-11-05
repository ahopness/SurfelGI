#include "Common/constants.wgsl"
#include "Common/hash_utils.wgsl"
#include "Common/vertex_packed.wgsl"
#include "Common/grid_helpers.wgsl"

struct FindMissingUniforms {
    projInv: mat4x4f,
    camWorld: mat4x4f,
    camPos: vec4f,
    prevCamPos: vec4f,
    prevGridOrigin: vec4f,
    screenSize: vec2f,
    gridStride: u32,
    frame: u32,
    readOffset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> u: FindMissingUniforms;
@group(0) @binding(1) var texDepth: texture_depth_2d;
@group(0) @binding(2) var texNormal: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> surfels: array<SurfelPacked>;
@group(0) @binding(4) var<storage, read> offsetsAndList: array<i32>;
@group(0) @binding(5) var<storage, read> moments: array<SurfelMoments>;
@group(0) @binding(6) var<storage, read_write> touched: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> tileAlloc: array<i32>;
@group(0) @binding(8) var<storage, read_write> candPacked: array<VertexPacked>;
@group(0) @binding(9) var<storage, read_write> tileIrradiance: array<vec4f>;

var<workgroup> gs_px_max_score_loc_packed: atomic<u32>;
var<workgroup> gs_px_min_score_loc_packed: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wgId: vec3u,
    @builtin(local_invocation_id) localId: vec3u
) {
    let gx = wgId.x;
    let gy = wgId.y;
    let lx = localId.x;
    let ly = localId.y;

    let laneIndex = ly * 8u + lx;
    let tileIndex = gy * u.gridStride + gx;
    // Break ties between uncovered pixels without snapping every birth to a tile corner.
    let tieMask = hashCombine2(tileIndex,u.frame) & 63u;

    var worldPos = vec3f(0.0);
    var normal = vec3f(0.0, 0.0, 1.0);
    var flat: i32 = 0;

    var accumIrr = vec3f(0.0);
    var accumIrrWeight: f32 = 0.0;

    if (laneIndex == 0u) {
        let base = tileIndex * 2u;
        tileAlloc[base] = 0;
        tileAlloc[base + 1u] = -1;
        tileIrradiance[tileIndex] = vec4f(0.0);
        atomicStore(&gs_px_max_score_loc_packed, 0u);
        atomicStore(&gs_px_min_score_loc_packed, 0xffffffffu);
    }

    workgroupBarrier();

    let widthF = u.screenSize.x;
    let heightF = u.screenSize.y;
    let widthU = u32(widthF);
    let heightU = u32(heightF);

    let px = gx * 8u + lx;
    let py = gy * 8u + ly;
    let pixelInBounds = px < widthU && py < heightU;

    var bestParentSid: i32 = -1;
    var maxParentWeight: f32 = -1.0;

    if (pixelInBounds) {
        let uv = vec2f((f32(px) + 0.5) / widthF, (f32(py) + 0.5) / heightF);
        var seed = hashCombine2(hashCombine2(px, hash1(py)), u.frame);

        let depth = textureLoad(texDepth, vec2i(i32(px), i32(py)), 0);
        let valid = depth > 1e-6 && depth < 1.0;

        if (valid) {
            let viewPosRaw = reconstruct_view_position(uv, depth, u.projInv);
            let pt_depth = -viewPosRaw.z;
            let worldPos4 = u.camWorld * vec4f(viewPosRaw, 1.0);
            worldPos = worldPos4.xyz;

            let encN = textureLoad(texNormal, vec2i(i32(px), i32(py)), 0).xyz;
            normal = normalize(encN * 2.0 - 1.0);

            let pRelPrev = worldPos - u.prevGridOrigin.xyz;
            let gridCoord = surfel_pos_to_grid_coord(pRelPrev);
            let c4 = surfel_grid_coord_to_c4(gridCoord);
            let hash = surfel_grid_c4_to_hash(c4);
            flat = i32(hash);

            let startIdx = offsetsAndList[flat];
            let endIdx = offsetsAndList[flat + 1];
            let cnt = max(endIdx - startIdx, 0);
            // Coverage must not depend on the grid's atomic insertion order.
            let listEnd = min(endIdx, i32(arrayLength(&offsetsAndList)) - OFFSETS_AND_LIST_START);
            let loopCount = max(listEnd - startIdx, 0);

            var totalWeight: f32 = 0.0;
            var scoringWeight: f32 = 0.0;
            var highest: f32 = 0.0;
            var second: f32 = 0.0;

            for (var i: i32 = 0; i < loopCount; i = i + 1) {
                let sid = offsetsAndList[OFFSETS_AND_LIST_START + startIdx + i];
                if (sid < 0 || u32(sid) >= arrayLength(&surfels)) {
                    continue;
                }
                let packed = surfels[sid];
                if (packed.age >= SURFEL_TTL) { continue; }
                let posb = packed.posb;
                let surfelNormal = packed.normal;

                let dotN = dot(surfelNormal, normal);
                if (dotN <= 0.0) { continue; }

                let surfelRadius = surfel_radius_for_pos(posb.xyz, u.camPos.xyz);
                let maxRad = surfelRadius * SURFEL_RADIUS_OVERSCALE;
                let posOffset = worldPos - posb.xyz;
                let dSq = dot(posOffset, posOffset);
                if (dSq >= maxRad * maxRad) { continue; }

                let surfel_pRel = posb.xyz - u.prevGridOrigin.xyz;
                let surfel_c4_hash = surfel_grid_coord_to_hash(surfel_pos_to_grid_coord(surfel_pRel));
                let isSameHash = surfel_c4_hash == hash;

                let d = sqrt(dSq);
                let alignPenalty = abs(dot(posOffset, surfelNormal)) * SURFEL_NORMAL_DIRECTION_SQUISH;
                let mahal = d * (1.0 + alignPenalty);

                let weight = (1.0 - smoothstep(0.0, maxRad, mahal)) * dotN;
                let scoreW = (1.0 - smoothstep(0.0, surfelRadius, mahal)) * dotN;

                totalWeight += weight;
                scoringWeight += scoreW;

                let momentsIndex = u32(sid) + u.readOffset;
                let surfelIrr = moments[momentsIndex].irradiance;

                let isMature = (f32(u.frame) - posb.w) >= 2.0 && surfelIrr.w >= 1.0 && packed.age < (SURFEL_TTL - 16) && dotN > 0.9;
                let parentScore = weight * clamp(surfelIrr.w / 8.0, 0.2, 1.0);
                if (isMature && parentScore > maxParentWeight) {
                    bestParentSid = sid;
                    maxParentWeight = parentScore;
                }

                if (weight > highest) {
                    second = highest;
                    highest = weight;
                } else if (weight > second) {
                    second = weight;
                }

                if (surfelIrr.w > 0.0 && dot(surfelIrr.xyz, vec3f(1.0)) > 1e-5) {
                    let irrConf = clamp(surfelIrr.w / TARGET_SAMPLE_COUNT, 0.05, 1.0);
                    let irrW = weight * irrConf;
                    accumIrr += surfelIrr.xyz * irrW;
                    accumIrrWeight += irrW;
                }

                let sameCellKeep = cnt <= MAX_SURFELS_PER_CELL_FOR_KEEP_ALIVE && dotN > 0.8 && isSameHash;
                let isRecycle = packed.age == SURFEL_LIFE_RECYCLE;
                if (sameCellKeep && !isRecycle) {
                    atomicMax(&touched[sid], 5);
                }
            }

            let maxKeepF = f32(MAX_SURFELS_PER_CELL_FOR_KEEP_ALIVE);
            let fullness = smoothstep(maxKeepF * 0.75, maxKeepF * 1.0, f32(cnt));
            let despawnThreshold = mix(3.5, 3.0, fullness);
            let secondThresh = mix(0.9, 0.8, fullness);
            let shouldConsiderDespawn = scoringWeight > despawnThreshold && second > secondThresh;

            if (shouldConsiderDespawn) {
                let scoreBits = bitcast<u32>(scoringWeight);
                let packedVote = (scoreBits & 0xffffffc0u) | laneIndex;
                atomicMax(&gs_px_max_score_loc_packed, packedVote);
            }

            let gateCoverage = second < 0.4 && scoringWeight < 0.15 && cnt < MAX_SURFELS_PER_CELL_FOR_KEEP_ALIVE;
            let insideGrid = all(abs(pRelPrev) < vec3f(SURFEL_GRID_CELL_DIAMETER * f32(SURFEL_CS) * 0.5 * exp2(f32(SURFEL_CASCADES-1))));
            let wantSpawn = valid && insideGrid && gateCoverage;

            if (wantSpawn) {
                let scoreAsUint = bitcast<u32>(totalWeight);
                let packedVal = (scoreAsUint & 0xffffffc0u) | (laneIndex ^ tieMask);
                atomicMin(&gs_px_min_score_loc_packed, packedVal);
            }
        }
    }

    workgroupBarrier();

    let maxScorePacked = atomicLoad(&gs_px_max_score_loc_packed);
    if (maxScorePacked != 0u) {
        let winningLane = maxScorePacked & 63u;
        if (laneIndex == winningLane) {
            var victimIdx: i32 = -1;
            var maxVictimWeight: f32 = -1.0;

            let startIdx = offsetsAndList[flat];
            let endIdx = offsetsAndList[flat + 1];
            let listEnd = min(endIdx, i32(arrayLength(&offsetsAndList)) - OFFSETS_AND_LIST_START);
            let loopCount = max(listEnd - startIdx, 0);

            for (var i: i32 = 0; i < loopCount; i = i + 1) {
                let sid = offsetsAndList[OFFSETS_AND_LIST_START + startIdx + i];
                if (sid < 0 || u32(sid) >= arrayLength(&surfels)) {
                    continue;
                }
                let packed = surfels[sid];
                if (packed.age >= SURFEL_TTL) { continue; }
                let posb = packed.posb;
                let surfelNormal = packed.normal;

                let surfelRadius = surfel_radius_for_pos(posb.xyz, u.camPos.xyz);
                let posOffset = worldPos - posb.xyz;
                let d = length(posOffset);
                let alignPenalty = abs(dot(posOffset, surfelNormal)) * SURFEL_NORMAL_DIRECTION_SQUISH;
                let mahal = d * (1.0 + alignPenalty);
                let dirW = max(0.0, dot(surfelNormal, normal));

                let w = (1.0 - smoothstep(0.0, surfelRadius * SURFEL_RADIUS_OVERSCALE, mahal)) * dirW;
                if (w >= maxVictimWeight) {
                    maxVictimWeight = w;
                    victimIdx = sid;
                }
            }

            // Overlap at one pixel does not prove that two surfels are interchangeable.
            // Thin duplicates gradually and keep the oldest matching sample.
            if (victimIdx >= 0 && ((u32(victimIdx) + u.frame) & 15u) == 0u) {
                let victim = surfels[victimIdx];
                let radius = surfel_radius_for_pos(victim.posb.xyz, u.camPos.xyz);
                for (var j = 0; j < loopCount; j++) {
                    let otherId = offsetsAndList[OFFSETS_AND_LIST_START + startIdx + j];
                    if (otherId < 0 || u32(otherId) >= arrayLength(&surfels)) { continue; }
                    let other = surfels[otherId];
                    let older = other.posb.w < victim.posb.w ||
                        (other.posb.w == victim.posb.w && otherId < victimIdx);
                    let offset = other.posb.xyz - victim.posb.xyz;
                    if (older && other.age < SURFEL_TTL && dot(other.normal,victim.normal) > 0.97 &&
                        length(offset) < radius * 0.35 && abs(dot(offset,victim.normal)) < radius * 0.04) {
                        atomicMax(&touched[victimIdx], SURFEL_KILL_SIGNAL);
                        break;
                    }
                }
            }
        }
    }

    let bestPacked = atomicLoad(&gs_px_min_score_loc_packed);
    if (bestPacked != 0xffffffffu) {
        let winningLaneIndex = (bestPacked & 63u) ^ tieMask;
        if (laneIndex == winningLaneIndex) {
            let base = tileIndex * 2u;
            tileAlloc[base] = 1;
            tileAlloc[base + 1u] = bestParentSid;

            var vertex: Vertex;
            vertex.position = worldPos;
            vertex.normal = normal;
            candPacked[tileIndex] = pack_vertex(vertex);

            if (accumIrrWeight > 1e-5) {
                let avgIrr = accumIrr / accumIrrWeight;
                tileIrradiance[tileIndex] = vec4f(avgIrr, accumIrrWeight);
            }
        }
    }
}
