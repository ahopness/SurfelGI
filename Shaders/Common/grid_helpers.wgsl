struct SurfelGridMinMax {
    c0_min: vec4u,
    c0_max: vec4u,
    c1_min: vec4u,
    c1_max: vec4u,
    cascadeCount: u32,
};

fn surfel_pos_to_grid_coord(pRel: vec3f) -> vec3i {
    return vec3i(floor(pRel / SURFEL_GRID_CELL_DIAMETER));
}

fn surfel_grid_coord_to_cascade_float(coord: vec3i) -> f32 {
    let fcoord = vec3f(coord) + vec3f(0.5);
    let max_c = max(abs(fcoord.x), max(abs(fcoord.y), abs(fcoord.z)));
    return log2(max_c / (f32(SURFEL_CS) * 0.5));
}

fn surfel_cascade_float_to_cascade(cf: f32) -> u32 {
    let v = ceil(max(0.0, cf));
    let clamped = clamp(v, 0.0, f32(SURFEL_CASCADES - 1));
    return u32(clamped);
}

fn surfel_grid_coord_within_cascade(coord: vec3i, cascade: u32) -> vec3i {
    return (coord >> vec3u(cascade)) + vec3i(SURFEL_CS / 2);
}

fn surfel_grid_coord_center(coord: vec4u, eye_pos: vec3f) -> vec3f {
    let gridPos = vec3f(coord.xyz) + vec3f(0.5) - vec3f(f32(SURFEL_CS) * 0.5);
    let posInCascade = gridPos * SURFEL_GRID_CELL_DIAMETER;
    let cascadeScale = f32(1u << coord.w);
    return eye_pos + posInCascade * cascadeScale;
}

fn surfel_grid_coord_to_c4(coord: vec3i) -> vec4u {
    let cf = surfel_grid_coord_to_cascade_float(coord);
    let cascade = surfel_cascade_float_to_cascade(cf);
    let ucoord = surfel_grid_coord_within_cascade(coord, cascade);
    let clamped = clamp(ucoord, vec3i(0), vec3i(SURFEL_CS - 1));
    return vec4u(u32(clamped.x), u32(clamped.y), u32(clamped.z), cascade);
}

fn surfel_grid_c4_to_hash(c4: vec4u) -> u32 {
    let cs = u32(SURFEL_CS);
    return c4.x + c4.y * cs + c4.z * cs * cs + c4.w * cs * cs * cs;
}

fn surfel_grid_coord_to_hash(coord: vec3i) -> u32 {
    return surfel_grid_c4_to_hash(surfel_grid_coord_to_c4(coord));
}

fn surfel_radius_for_pos(pos: vec3f, camPos: vec3f) -> f32 {
    let dist = length(pos - camPos);
    let cascadeRadius = SURFEL_GRID_CELL_DIAMETER * f32(SURFEL_CS) * 0.5;
    return SURFEL_BASE_RADIUS * max(1.0, dist / cascadeRadius);
}

fn surfel_intersects_grid_coord(
    surfelWorldPos: vec3f,
    normal: vec3f,
    radius: f32,
    c4: vec4u,
    gridOrigin: vec3f
) -> bool {
    let cellCenterWorld = surfel_grid_coord_center(c4, gridOrigin);
    let cascadeScale = f32(1u << c4.w);
    let cellRadius = SURFEL_GRID_CELL_DIAMETER * 0.5 * cascadeScale;

    let cellLocalPos = surfelWorldPos - cellCenterWorld;
    let closestPoint = clamp(cellLocalPos, vec3f(-cellRadius), vec3f(cellRadius));
    let posOffset = cellLocalPos - closestPoint;

    let distLen = length(posOffset);
    let dotNormal = abs(dot(posOffset, normal));
    let mahalanobisDist = distLen * (1.0 + dotNormal * SURFEL_NORMAL_DIRECTION_SQUISH);

    return mahalanobisDist < radius;
}

fn get_surfel_grid_box_min_max(pRel: vec3f) -> SurfelGridMinMax {
    let radius = surfel_radius_for_pos(pRel, vec3f(0.0)) * SURFEL_RADIUS_OVERSCALE;
    let boxMin = pRel - vec3f(radius);
    let boxMax = pRel + vec3f(radius);

    let gridMin = surfel_pos_to_grid_coord(boxMin);
    let gridMax = surfel_pos_to_grid_coord(boxMax);
    let centerCoord = surfel_pos_to_grid_coord(pRel);

    let cf = surfel_grid_coord_to_cascade_float(centerCoord);
    let c0 = surfel_cascade_float_to_cascade(cf - 0.2);
    let c1 = surfel_cascade_float_to_cascade(cf + 0.2);

    let limit = vec3i(SURFEL_CS - 1);
    let minC0 = clamp(surfel_grid_coord_within_cascade(gridMin, c0), vec3i(0), limit);
    let maxC0 = clamp(surfel_grid_coord_within_cascade(gridMax, c0), vec3i(0), limit);
    let minC1 = clamp(surfel_grid_coord_within_cascade(gridMin, c1), vec3i(0), limit);
    let maxC1 = clamp(surfel_grid_coord_within_cascade(gridMax, c1), vec3i(0), limit);

    var res: SurfelGridMinMax;
    res.c0_min = vec4u(u32(minC0.x), u32(minC0.y), u32(minC0.z), c0);
    res.c0_max = vec4u(u32(maxC0.x), u32(maxC0.y), u32(maxC0.z), c0);
    res.c1_min = vec4u(u32(minC1.x), u32(minC1.y), u32(minC1.z), c1);
    res.c1_max = vec4u(u32(maxC1.x), u32(maxC1.y), u32(maxC1.z), c1);
    res.cascadeCount = select(1u, 2u, c0 != c1);
    return res;
}

fn reconstruct_view_position(uv: vec2f, depth: f32, projInv: mat4x4f) -> vec3f {
    // WebGPU texture UV is Y-down [0,1]; NDC is X[-1,1], Y[-1,1] (Y-up), Z[0,1] (or [-1,1] if GL proj)
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let clip = projInv * ndc;
    return clip.xyz / clip.w;
}
