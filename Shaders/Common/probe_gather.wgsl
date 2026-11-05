// Requires a read-only probes buffer and uniforms with camPos.
fn gatherProbeFallback(position: vec3f, normal: vec3f) -> vec4f {
    let cameraDistance = length(position - u.camPos.xyz);
    let startLevel = u32(clamp(floor(log2(max(1.0, cameraDistance / 6.0))), 0.0, 4.0));
    var result = vec3f(0.0);
    var remaining = 1.0;
    for (var level = startLevel; level < PROBE_LEVELS; level++) {
        let spacing = probe_spacing(level);
        let biased = position + normal * spacing * 0.08;
        let gridPosition = biased / spacing;
        let base = vec3i(floor(gridPosition));
        let fraction = fract(gridPosition);
        var light = vec3f(0.0);
        var total = 0.0;
        var support = 0.0;
        for (var corner = 0u; corner < 8u; corner++) {
            let c = vec3i(i32(corner & 1u), i32((corner >> 1u) & 1u), i32(corner >> 2u));
            let cell = base + c;
            let p = probes[probe_slot(cell, level)];
            if (p.offset.w <= 0.0 || any(p.key.xyz != cell)) { continue; }
            let tri = mix(vec3f(1.0) - fraction, fraction, vec3f(c));
            let delta = biased - (vec3f(cell) * spacing + p.offset.xyz);
            let distance = length(delta);
            let direction = delta / max(distance, 1e-5);
            let moments = p.depth[probe_direction_lobe(direction)].xy;
            let variance = max(moments.y - moments.x * moments.x, spacing * spacing * 0.05);
            let difference = max(0.0, distance - moments.x);
            // Deliberately permissive: this is a low-detail hole filler.
            let visibility = max(0.1, variance / (variance + difference * difference));
            // Prefer probes on the receiver's exposed side, without a hard leak cutoff.
            let facing = max(0.05, pow(clamp(dot(normal, -direction) * 0.5 + 0.5, 0.0, 1.0), 2.0));
            let maturity = min(1.0, p.offset.w * 0.5);
            let interpolation = tri.x * tri.y * tri.z * maturity;
            let weight = interpolation * visibility * facing;
            // The permissive visibility floor may contribute color, but cannot
            // claim reliable support and stop the search for a coarser estimate.
            support += interpolation * max(0.0, (visibility - 0.1) / 0.9);
            var irradiance = vec3f(0.0);
            let n2 = normal * normal;
            irradiance += p.light[select(0u, 1u, normal.x < 0.0)].rgb * n2.x;
            irradiance += p.light[select(2u, 3u, normal.y < 0.0)].rgb * n2.y;
            irradiance += p.light[select(4u, 5u, normal.z < 0.0)].rgb * n2.z;
            light += irradiance * weight;
            total += weight;
        }
        // Sparse or occluded corners must not become full-strength after normalization.
        // Fill their missing support with a coarser estimate instead of stopping here.
        let confidence = smoothstep(0.0, 0.5, support);
        let contribution = remaining * confidence;
        result += light / max(total, 1e-5) * contribution;
        remaining *= 1.0 - confidence;
        if (remaining < 0.01) { break; }
    }
    return vec4f(result, 1.0 - remaining);
}

