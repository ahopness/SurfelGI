const PROBE_LEVELS: u32 = 5u;
const PROBE_SIDE: i32 = 16;
const PROBES_PER_LEVEL: u32 = 4096u;
struct FallbackProbe {
    key: vec4i,
    offset: vec4f, // xyz relocation; w valid accumulated updates
    light: array<vec4f, 6>, // diffuse irradiance / pi
    depth: array<vec4f, 6>, // mean, mean-square
};
fn probe_spacing(level: u32) -> f32 { return 1.0 * f32(1u << level); }
fn probe_slot(cell: vec3i, level: u32) -> u32 {
    let c = vec3u(cell & vec3i(15));
    return level * PROBES_PER_LEVEL + c.x + c.y * 16u + c.z * 256u;
}
fn probe_axis(lobe: u32) -> vec3f {
    var axis = vec3f(0.0);
    axis[lobe / 2u] = select(1.0, -1.0, (lobe & 1u) != 0u);
    return axis;
}
fn probe_direction_lobe(d: vec3f) -> u32 {
    let a = abs(d);
    var axis = 0u;
    if (a.y > a.x) { axis = 1u; }
    if (a.z > a[axis]) { axis = 2u; }
    return axis * 2u + select(0u, 1u, d[axis] < 0.0);
}
