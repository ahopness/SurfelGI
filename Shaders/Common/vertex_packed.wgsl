struct VertexPacked {
    data0: vec4f,
};

struct Vertex {
    position: vec3f,
    normal: vec3f,
};

fn pack_unorm(val: f32, bitCount: u32) -> u32 {
    let maxVal = (1u << bitCount) - 1u;
    return u32(clamp(val, 0.0, 1.0) * f32(maxVal));
}

fn pack_normal_11_10_11(n: vec3f) -> f32 {
    var pckd = 0u;
    pckd += pack_unorm(n.x * 0.5 + 0.5, 11u);
    pckd += pack_unorm(n.y * 0.5 + 0.5, 10u) << 11u;
    pckd += pack_unorm(n.z * 0.5 + 0.5, 11u) << 21u;
    return bitcast<f32>(pckd);
}

fn unpack_unit_direction_11_10_11(pck: u32) -> vec3f {
    let mask11 = 2047u;
    let mask10 = 1023u;
    let div11 = 2047.0;
    let div10 = 1023.0;

    let x = f32(pck & mask11) * 2.0 / div11 - 1.0;
    let y = f32((pck >> 11u) & mask10) * 2.0 / div10 - 1.0;
    let z = f32(pck >> 21u) * 2.0 / div11 - 1.0;
    return vec3f(x, y, z);
}

fn pack_vertex(v: Vertex) -> VertexPacked {
    var p: VertexPacked;
    p.data0 = vec4f(v.position, pack_normal_11_10_11(v.normal));
    return p;
}

fn unpack_vertex(p: VertexPacked) -> Vertex {
    var res: Vertex;
    res.position = p.data0.xyz;
    res.normal = unpack_unit_direction_11_10_11(bitcast<u32>(p.data0.w));
    return res;
}
