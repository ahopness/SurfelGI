struct GBufferUniforms {
    viewProj: mat4x4f,
    model: mat4x4f,
    normalMat: mat4x4f,
    baseColor: vec4f,
    hasTexture: u32,
    roughness: f32,
    metalness: f32,
    _pad0: f32,
};

@group(0) @binding(0) var<uniform> u: GBufferUniforms;
@group(0) @binding(1) var albedoTex: texture_2d<f32>;
@group(0) @binding(2) var albedoSampler: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
};

struct VertexOutput {
    @builtin(position) clipPos: vec4f,
    @location(0) worldNormal: vec3f,
    @location(1) uv: vec2f,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let worldPos = u.model * vec4f(in.position, 1.0);
    out.clipPos = u.viewProj * worldPos;
    out.worldNormal = normalize((u.normalMat * vec4f(in.normal, 0.0)).xyz);
    out.uv = in.uv;
    return out;
}

struct FragmentOutput {
    @location(0) normal: vec4f,
    @location(1) diffuseColor: vec4f,
};

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) isFront: bool) -> FragmentOutput {
    var out: FragmentOutput;
    var n = normalize(in.worldNormal);
    if (!isFront) {
        n = -n;
    }
    out.normal = vec4f(n * 0.5 + 0.5, u.roughness);

    var color = u.baseColor.rgb;
    if (u.hasTexture != 0u) {
        let texCol = textureSample(albedoTex, albedoSampler, vec2f(in.uv.x, 1.0 - in.uv.y)).rgb;
        color *= texCol;
    }
    out.diffuseColor = vec4f(color, 1.0);
    return out;
}

@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4f {
    let worldPos = u.model * vec4f(in.position, 1.0);
    return u.viewProj * worldPos;
}
