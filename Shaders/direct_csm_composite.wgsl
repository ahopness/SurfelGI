const PI: f32 = 3.141592653589793238462;

struct CompositeUniforms {
    projInv: mat4x4f,
    camWorld: mat4x4f,
    viewMat: mat4x4f,
    csmLightViewProj0: mat4x4f,
    csmLightViewProj1: mat4x4f,
    csmLightViewProj2: mat4x4f,
    csmLightViewProj3: mat4x4f,
    csmSplits: vec4f,
    lightDir: vec4f,
    lightColor: vec4f,
    camPos: vec4f,
    screenSize: vec2f,
    giMode: u32, // 0 = Combined, 1 = Direct, 2 = Indirect
    indirectIntensity: f32,
    shadowBias: f32,
    exposure: f32,
    csmEnabled: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> u: CompositeUniforms;
@group(0) @binding(1) var texDepth: texture_depth_2d;
@group(0) @binding(2) var texNormal: texture_2d<f32>;
@group(0) @binding(3) var texAlbedo: texture_2d<f32>;
@group(0) @binding(4) var texGi: texture_2d<f32>;
@group(0) @binding(5) var texEnv: texture_2d<f32>;
@group(0) @binding(6) var envSampler: sampler;
@group(0) @binding(7) var shadowMapArray: texture_depth_2d_array;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VSOut {
    var out: VSOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.pos = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2f(x, y);
    return out;
}

fn reconstruct_view_pos(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let clip = u.projInv * ndc;
    return clip.xyz / clip.w;
}

fn envEquirectUV(dirW: vec3f) -> vec2f {
    let d = normalize(dirW);
    let uCoord = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
    let vCoord = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2f(uCoord, vCoord);
}

fn shadowMatrix(c: i32) -> mat4x4f {
    switch c {
        case 0: { return u.csmLightViewProj0; }
        case 1: { return u.csmLightViewProj1; }
        case 2: { return u.csmLightViewProj2; }
        default: { return u.csmLightViewProj3; }
    }
}

fn filterShadow(c: i32, worldPos: vec3f, normal: vec3f, NdotL: f32) -> f32 {
    let m = shadowMatrix(c);
    let dims = vec2f(textureDimensions(shadowMapArray, 0));
    // Orthographic matrix row norms give world units per shadow texel.
    let sx = length(vec3f(m[0].x, m[1].x, m[2].x));
    let sy = length(vec3f(m[0].y, m[1].y, m[2].y));
    let texelWorld = 2.0 / (max(sx, sy) * dims.x);
    let offset = normal * texelWorld * 0.35 * sqrt(max(0.0, 1.0 - NdotL * NdotL));
    let clip = m * vec4f(worldPos + offset, 1.0);
    let p = clip.xyz / clip.w;
    let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || p.z < 0.0 || p.z > 1.0) { return 1.0; }
    let lightDepthScale = length(vec3f(m[0].z, m[1].z, m[2].z));
    let bias = lightDepthScale * texelWorld * (0.08 + 0.25 * (1.0 - NdotL));
    // Project the receiver plane into shadow UV/depth space. Neighboring PCF taps
    // compare against their own plane depth rather than increasing a constant bias.
    let axisX = normalize(vec3f(m[0].x,m[1].x,m[2].x));
    let axisY = normalize(vec3f(m[0].y,m[1].y,m[2].y));
    let axisZ = normalize(vec3f(m[0].z,m[1].z,m[2].z));
    let nz = dot(normal,axisZ);
    let safeNz = select(-1.0,1.0,nz >= 0.0)*max(abs(nz),0.1);
    let gradient = vec2f(-dot(normal,axisX)/safeNz*lightDepthScale/sx*2.0,
                         dot(normal,axisY)/safeNz*lightDepthScale/sy*2.0);
    var sum = 0.0;
    var total = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let w = f32((3 - abs(x)) * (3 - abs(y)));
            let samplePx = clamp(vec2i(uv * dims) + vec2i(x,y), vec2i(0), vec2i(dims) - vec2i(1));
            let d = textureLoad(shadowMapArray, samplePx, c, 0);
            let receiverDepth = p.z + dot(gradient,(vec2f(samplePx)+vec2f(0.5))/dims-uv);
            sum += w * select(0.0, 1.0, receiverDepth - bias <= d);
            total += w;
        }
    }
    return sum / total;
}

fn sampleShadowCascade(worldPos: vec3f, worldNormal: vec3f, viewDepth: f32, NdotL: f32) -> f32 {
    if (u.csmEnabled == 0u) { return 1.0; }
    var c = 0;
    if (viewDepth > u.csmSplits.x) { c = 1; }
    if (viewDepth > u.csmSplits.y) { c = 2; }
    if (viewDepth > u.csmSplits.z) { c = 3; }
    let visibility = filterShadow(c, worldPos, worldNormal, NdotL);
    if (c < 3) {
        let nearDepth = select(0.0, u.csmSplits[u32(max(c-1,0))], c > 0);
        let farDepth = u.csmSplits[u32(c)];
        let blend = smoothstep(mix(nearDepth, farDepth, 0.9), farDepth, viewDepth);
        if (blend > 0.0) { return mix(visibility, filterShadow(c+1, worldPos, worldNormal, NdotL), blend); }
    }
    return visibility;
}

fn reinhardToneMapping(colorIn: vec3f) -> vec3f {
    let color = max(colorIn * u.exposure, vec3f(0.0));
    return color / (vec3f(1.0) + color);
}

// Depth/normal/coverage guided reconstruction supports both full and half resolution GI.
fn gatherGi(uv: vec2f, normal: vec3f, viewDepth: f32) -> vec3f {
    let size = vec2i(textureDimensions(texGi));
    let p = uv * vec2f(size) - vec2f(0.5);
    let base = vec2i(floor(p)); let f = fract(p);
    let screen = vec2i(textureDimensions(texDepth));
    var light = vec3f(0.0); var total = 0.0;
    for(var y=0;y<2;y++) { for(var x=0;x<2;x++) {
        let tap = clamp(base + vec2i(x,y),vec2i(0),size-vec2i(1));
        let tapUv = (vec2f(tap)+vec2f(0.5))/vec2f(size);
        let gb = clamp(vec2i(tapUv*vec2f(screen)),vec2i(0),screen-vec2i(1));
        let d = textureLoad(texDepth,gb,0);
        if (d >= 1.0) { continue; }
        let z = -reconstruct_view_pos(tapUv,d).z;
        let n = normalize(textureLoad(texNormal,gb,0).xyz*2.0-1.0);
        let tapGi = textureLoad(texGi,tap,0);
        let covW = max(1e-3, tapGi.a);
        let bilinear = mix(vec2f(1.0)-f,f,vec2f(f32(x),f32(y)));
        let w = bilinear.x*bilinear.y*covW*pow(max(0.0,dot(n,normal)),16.0)*exp(-abs(z-viewDepth)/max(0.02,viewDepth*0.01));
        light += tapGi.rgb*w; total += w;
    }}
    if(total < 1e-6) { return textureLoad(texGi,clamp(vec2i(uv*vec2f(size)),vec2i(0),size-vec2i(1)),0).rgb; }
    return light / total;
}

fn evalPixelLinear(px: vec2i, uv: vec2f) -> vec3f {
    let depth = textureLoad(texDepth, px, 0);
    let viewPos = reconstruct_view_pos(uv, depth);
    let worldPos = (u.camWorld * vec4f(viewPos, 1.0)).xyz;

    if (depth >= 1.0) {
        let rayDir = normalize(worldPos - u.camPos.xyz);
        let envUv = envEquirectUV(rayDir);
        let sky = textureSampleLevel(texEnv, envSampler, envUv, 0.0).rgb;
        return reinhardToneMapping(sky);
    }

    let normRough = textureLoad(texNormal, px, 0);
    let N = normalize(normRough.xyz * 2.0 - 1.0);
    let roughness = clamp(normRough.w, 0.05, 1.0);
    let albedo = textureLoad(texAlbedo, px, 0).rgb;
    let giIrr = gatherGi(uv, N, -viewPos.z);

    let L = normalize(u.lightDir.xyz);
    let V = normalize(u.camPos.xyz - worldPos);
    let H = normalize(L + V);

    let NdotL = max(0.0, dot(N, L));
    let NdotH = max(0.0, dot(N, H));
    let shadow = select(0.0, sampleShadowCascade(worldPos, N, -viewPos.z, NdotL), NdotL > 0.0);

    // Diffuse + subtle GGX-like specular lobe matching Standard material
    let diff = albedo * (1.0 / PI);
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    let D = alpha2 / max(1e-4, PI * denom * denom);
    let spec = vec3f(0.04) * min(D * 0.25, 2.0);

    let directLight = (diff + spec) * u.lightColor.rgb * NdotL * shadow;
    let indirectLight = giIrr * albedo * u.indirectIntensity;

    var combined = directLight + indirectLight;
    if (u.giMode == 1u) {
        combined = directLight;
    } else if (u.giMode == 2u) {
        combined = indirectLight;
    }

    return reinhardToneMapping(combined);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    let dims = vec2i(i32(u.screenSize.x), i32(u.screenSize.y));
    let px = clamp(vec2i(in.uv * u.screenSize), vec2i(0), dims - vec2i(1));

    // Simple 5-tap FXAA-like edge smoothing
    let cM = evalPixelLinear(px, in.uv);
    return vec4f(cM, 1.0);
}
