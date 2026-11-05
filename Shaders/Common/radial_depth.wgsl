fn hemiOctSquareEncode(d: vec3f) -> vec2f {
    let invL1 = 1.0 / max(1e-8, abs(d.x) + abs(d.y) + max(0.0, d.z));
    let p = d.xy * invL1;
    let q = vec2f(p.x + p.y, p.x - p.y);
    return q * 0.5 + 0.5;
}

fn hemiOctSquareDecode(uv: vec2f) -> vec3f {
    let q = uv * 2.0 - 1.0;
    let p = vec2f(q.x + q.y, q.x - q.y) * 0.5;
    let z = max(0.0, 1.0 - abs(p.x) - abs(p.y));
    return normalize(vec3f(p.x, p.y, z));
}

fn hemiOctJacobian(uv: vec2f) -> f32 {
    let q = uv * 2.0 - 1.0;
    let p = vec2f(q.x + q.y, q.x - q.y) * 0.5;
    let z = 1.0 - abs(p.x) - abs(p.y);
    let v = vec3f(p.x, p.y, z);
    let r2 = dot(v, v);
    let invR = inverseSqrt(max(1e-12, r2));
    return 2.0 * invR * invR * invR;
}

fn reduce_light_bleeding(visibility: f32, amount: f32) -> f32 {
    let a = clamp(amount, 0.0, 0.99);
    return clamp((visibility - a) / (1.0 - a), 0.0, 1.0);
}

fn surfel_depth_base_index(surfelIndex: u32) -> u32 {
    let t = u32(SURFEL_DEPTH_TEXELS);
    return surfelIndex * (t * t);
}

fn compute_surfel_depth_weight(
    mIn: vec4f,
    distIn: f32,
    cosThetaIn: f32,
    shadowStrengthIn: f32,
    bleedReductionIn: f32,
    grazingBiasScaleIn: f32,
    varianceBleedScaleIn: f32,
) -> f32 {
    if (mIn.w == 0.0) { return 1.0; }

    let zf = max(distIn, 1e-4);
    let m1 = max(0.0, mIn.x);
    let m2 = max(0.0, mIn.y);
    let m3 = max(0.0, mIn.z);
    let m4 = max(0.0, mIn.w);

    let s = max(1e-4, sqrt(m2));
    let invS = 1.0 / s;
    let invS2 = invS * invS;
    let invS3 = invS2 * invS;
    let invS4 = invS2 * invS2;

    let z = zf * invS;

    var b = vec4f(
        m1 * invS,
        m2 * invS2,
        m3 * invS3,
        m4 * invS4
    );

    let alpha = 2e-6;
    b = b * (1.0 - alpha) + vec4f(0.5) * alpha;

    let b1 = b.x;
    let b2 = b.y;
    let b3 = b.z;
    let b4 = b.w;

    let varN = max(0.0, b2 - b1 * b1);
    let varFactor = clamp(sqrt(varN), 0.0, 1.0);

    let cosTheta = clamp(cosThetaIn, 0.0, 1.0);
    let grazing = 1.0 - cosTheta;

    let bleed = clamp(
        bleedReductionIn
            + varianceBleedScaleIn * varFactor
            + grazingBiasScaleIn * grazing * varFactor,
        0.0,
        0.99
    );

    var shadow: f32;
    let eps = 1e-6;

    let l10 = b1;
    let l20 = b2;
    let d1 = max(eps, b2 - b1 * b1);
    let l21 = (b3 - b2 * b1) / d1;
    let d2 = max(eps, b4 - b2 * b2 - l21 * l21 * d1);

    let rhs0 = 1.0;
    let rhs1 = z;
    let rhs2 = z * z;

    let y0 = rhs0;
    let y1 = rhs1 - l10 * y0;
    let y2 = rhs2 - l20 * y0 - l21 * y1;

    let z0 = y0;
    let z1 = y1 / d1;
    let z2v = y2 / d2;

    let c2 = z2v;
    let c1 = z1 - l21 * c2;
    let c0 = z0 - l10 * c1 - l20 * c2;

    if (abs(c2) < 1e-6) {
        if (z <= b1) {
            shadow = 0.0;
        } else {
            let diff = z - b1;
            shadow = 1.0 - d1 / (d1 + diff * diff);
        }
    } else {
        let disc = max(0.0, c1 * c1 - 4.0 * c2 * c0);
        let sd = sqrt(disc);

        var z2r = (-c1 - sd) / (2.0 * c2);
        var z3r = (-c1 + sd) / (2.0 * c2);

        if (z2r > z3r) {
            let t = z2r;
            z2r = z3r;
            z3r = t;
        }

        if (abs(z3r - z2r) < 1e-6) {
            if (z <= b1) {
                shadow = 0.0;
            } else {
                let diff = z - b1;
                shadow = 1.0 - d1 / (d1 + diff * diff);
            }
        } else if (z <= z2r) {
            shadow = 0.0;
        } else if (z <= z3r) {
            let denom = (z3r - z2r) * max(eps, (z - z2r));
            shadow = (z * z3r - b1 * (z + z3r) + b2) / denom;
        } else {
            let denom = max(eps, (z - z2r) * (z - z3r));
            shadow = 1.0 - (z2r * z3r - b1 * (z2r + z3r) + b2) / denom;
        }
    }

    shadow = clamp(shadow, 0.0, 1.0);
    let shadowStrength = max(0.0, shadowStrengthIn);
    shadow = clamp(shadow * shadowStrength, 0.0, 1.0);

    var visibility = 1.0 - shadow;
    visibility = reduce_light_bleeding(visibility, bleed);
    return clamp(visibility, 0.0, 1.0);
}

fn linear_sample_radial_depth(surfelIndex: u32, uvIn: vec2f) -> vec4f {
    let uv = clamp(uvIn, vec2f(0.0), vec2f(0.999));
    let t = i32(SURFEL_DEPTH_TEXELS);
    let tu = u32(SURFEL_DEPTH_TEXELS);
    let base = surfel_depth_base_index(surfelIndex);

    let p = uv * f32(SURFEL_DEPTH_TEXELS) - vec2f(0.5);
    let i0 = vec2i(floor(p));
    let f = fract(p);
    let i1 = i0 + vec2i(1);

    let maxC = t - 1;
    let c0 = clamp(i0, vec2i(0), vec2i(maxC));
    let c1 = clamp(i1, vec2i(0), vec2i(maxC));

    let idx00 = base + u32(c0.y) * tu + u32(c0.x);
    let idx10 = base + u32(c0.y) * tu + u32(c1.x);
    let idx01 = base + u32(c1.y) * tu + u32(c0.x);
    let idx11 = base + u32(c1.y) * tu + u32(c1.x);

    let p00 = load_radial_depth(idx00);
    let p10 = load_radial_depth(idx10);
    let p01 = load_radial_depth(idx01);
    let p11 = load_radial_depth(idx11);

    let w00 = (1.0 - f.x) * (1.0 - f.y);
    let w10 = f.x * (1.0 - f.y);
    let w01 = (1.0 - f.x) * f.y;
    let w11 = f.x * f.y;

    let v00 = select(0.0, 1.0, p00.w != 0.0);
    let v10 = select(0.0, 1.0, p10.w != 0.0);
    let v01 = select(0.0, 1.0, p01.w != 0.0);
    let v11 = select(0.0, 1.0, p11.w != 0.0);

    let sw = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
    if (sw < 1e-8) { return vec4f(0.0); }

    return (p00 * w00 * v00 + p10 * w10 * v10 + p01 * w01 * v01 + p11 * w11 * v11) / sw;
}

fn point_sample_radial_depth(surfelIndex: u32, uvIn: vec2f) -> vec4f {
    let uv = clamp(uvIn, vec2f(0.0), vec2f(0.999));
    let tu = u32(SURFEL_DEPTH_TEXELS);
    let base = surfel_depth_base_index(surfelIndex);

    let px = vec2u(floor(uv * f32(SURFEL_DEPTH_TEXELS)));
    let x = min(px.x, tu - 1u);
    let y = min(px.y, tu - 1u);

    let idx = base + y * tu + x;
    return load_radial_depth(idx);
}

fn surfel_radial_occlusion(
    surfelIndex: u32,
    dirWS: vec3f,
    normalWS: vec3f,
    dist: f32,
    params: vec4f
) -> f32 {
    if (dist <= 0.0001) { return 0.0; }

    let n = normalize(normalWS);
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let t = normalize(cross(up, n));
    let b = cross(n, t);

    var hemi = vec3f(dot(dirWS, t), dot(dirWS, b), dot(dirWS, n));
    hemi.z = max(0.0, hemi.z);
    let uv = hemiOctSquareEncode(normalize(hemi));

    let m = linear_sample_radial_depth(surfelIndex, uv);
    let cosTheta = clamp(hemi.z, 0.0, 1.0);
    return compute_surfel_depth_weight(m, dist, cosTheta, params.x, params.y, params.z, params.w);
}

fn surfel_radial_occlusion_rw(
    surfelIndex: u32,
    dirWS: vec3f,
    normalWS: vec3f,
    dist: f32,
    params: vec4f
) -> f32 {
    if (dist <= 0.0001) { return 0.0; }

    let n = normalize(normalWS);
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let t = normalize(cross(up, n));
    let b = cross(n, t);

    var hemi = vec3f(dot(dirWS, t), dot(dirWS, b), dot(dirWS, n));
    hemi.z = max(0.0, hemi.z);
    let uv = hemiOctSquareEncode(normalize(hemi));

    let m = point_sample_radial_depth(surfelIndex, uv);
    let cosTheta = clamp(hemi.z, 0.0, 1.0);
    return compute_surfel_depth_weight(m, dist, cosTheta, params.x, params.y, params.z, params.w);
}
