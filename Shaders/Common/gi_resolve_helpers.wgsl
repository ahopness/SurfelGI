fn surfel_resolve_confidence(geometryWeight: f32, confidentWeight: f32) -> f32 {
    let coverage = smoothstep(0.0, 0.1, geometryWeight);
    // Overlap increases geometric coverage, not the maturity of the lighting estimate.
    let confidence = clamp(confidentWeight / max(geometryWeight, 1e-5), 0.0, 1.0);
    return coverage * confidence;
}

// Probe RGB is premultiplied by its support. If probes are still initializing,
// retain the provisional surfel estimate for the part they cannot replace yet.
fn blend_surfel_probe(surfelLight: vec3f, surfelConfidence: f32, fallback: vec4f) -> vec4f {
    let missing = 1.0 - surfelConfidence;
    let probeShare = missing * fallback.w;
    let light = surfelLight * (1.0 - probeShare) + fallback.rgb * missing;
    return vec4f(light, surfelConfidence + probeShare);
}
