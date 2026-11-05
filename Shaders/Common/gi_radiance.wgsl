// Compress rare HDR sun samples before temporal integration, preserving chromaticity.
// Shared by both caches so switching GI sources does not change the radiance scale.
fn gi_environment_radiance(hdr: vec3f) -> vec3f {
    let color = max(hdr, vec3f(0.0));
    let lum = dot(color, vec3f(0.2126, 0.7152, 0.0722));
    let knee = 5.0;
    let maxVal = 15.0;
    if (lum <= knee) { return color; }
    let compressed = knee + (maxVal - knee) * (1.0 - exp(-(lum - knee) / (maxVal - knee)));
    return color * (compressed / lum);
}
