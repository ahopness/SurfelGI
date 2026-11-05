const SURFEL_NORMAL_DIRECTION_SQUISH: f32 = 2.0;
const SURFEL_LIFE_RECYCLE: i32 = 0x8000000;
const SURFEL_LIFE_RECYCLED: i32 = 0x8000001;
const SURFEL_CS: i32 = 32;
const SURFEL_CS_U: u32 = 32u;
const SURFEL_CASCADES: i32 = 8;
const SURFEL_GRID_CELL_DIAMETER: f32 = 0.2;
const SURFEL_TTL: i32 = 500;
const SURFEL_BASE_RADIUS: f32 = 0.24;
const MAX_SURFELS: u32 = 262144u;
const MAX_SURFELS_PER_CELL: i32 = 64;
const MAX_SURFELS_PER_CELL_FOR_KEEP_ALIVE: i32 = 32;
const TOTAL_CELLS: u32 = 262144u;
const TOTAL_CELLS_I: i32 = 262144;
const SURFEL_RADIUS_OVERSCALE: f32 = 1.25;

const SLG_DIM: u32 = 8u;
const SLG_LOBE_COUNT: u32 = 64u;
const SLG_TOTAL_FLOATS: u32 = 72u;

const SURFEL_MAX_HEALTH: i32 = 100;
const SURFEL_KILL_SIGNAL: i32 = 255;
const SURFEL_IMPORTANCE_INDIRECT_MAX: i32 = 50;
const SURFEL_IMPORTANCE_DIRECT_MAX: i32 = 100;

const SURFEL_DEPTH_TEXELS: u32 = 4u;
const SURFEL_DEPTH_DEFAULT: f32 = 10.0;

const FADE_FRAMES: f32 = 4.0;
const TARGET_SAMPLE_COUNT: f32 = 32.0;
const OFFSETS_AND_LIST_START: i32 = 262145;

const MAX_TEMPORAL_M: f32 = 200.0;
const BLUE_NOISE_SIZE: u32 = 1024u;
const BLUE_NOISE_MASK: u32 = 1048575u;
const BLUE_NOISE_STRIDE: u32 = 64u;
const LEARNING_RATE: f32 = 0.02;
const PI: f32 = 3.141592653589793238462;

struct SurfelPacked {
    posb: vec4f,
    normal: vec3f,
    age: i32,
};

struct SurfelMoments {
    irradiance: vec4f,
    msmeData0: vec4f,
    msmeData1: vec4f,
    guiding: vec4f,
    hit: vec4f,
};

struct SLGSample {
    dirLocal: vec3f,
    uv: vec2f,
};

fn radius_based_epsilon(sRad: f32) -> f32 {
    return clamp(sRad * 0.01, 0.0005, 0.01);
}
