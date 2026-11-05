using System.Numerics;
using System.Runtime.InteropServices;

namespace SurfelGI;

public static class Constants
{
    public const float SURFEL_NORMAL_DIRECTION_SQUISH = 2.0f;
    public const int SURFEL_LIFE_RECYCLE = 0x8000000;
    public const int SURFEL_LIFE_RECYCLED = SURFEL_LIFE_RECYCLE + 1;
    public const int SURFEL_CS = 32;
    public const float SURFEL_GRID_CELL_DIAMETER = 0.2f;
    public const int SURFEL_TTL = 500;

    public const int CASCADES = 8;
    public const float SURFEL_BASE_RADIUS = 0.24f;
    public const int MAX_SURFELS = 262144;
    public const int MAX_SURFELS_PER_CELL = 64;
    public const int MAX_SURFELS_PER_CELL_FOR_KEEP_ALIVE = 32;
    public const int TOTAL_CELLS = SURFEL_CS * SURFEL_CS * SURFEL_CS * CASCADES; // 262144
    public const float SURFEL_RADIUS_OVERSCALE = 1.25f;

    public const int SLG_DIM = 8;
    public const int SLG_LOBE_COUNT = SLG_DIM * SLG_DIM; // 64
    public const int SLG_TOTAL_FLOATS = SLG_LOBE_COUNT + SLG_DIM; // 72

    public const int SURFEL_MAX_HEALTH = 100;
    public const int SURFEL_KILL_SIGNAL = 255;
    public const int SURFEL_IMPORTANCE_INDIRECT_MAX = 50;
    public const int SURFEL_IMPORTANCE_DIRECT_MAX = 100;

    public const int SURFEL_DEPTH_TEXELS = 4;
    public const float SURFEL_DEPTH_DEFAULT = 10.0f;
    public const float SURFEL_DEPTH_DEFAULT2 = SURFEL_DEPTH_DEFAULT * SURFEL_DEPTH_DEFAULT;

    public const int FADE_FRAMES = 4;
    public const int TARGET_SAMPLE_COUNT = 32;

    public const int OFFSETS_AND_LIST_START = TOTAL_CELLS + 1;

    public static Vector3 SnapToSurfelGridOrigin(Vector3 pos)
    {
        float s = SURFEL_GRID_CELL_DIAMETER;
        return new Vector3(
            MathF.Floor(pos.X / s) * s + 0.001f,
            MathF.Floor(pos.Y / s) * s + 0.001f,
            MathF.Floor(pos.Z / s) * s + 0.001f
        );
    }
}

/// <summary>
/// Matches WGSL struct SurfelPacked (32 bytes):
/// posb: vec4f (offset 0)
/// normal: vec3f (offset 16)
/// age: i32 (offset 28)
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 32)]
public struct SurfelPacked
{
    [FieldOffset(0)]
    public Vector4 PosB;

    [FieldOffset(16)]
    public Vector3 Normal;

    [FieldOffset(28)]
    public int Age;
}

/// <summary>
/// Matches WGSL struct SurfelMoments (80 bytes):
/// irradiance: vec4f
/// msmeData0: vec4f
/// msmeData1: vec4f
/// guiding: vec4f
/// hit: vec4f
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 80)]
public struct SurfelMoments
{
    [FieldOffset(0)]
    public Vector4 Irradiance;

    [FieldOffset(16)]
    public Vector4 MsmeData0;

    [FieldOffset(32)]
    public Vector4 MsmeData1;

    [FieldOffset(48)]
    public Vector4 Guiding;

    [FieldOffset(64)]
    public Vector4 Hit;
}

/// <summary>
/// Matches WGSL struct BVHNode (32 bytes):
/// bounds.min: array&lt;f32, 3&gt; (offset 0)
/// bounds.max: array&lt;f32, 3&gt; (offset 12)
/// rightChildOrTriangleOffset: u32 (offset 24)
/// splitAxisOrTriangleCount: u32 (offset 28)
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 32)]
public struct BvhNodeGpu
{
    [FieldOffset(0)]
    public Vector3 BoundsMin;

    [FieldOffset(12)]
    public Vector3 BoundsMax;

    [FieldOffset(24)]
    public uint RightChildOrTriangleOffset;

    [FieldOffset(28)]
    public uint SplitAxisOrTriangleCount;
}
