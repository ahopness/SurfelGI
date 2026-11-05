using System.Numerics;
using System.Runtime.InteropServices;
using SurfelGI.Core;

namespace SurfelGI.Scene;

[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct MeshVertex
{
    public Vector3 Position;
    public Vector3 Normal;
    public Vector2 UV;

    public MeshVertex(Vector3 position, Vector3 normal, Vector2 uv)
    {
        Position = position;
        Normal = normal;
        UV = uv;
    }
}

public class CpuTextureData
{
    public string Name { get; set; } = "";
    public int Width { get; set; }
    public int Height { get; set; }
    public byte[]? Rgba8Data { get; set; }
    public float[]? Rgba32FloatData { get; set; }
    public bool FlipY { get; set; }
}

public class MaterialData
{
    public string Uuid { get; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = "Default";
    public Vector3 BaseColorLinear { get; set; } = Vector3.One;
    public float Roughness { get; set; } = 0.8f;
    public float Metalness { get; set; } = 0.0f;
    public bool DoubleSided { get; set; }
    public CpuTextureData? DiffuseMap { get; set; }
    public GpuTexture? GpuDiffuseTexture { get; set; }

    public static Vector3 SrgbHexToLinear(uint hex)
    {
        float r = ((hex >> 16) & 0xFF) / 255f;
        float g = ((hex >> 8) & 0xFF) / 255f;
        float b = (hex & 0xFF) / 255f;
        return new Vector3(SrgbToLinear(r), SrgbToLinear(g), SrgbToLinear(b));
    }

    public static float SrgbToLinear(float c)
    {
        return c <= 0.04045f ? c / 12.92f : MathF.Pow((c + 0.055f) / 1.055f, 2.4f);
    }
}

public struct SubMeshGroup
{
    public int StartIndex;
    public int IndexCount;
    public int MaterialIndex;
}

public class CpuMesh
{
    public string Name { get; set; } = "";
    public MeshVertex[] Vertices { get; set; } = Array.Empty<MeshVertex>();
    public uint[] Indices { get; set; } = Array.Empty<uint>();
    public List<SubMeshGroup> Groups { get; } = new();
}

public class GpuMeshBuffers : IDisposable
{
    public GpuBuffer<MeshVertex> VertexBuffer { get; }
    public GpuBuffer<uint> IndexBuffer { get; }
    public int IndexCount { get; }

    public GpuMeshBuffers(WebGpuContext ctx, CpuMesh mesh)
    {
        VertexBuffer = new GpuBuffer<MeshVertex>(ctx, mesh.Vertices, Silk.NET.WebGPU.BufferUsage.Vertex, $"{mesh.Name}_VB");
        IndexBuffer = new GpuBuffer<uint>(ctx, mesh.Indices, Silk.NET.WebGPU.BufferUsage.Index, $"{mesh.Name}_IB");
        IndexCount = mesh.Indices.Length;
    }

    public void Dispose()
    {
        VertexBuffer.Dispose();
        IndexBuffer.Dispose();
    }
}

public class SceneMeshInstance
{
    public CpuMesh Mesh { get; set; } = null!;
    public GpuMeshBuffers? GpuBuffers { get; set; }
    public List<MaterialData> Materials { get; } = new();
    public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;
    public bool Visible { get; set; } = true;
    public bool CastShadow { get; set; } = true;
    public bool ReceiveShadow { get; set; } = true;
}

public class DirectionalLightData
{
    public Vector3 Color { get; set; } = Vector3.One;
    public float Intensity { get; set; } = 4.0f;
    public Vector3 Position { get; set; } = new(10f, 30f, 10f);
    public Vector3 Target { get; set; } = Vector3.Zero;
    public bool CastShadow { get; set; } = true;
    public int ShadowMapSize { get; set; } = 4096;
    public float ShadowBias { get; set; } = -0.0002f;
    public float ShadowNear { get; set; } = 0.1f;
    public float ShadowFar { get; set; } = 100f;
    public float OrthoExtent { get; set; } = 10f;
    public bool UseCsm { get; set; }
    public float CsmMaxFar { get; set; } = 50f;
    public float CsmLightMargin { get; set; } = 15f;

    public Vector3 GetLightDirectionToSource()
    {
        Vector3 d = Position - Target;
        return d.LengthSquared() > 1e-6f ? Vector3.Normalize(d) : Vector3.UnitY;
    }
}

public class SceneGraph : IDisposable
{
    public List<SceneMeshInstance> Meshes { get; } = new();
    public DirectionalLightData DirLight { get; set; } = new();
    public GpuTexture? EnvironmentTexture { get; set; }

    public void ClearMeshes()
    {
        foreach (var m in Meshes)
        {
            m.GpuBuffers?.Dispose();
            foreach (var mat in m.Materials)
            {
                mat.GpuDiffuseTexture?.Dispose();
                mat.GpuDiffuseTexture = null;
            }
        }
        Meshes.Clear();
    }

    public void ResetDirectionalLightDefaults()
    {
        DirLight = new DirectionalLightData
        {
            Color = Vector3.One,
            Intensity = 4.0f,
            Position = new Vector3(10f, 30f, 10f),
            Target = Vector3.Zero,
            CastShadow = true,
            ShadowMapSize = 4096,
            ShadowBias = -0.0002f,
            ShadowNear = 0.1f,
            ShadowFar = 100f,
            OrthoExtent = 10f,
            UseCsm = false,
            CsmMaxFar = 50f,
            CsmLightMargin = 15f
        };
    }

    public void Dispose()
    {
        ClearMeshes();
    }
}
