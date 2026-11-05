using System.Globalization;
using System.Numerics;
using Silk.NET.WebGPU;
using SurfelGI.Core;

namespace SurfelGI.Scene;

public class AssetLoader
{
    private readonly WebGpuContext _ctx;
    public string PublicRootDir { get; }
    private readonly Dictionary<string, (CpuTextureData Cpu, GpuTexture Gpu)> _envCache = new();

    public AssetLoader(WebGpuContext ctx, string? publicRootDir = null)
    {
        _ctx = ctx;
        PublicRootDir = publicRootDir ?? FindPublicDirectory();
    }

    internal static string FindPublicDirectory()
    {
        string? curr = Directory.GetCurrentDirectory();
        while (curr != null)
        {
            string candidate = Path.Combine(curr, "public");
            if (Directory.Exists(candidate)) return candidate;
            curr = Directory.GetParent(curr)?.FullName;
        }
        return Path.Combine(AppContext.BaseDirectory, "public");
    }

    public string ResolveAssetPath(string relativePath)
    {
        string normalized = relativePath.TrimStart('/', '\\');
        // Automatically substitute uncompressed sponza.glb for Sponza-Draco.glb
        if (normalized.EndsWith("Sponza-Draco.glb", StringComparison.OrdinalIgnoreCase))
        {
            string uncompressed = Path.Combine(PublicRootDir, "models", "sponza.glb");
            if (File.Exists(uncompressed))
                return uncompressed;
        }
        return Path.Combine(PublicRootDir, normalized);
    }

    public GpuTexture LoadBlueNoiseTexture(string relativePath = "textures/LDR_RGBA_0.png")
    {
        string fullPath = ResolveAssetPath(relativePath);
        using var stream = File.OpenRead(fullPath);
        byte[] rgba = GltfAndHdrReader.DecodeImageRgba8(stream, out int width, out int height);

        var tex = new GpuTexture(
            _ctx,
            width,
            height,
            TextureFormat.Rgba8Unorm,
            TextureUsage.TextureBinding | TextureUsage.CopyDst,
            filter: FilterMode.Nearest,
            addressMode: AddressMode.Repeat,
            label: "BlueNoise");

        tex.UploadLayer<byte>(rgba, 4);
        return tex;
    }

    public (CpuTextureData Cpu, GpuTexture Gpu) LoadEnvironmentTexture(string relativePath)
    {
        if (_envCache.TryGetValue(relativePath, out var cached))
            return cached;

        string fullPath = ResolveAssetPath(relativePath);
        string ext = Path.GetExtension(fullPath).ToLowerInvariant();

        CpuTextureData cpuData;
        if (ext == ".hdr")
        {
            cpuData = GltfAndHdrReader.LoadRadianceHdr(fullPath);
        }
        else
        {
            cpuData = ExrReader.Load(fullPath);
        }

        uint mipCount = (uint)Math.Floor(Math.Log2(Math.Max(cpuData.Width, cpuData.Height))) + 1u;

        // Use Rgba16Float so standard linear filtering Sampler works on all WebGPU adapters without float32-filterable feature
        var gpuTex = new GpuTexture(
            _ctx,
            cpuData.Width,
            cpuData.Height,
            TextureFormat.Rgba16Float,
            TextureUsage.TextureBinding | TextureUsage.CopyDst,
            mipLevelCount: mipCount,
            filter: FilterMode.Linear,
            addressMode: AddressMode.Repeat,
            label: cpuData.Name);

        float[] currentLevel = cpuData.Rgba32FloatData!;
        int mipW = cpuData.Width;
        int mipH = cpuData.Height;

        for (uint mip = 0; mip < mipCount; mip++)
        {
            ushort[] halfData = new ushort[mipW * mipH * 4];
            for (int i = 0; i < halfData.Length; i++)
            {
                float v = Math.Clamp(currentLevel[i], -65500f, 65500f);
                halfData[i] = BitConverter.HalfToUInt16Bits((Half)v);
            }
            gpuTex.UploadLayer<ushort>(halfData, 8, 0, mip, mipW, mipH);

            if (mip + 1 < mipCount)
            {
                int nextW = Math.Max(1, mipW / 2);
                int nextH = Math.Max(1, mipH / 2);
                float[] nextLevel = new float[nextW * nextH * 4];
                for (int y = 0; y < nextH; y++)
                {
                    for (int x = 0; x < nextW; x++)
                    {
                        int x0 = Math.Min(mipW - 1, x * 2);
                        int x1 = Math.Min(mipW - 1, x * 2 + 1);
                        int y0 = Math.Min(mipH - 1, y * 2);
                        int y1 = Math.Min(mipH - 1, y * 2 + 1);

                        int i00 = (y0 * mipW + x0) * 4;
                        int i10 = (y0 * mipW + x1) * 4;
                        int i01 = (y1 * mipW + x0) * 4;
                        int i11 = (y1 * mipW + x1) * 4;
                        int dst = (y * nextW + x) * 4;

                        for (int c = 0; c < 4; c++)
                        {
                            nextLevel[dst + c] = 0.25f * (
                                currentLevel[i00 + c] +
                                currentLevel[i10 + c] +
                                currentLevel[i01 + c] +
                                currentLevel[i11 + c]);
                        }
                    }
                }
                currentLevel = nextLevel;
                mipW = nextW;
                mipH = nextH;
            }
        }

        var bundle = (cpuData, gpuTex);
        _envCache[relativePath] = bundle;
        return bundle;
    }

    public List<SceneMeshInstance> LoadGltf(string relativePath, Matrix4x4 rootTransform, MaterialData? overrideMaterial = null)
    {
        string fullPath = ResolveAssetPath(relativePath);
        var parsedMeshes = GltfAndHdrReader.ParseGltfFile(fullPath, rootTransform, overrideMaterial);
        var instances = new List<SceneMeshInstance>(parsedMeshes.Count);

        foreach (var (cpuMesh, materials, worldMatrix) in parsedMeshes)
        {
            var inst = new SceneMeshInstance
            {
                Mesh = cpuMesh,
                GpuBuffers = new GpuMeshBuffers(_ctx, cpuMesh),
                WorldMatrix = worldMatrix
            };
            inst.Materials.AddRange(materials);
            instances.Add(inst);
        }

        return instances;
    }

    public CpuMesh LoadPlyNormalized(string relativePath, float targetRadius = 0.8f)
    {
        string fullPath = ResolveAssetPath(relativePath);
        using var reader = new StreamReader(fullPath);

        int vertexCount = 0;
        int faceCount = 0;
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            string trimmed = line.Trim();
            if (trimmed.StartsWith("element vertex ", StringComparison.Ordinal))
                vertexCount = int.Parse(trimmed.Substring("element vertex ".Length), CultureInfo.InvariantCulture);
            else if (trimmed.StartsWith("element face ", StringComparison.Ordinal))
                faceCount = int.Parse(trimmed.Substring("element face ".Length), CultureInfo.InvariantCulture);
            else if (trimmed == "end_header")
                break;
        }

        var positions = new Vector3[vertexCount];
        Vector3 minB = new(float.MaxValue);
        Vector3 maxB = new(float.MinValue);

        for (int i = 0; i < vertexCount; i++)
        {
            line = reader.ReadLine();
            if (line == null) break;
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            float x = float.Parse(parts[0], CultureInfo.InvariantCulture);
            float y = float.Parse(parts[1], CultureInfo.InvariantCulture);
            float z = float.Parse(parts[2], CultureInfo.InvariantCulture);
            Vector3 p = new(x, y, z);
            positions[i] = p;
            minB = Vector3.Min(minB, p);
            maxB = Vector3.Max(maxB, p);
        }

        Vector3 center = (minB + maxB) * 0.5f;
        float maxR2 = 0f;
        for (int i = 0; i < vertexCount; i++)
        {
            positions[i] -= center;
            maxR2 = MathF.Max(maxR2, positions[i].LengthSquared());
        }
        float radius = MathF.Sqrt(MathF.Max(1e-6f, maxR2));
        float scale = targetRadius / radius;
        for (int i = 0; i < vertexCount; i++)
            positions[i] *= scale;

        var indices = new List<uint>(faceCount * 3);
        var normals = new Vector3[vertexCount];

        for (int i = 0; i < faceCount; i++)
        {
            line = reader.ReadLine();
            if (line == null) break;
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int count = int.Parse(parts[0], CultureInfo.InvariantCulture);
            if (count >= 3)
            {
                uint i0 = uint.Parse(parts[1], CultureInfo.InvariantCulture);
                uint i1 = uint.Parse(parts[2], CultureInfo.InvariantCulture);
                uint i2 = uint.Parse(parts[3], CultureInfo.InvariantCulture);
                indices.Add(i0);
                indices.Add(i1);
                indices.Add(i2);

                Vector3 n = Vector3.Cross(positions[i1] - positions[i0], positions[i2] - positions[i0]);
                normals[i0] += n;
                normals[i1] += n;
                normals[i2] += n;
            }
        }

        var verts = new MeshVertex[vertexCount];
        for (int i = 0; i < vertexCount; i++)
        {
            Vector3 n = normals[i].LengthSquared() > 1e-12f ? Vector3.Normalize(normals[i]) : Vector3.UnitY;
            verts[i] = new MeshVertex(positions[i], n, Vector2.Zero);
        }

        var mesh = new CpuMesh
        {
            Name = Path.GetFileNameWithoutExtension(relativePath),
            Vertices = verts,
            Indices = indices.ToArray()
        };
        mesh.Groups.Add(new SubMeshGroup { StartIndex = 0, IndexCount = mesh.Indices.Length, MaterialIndex = 0 });
        return mesh;
    }

    public SceneMeshInstance CreateBoxInstance(Vector3 size, MaterialData mat, Matrix4x4 worldMatrix)
    {
        float hx = size.X * 0.5f;
        float hy = size.Y * 0.5f;
        float hz = size.Z * 0.5f;

        MeshVertex[] verts =
        {
            // +Z
            new(new(-hx, -hy,  hz), new(0, 0, 1), new(0, 0)),
            new(new( hx, -hy,  hz), new(0, 0, 1), new(1, 0)),
            new(new( hx,  hy,  hz), new(0, 0, 1), new(1, 1)),
            new(new(-hx,  hy,  hz), new(0, 0, 1), new(0, 1)),
            // -Z
            new(new( hx, -hy, -hz), new(0, 0, -1), new(0, 0)),
            new(new(-hx, -hy, -hz), new(0, 0, -1), new(1, 0)),
            new(new(-hx,  hy, -hz), new(0, 0, -1), new(1, 1)),
            new(new( hx,  hy, -hz), new(0, 0, -1), new(0, 1)),
            // +Y
            new(new(-hx,  hy,  hz), new(0, 1, 0), new(0, 0)),
            new(new( hx,  hy,  hz), new(0, 1, 0), new(1, 0)),
            new(new( hx,  hy, -hz), new(0, 1, 0), new(1, 1)),
            new(new(-hx,  hy, -hz), new(0, 1, 0), new(0, 1)),
            // -Y
            new(new(-hx, -hy, -hz), new(0, -1, 0), new(0, 0)),
            new(new( hx, -hy, -hz), new(0, -1, 0), new(1, 0)),
            new(new( hx, -hy,  hz), new(0, -1, 0), new(1, 1)),
            new(new(-hx, -hy,  hz), new(0, -1, 0), new(0, 1)),
            // +X
            new(new( hx, -hy,  hz), new(1, 0, 0), new(0, 0)),
            new(new( hx, -hy, -hz), new(1, 0, 0), new(1, 0)),
            new(new( hx,  hy, -hz), new(1, 0, 0), new(1, 1)),
            new(new( hx,  hy,  hz), new(1, 0, 0), new(0, 1)),
            // -X
            new(new(-hx, -hy, -hz), new(-1, 0, 0), new(0, 0)),
            new(new(-hx, -hy,  hz), new(-1, 0, 0), new(1, 0)),
            new(new(-hx,  hy,  hz), new(-1, 0, 0), new(1, 1)),
            new(new(-hx,  hy, -hz), new(-1, 0, 0), new(0, 1)),
        };

        uint[] idx = new uint[36];
        for (uint f = 0; f < 6; f++)
        {
            uint b = f * 4;
            idx[f * 6 + 0] = b + 0;
            idx[f * 6 + 1] = b + 1;
            idx[f * 6 + 2] = b + 2;
            idx[f * 6 + 3] = b + 0;
            idx[f * 6 + 4] = b + 2;
            idx[f * 6 + 5] = b + 3;
        }

        var mesh = new CpuMesh { Name = "Box", Vertices = verts, Indices = idx };
        mesh.Groups.Add(new SubMeshGroup { StartIndex = 0, IndexCount = 36, MaterialIndex = 0 });

        var inst = new SceneMeshInstance
        {
            Mesh = mesh,
            GpuBuffers = new GpuMeshBuffers(_ctx, mesh),
            WorldMatrix = worldMatrix
        };
        inst.Materials.Add(mat);
        return inst;
    }

    public SceneMeshInstance CreatePlaneInstance(float width, float height, MaterialData mat, Matrix4x4 worldMatrix)
    {
        float hw = width * 0.5f;
        float hh = height * 0.5f;

        MeshVertex[] verts =
        {
            new(new(-hw, -hh, 0f), new(0, 0, 1), new(0, 0)),
            new(new( hw, -hh, 0f), new(0, 0, 1), new(1, 0)),
            new(new( hw,  hh, 0f), new(0, 0, 1), new(1, 1)),
            new(new(-hw,  hh, 0f), new(0, 0, 1), new(0, 1)),
        };
        uint[] idx = { 0, 1, 2, 0, 2, 3 };

        var mesh = new CpuMesh { Name = "Plane", Vertices = verts, Indices = idx };
        mesh.Groups.Add(new SubMeshGroup { StartIndex = 0, IndexCount = 6, MaterialIndex = 0 });

        var inst = new SceneMeshInstance
        {
            Mesh = mesh,
            GpuBuffers = new GpuMeshBuffers(_ctx, mesh),
            WorldMatrix = worldMatrix
        };
        inst.Materials.Add(mat);
        return inst;
    }
}
