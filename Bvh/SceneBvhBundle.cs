using System.Diagnostics;
using System.Numerics;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.Scene;

namespace SurfelGI.Bvh;

public class SceneBvhBundle : IDisposable
{
    public GpuBuffer<BvhNodeGpu> BvhBuffer { get; }
    public GpuBuffer<Vector4> PositionBuffer { get; }
    public GpuBuffer<Vector4> IndexBuffer { get; }
    public GpuBuffer<Vector4> AttributeBuffer { get; }
    public GpuTexture DiffuseArrayTex { get; }

    private SceneBvhBundle(
        GpuBuffer<BvhNodeGpu> bvhBuffer,
        GpuBuffer<Vector4> positionBuffer,
        GpuBuffer<Vector4> indexBuffer,
        GpuBuffer<Vector4> attributeBuffer,
        GpuTexture diffuseArrayTex)
    {
        BvhBuffer = bvhBuffer;
        PositionBuffer = positionBuffer;
        IndexBuffer = indexBuffer;
        AttributeBuffer = attributeBuffer;
        DiffuseArrayTex = diffuseArrayTex;
    }

    public static SceneBvhBundle Create(WebGpuContext ctx, SceneGraph scene)
    {
        var sw = Stopwatch.StartNew();
        var (diffuseArrayTex, materialIdByUuid) = DiffuseArrayBaker.BuildDiffuseArrayTexture(ctx, scene, 1024);

        var allPositions = new List<Vector4>();
        var allAttributes = new List<Vector4>();
        var allTriIndices = new List<Vector4>();

        foreach (var inst in scene.Meshes)
        {
            if (!inst.Visible) continue;
            var mesh = inst.Mesh;
            Matrix4x4 world = inst.WorldMatrix;

            foreach (var grp in mesh.Groups)
            {
                var mat = grp.MaterialIndex < inst.Materials.Count ? inst.Materials[grp.MaterialIndex] : null;
                int matId = mat != null && materialIdByUuid.TryGetValue(mat.Uuid, out int foundId) ? foundId : 0;

                int end = Math.Min(mesh.Indices.Length, grp.StartIndex + grp.IndexCount);
                for (int i = grp.StartIndex; i + 2 < end; i += 3)
                {
                    uint idx0 = mesh.Indices[i];
                    uint idx1 = mesh.Indices[i + 1];
                    uint idx2 = mesh.Indices[i + 2];

                    Vector3 p0 = Vector3.Transform(mesh.Vertices[idx0].Position, world);
                    Vector3 p1 = Vector3.Transform(mesh.Vertices[idx1].Position, world);
                    Vector3 p2 = Vector3.Transform(mesh.Vertices[idx2].Position, world);

                    Vector2 uv0 = mesh.Vertices[idx0].UV;
                    Vector2 uv1 = mesh.Vertices[idx1].UV;
                    Vector2 uv2 = mesh.Vertices[idx2].UV;

                    uint baseV = (uint)allPositions.Count;
                    allPositions.Add(new Vector4(p0, 0f));
                    allPositions.Add(new Vector4(p1, 0f));
                    allPositions.Add(new Vector4(p2, 0f));

                    allAttributes.Add(new Vector4(uv0.X, uv0.Y, matId, 0f));
                    allAttributes.Add(new Vector4(uv1.X, uv1.Y, matId, 0f));
                    allAttributes.Add(new Vector4(uv2.X, uv2.Y, matId, 0f));

                    allTriIndices.Add(new Vector4(
                        BitConverter.UInt32BitsToSingle(baseV),
                        BitConverter.UInt32BitsToSingle(baseV + 1),
                        BitConverter.UInt32BitsToSingle(baseV + 2),
                        0f));
                }
            }
        }

        var posArray = allPositions.ToArray();
        var attrArray = allAttributes.ToArray();
        var triArray = allTriIndices.ToArray();

        var (bvhNodes, reorderedIndices) = MeshBvhBuilder.BuildSah(posArray, triArray);
        sw.Stop();
        Console.WriteLine($"[BVH] Built {reorderedIndices.Length} triangles ({bvhNodes.Length} nodes) in {sw.ElapsedMilliseconds} ms");

        var bvhBuf = new GpuBuffer<BvhNodeGpu>(ctx, bvhNodes, BufferUsage.Storage, "BVH_Nodes");
        var posBuf = new GpuBuffer<Vector4>(ctx, posArray, BufferUsage.Storage, "BVH_Positions");
        var idxBuf = new GpuBuffer<Vector4>(ctx, reorderedIndices, BufferUsage.Storage, "BVH_Indices");
        var attrBuf = new GpuBuffer<Vector4>(ctx, attrArray, BufferUsage.Storage, "BVH_Attributes");

        return new SceneBvhBundle(bvhBuf, posBuf, idxBuf, attrBuf, diffuseArrayTex);
    }

    public void Dispose()
    {
        BvhBuffer.Dispose();
        PositionBuffer.Dispose();
        IndexBuffer.Dispose();
        AttributeBuffer.Dispose();
        DiffuseArrayTex.Dispose();
    }
}
