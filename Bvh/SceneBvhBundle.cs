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
        Span<uint> triangle = stackalloc uint[3];

        foreach (var inst in scene.Meshes)
        {
            if (!inst.Visible) continue;
            var mesh = inst.Mesh;
            Matrix4x4 world = inst.WorldMatrix;

            foreach (var grp in mesh.Groups)
            {
                var mat = grp.MaterialIndex < inst.Materials.Count ? inst.Materials[grp.MaterialIndex] : null;
                int matId = mat != null && materialIdByUuid.TryGetValue(mat.Uuid, out int foundId) ? foundId : 0;
                // Material IDs live in the vertex attributes, so share vertices
                // within a material group, never across material boundaries.
                var vertexRemap = new Dictionary<uint, uint>();

                int end = Math.Min(mesh.Indices.Length, grp.StartIndex + grp.IndexCount);
                for (int i = grp.StartIndex; i + 2 < end; i += 3)
                {
                    for (int corner = 0; corner < 3; corner++)
                    {
                        uint sourceIndex = mesh.Indices[i + corner];
                        if (!vertexRemap.TryGetValue(sourceIndex, out uint targetIndex))
                        {
                            targetIndex = (uint)allPositions.Count;
                            vertexRemap.Add(sourceIndex, targetIndex);
                            var vertex = mesh.Vertices[sourceIndex];
                            allPositions.Add(new Vector4(Vector3.Transform(vertex.Position, world), 0f));
                            allAttributes.Add(new Vector4(vertex.UV.X, vertex.UV.Y, matId, 0f));
                        }
                        triangle[corner] = targetIndex;
                    }

                    allTriIndices.Add(new Vector4(
                        BitConverter.UInt32BitsToSingle(triangle[0]),
                        BitConverter.UInt32BitsToSingle(triangle[1]),
                        BitConverter.UInt32BitsToSingle(triangle[2]),
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
