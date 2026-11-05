using System.Numerics;

namespace SurfelGI.Bvh;

public static class MeshBvhBuilder
{
    private const int BinCount = 32;
    private const int MaxDepth = 40;
    private const int MaxLeafTris = 4;

    private struct TriBounds
    {
        public Vector3 Min;
        public Vector3 Max;
        public Vector3 Centroid;
        public int OriginalTriIndex;
    }

    private class BuildNode
    {
        public Vector3 BoundsMin;
        public Vector3 BoundsMax;
        public int SplitAxis = -1;
        public int TriOffset;
        public int TriCount;
        public BuildNode? Left;
        public BuildNode? Right;
        public bool IsLeaf => Left == null && Right == null;
    }

    public static (BvhNodeGpu[] Nodes, Vector4[] ReorderedIndices) BuildSah(
        ReadOnlySpan<Vector4> positions,
        ReadOnlySpan<Vector4> triangleIndices)
    {
        int triCount = triangleIndices.Length;
        if (triCount == 0)
        {
            var dummyNode = new BvhNodeGpu
            {
                BoundsMin = Vector3.Zero,
                BoundsMax = Vector3.Zero,
                RightChildOrTriangleOffset = 0,
                SplitAxisOrTriangleCount = 0xFFFF0000u
            };
            return (new[] { dummyNode }, Array.Empty<Vector4>());
        }

        var prims = new TriBounds[triCount];
        for (int i = 0; i < triCount; i++)
        {
            Vector4 idx = triangleIndices[i];
            int i0 = (int)BitConverter.SingleToUInt32Bits(idx.X);
            int i1 = (int)BitConverter.SingleToUInt32Bits(idx.Y);
            int i2 = (int)BitConverter.SingleToUInt32Bits(idx.Z);

            Vector3 p0 = new(positions[i0].X, positions[i0].Y, positions[i0].Z);
            Vector3 p1 = new(positions[i1].X, positions[i1].Y, positions[i1].Z);
            Vector3 p2 = new(positions[i2].X, positions[i2].Y, positions[i2].Z);

            Vector3 bMin = Vector3.Min(p0, Vector3.Min(p1, p2));
            Vector3 bMax = Vector3.Max(p0, Vector3.Max(p1, p2));

            prims[i] = new TriBounds
            {
                Min = bMin,
                Max = bMax,
                Centroid = (bMin + bMax) * 0.5f,
                OriginalTriIndex = i
            };
        }

        BuildNode root = SplitRecursive(prims, 0, triCount, 0);

        int totalNodes = CountNodes(root);
        var flatNodes = new BvhNodeGpu[totalNodes];
        int nextIdx = 0;
        PopulateFlatBuffer(root, flatNodes, ref nextIdx);

        var reordered = new Vector4[triCount];
        for (int i = 0; i < triCount; i++)
        {
            reordered[i] = triangleIndices[prims[i].OriginalTriIndex];
        }

        return (flatNodes, reordered);
    }

    private static BuildNode SplitRecursive(TriBounds[] prims, int offset, int count, int depth)
    {
        Vector3 nodeMin = new(float.MaxValue);
        Vector3 nodeMax = new(float.MinValue);
        Vector3 centMin = new(float.MaxValue);
        Vector3 centMax = new(float.MinValue);

        for (int i = offset; i < offset + count; i++)
        {
            nodeMin = Vector3.Min(nodeMin, prims[i].Min);
            nodeMax = Vector3.Max(nodeMax, prims[i].Max);
            centMin = Vector3.Min(centMin, prims[i].Centroid);
            centMax = Vector3.Max(centMax, prims[i].Centroid);
        }

        var node = new BuildNode
        {
            BoundsMin = nodeMin,
            BoundsMax = nodeMax
        };

        if (count <= MaxLeafTris || depth >= MaxDepth)
        {
            node.TriOffset = offset;
            node.TriCount = count;
            return node;
        }

        var (bestAxis, splitPos) = FindOptimalSahSplit(prims, offset, count, nodeMin, nodeMax, centMin, centMax);
        if (bestAxis == -1)
        {
            node.TriOffset = offset;
            node.TriCount = count;
            return node;
        }

        int mid = Partition(prims, offset, count, bestAxis, splitPos);
        if (mid == offset || mid == offset + count)
        {
            // Fallback median split along largest centroid extent axis
            Vector3 ext = centMax - centMin;
            bestAxis = ext.X >= ext.Y && ext.X >= ext.Z ? 0 : (ext.Y >= ext.Z ? 1 : 2);
            Array.Sort(prims, offset, count, Comparer<TriBounds>.Create((a, b) => GetAxis(a.Centroid, bestAxis).CompareTo(GetAxis(b.Centroid, bestAxis))));
            mid = offset + count / 2;
        }

        node.SplitAxis = bestAxis;
        node.Left = SplitRecursive(prims, offset, mid - offset, depth + 1);
        node.Right = SplitRecursive(prims, mid, offset + count - mid, depth + 1);
        return node;
    }

    private static (int Axis, float SplitPos) FindOptimalSahSplit(
        TriBounds[] prims,
        int offset,
        int count,
        Vector3 nodeMin,
        Vector3 nodeMax,
        Vector3 centMin,
        Vector3 centMax)
    {
        int bestAxis = -1;
        float bestCost = float.MaxValue;
        float bestPos = 0f;

        Vector3 extent = centMax - centMin;
        Span<Vector3> binMin = stackalloc Vector3[BinCount];
        Span<Vector3> binMax = stackalloc Vector3[BinCount];
        Span<int> binCounts = stackalloc int[BinCount];
        Span<float> leftArea = stackalloc float[BinCount - 1];
        Span<int> leftCount = stackalloc int[BinCount - 1];

        for (int axis = 0; axis < 3; axis++)
        {
            float axisExtent = GetAxis(extent, axis);
            if (axisExtent <= 1e-6f)
                continue;

            float axisMin = GetAxis(centMin, axis);
            float invStep = BinCount / axisExtent;

            for (int b = 0; b < BinCount; b++)
            {
                binMin[b] = new Vector3(float.MaxValue);
                binMax[b] = new Vector3(float.MinValue);
                binCounts[b] = 0;
            }

            for (int i = offset; i < offset + count; i++)
            {
                float c = GetAxis(prims[i].Centroid, axis);
                int b = Math.Clamp((int)((c - axisMin) * invStep), 0, BinCount - 1);
                binCounts[b]++;
                binMin[b] = Vector3.Min(binMin[b], prims[i].Min);
                binMax[b] = Vector3.Max(binMax[b], prims[i].Max);
            }

            Vector3 curMin = new(float.MaxValue);
            Vector3 curMax = new(float.MinValue);
            int curCnt = 0;

            for (int b = 0; b < BinCount - 1; b++)
            {
                if (binCounts[b] > 0)
                {
                    curMin = Vector3.Min(curMin, binMin[b]);
                    curMax = Vector3.Max(curMax, binMax[b]);
                    curCnt += binCounts[b];
                }
                leftCount[b] = curCnt;
                leftArea[b] = curCnt > 0 ? SurfaceArea(curMin, curMax) : 0f;
            }

            curMin = new Vector3(float.MaxValue);
            curMax = new Vector3(float.MinValue);
            curCnt = 0;

            for (int b = BinCount - 1; b >= 1; b--)
            {
                if (binCounts[b] > 0)
                {
                    curMin = Vector3.Min(curMin, binMin[b]);
                    curMax = Vector3.Max(curMax, binMax[b]);
                    curCnt += binCounts[b];
                }

                int lCnt = leftCount[b - 1];
                if (lCnt == 0 || curCnt == 0)
                    continue;

                float rArea = SurfaceArea(curMin, curMax);
                float cost = leftArea[b - 1] * lCnt + rArea * curCnt;
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestAxis = axis;
                    bestPos = axisMin + (axisExtent / BinCount) * b;
                }
            }
        }

        return (bestAxis, bestPos);
    }

    private static int Partition(TriBounds[] prims, int offset, int count, int axis, float splitPos)
    {
        int left = offset;
        int right = offset + count - 1;
        while (left <= right)
        {
            if (GetAxis(prims[left].Centroid, axis) < splitPos)
            {
                left++;
            }
            else
            {
                (prims[left], prims[right]) = (prims[right], prims[left]);
                right--;
            }
        }
        return left;
    }

    private static float GetAxis(Vector3 v, int axis) => axis == 0 ? v.X : (axis == 1 ? v.Y : v.Z);

    private static float SurfaceArea(Vector3 min, Vector3 max)
    {
        Vector3 d = Vector3.Max(Vector3.Zero, max - min);
        return 2.0f * (d.X * d.Y + d.Y * d.Z + d.Z * d.X);
    }

    private static int CountNodes(BuildNode node)
    {
        if (node.IsLeaf) return 1;
        return 1 + CountNodes(node.Left!) + CountNodes(node.Right!);
    }

    private static int PopulateFlatBuffer(BuildNode node, BvhNodeGpu[] buffer, ref int nextIndex)
    {
        int myIndex = nextIndex++;
        if (node.IsLeaf)
        {
            uint countClamped = (uint)Math.Clamp(node.TriCount, 0, 0xFFFF);
            buffer[myIndex] = new BvhNodeGpu
            {
                BoundsMin = node.BoundsMin,
                BoundsMax = node.BoundsMax,
                RightChildOrTriangleOffset = (uint)node.TriOffset,
                SplitAxisOrTriangleCount = 0xFFFF0000u | countClamped
            };
            return myIndex;
        }

        // Left child is placed immediately at myIndex + 1
        PopulateFlatBuffer(node.Left!, buffer, ref nextIndex);
        int rightIndex = PopulateFlatBuffer(node.Right!, buffer, ref nextIndex);
        uint relativeRight = (uint)(rightIndex - myIndex);

        buffer[myIndex] = new BvhNodeGpu
        {
            BoundsMin = node.BoundsMin,
            BoundsMax = node.BoundsMax,
            RightChildOrTriangleOffset = relativeRight,
            SplitAxisOrTriangleCount = (uint)node.SplitAxis
        };
        return myIndex;
    }
}
