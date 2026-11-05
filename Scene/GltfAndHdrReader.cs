using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SurfelGI.Scene;

/// <summary>
/// Self-contained Radiance RGBE (.hdr) decoder and glTF 2.0 (.gltf / .glb) parser
/// using System.Text.Json and SixLabors.ImageSharp.
/// </summary>
public static class GltfAndHdrReader
{
    public static byte[] DecodeImageRgba8(Stream stream, out int width, out int height)
    {
        using var image = Image.Load<Rgba32>(stream);
        width = image.Width;
        height = image.Height;
        byte[] data = new byte[width * height * 4];
        int dst = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Rgba32 p = image[x, y];
                data[dst + 0] = p.R;
                data[dst + 1] = p.G;
                data[dst + 2] = p.B;
                data[dst + 3] = p.A;
                dst += 4;
            }
        }
        return data;
    }

    public static CpuTextureData LoadRadianceHdr(string fullPath)
    {
        using var fs = File.OpenRead(fullPath);
        string line;
        int width = 0, height = 0;

        while ((line = ReadAsciiLine(fs)).Length > 0)
        {
            // header lines
        }

        string resLine = ReadAsciiLine(fs);
        // Expected: -Y <height> +X <width>
        var parts = resLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        for (int i = 0; i + 1 < parts.Length; i += 2)
        {
            if (parts[i].EndsWith('Y'))
                height = int.Parse(parts[i + 1], CultureInfo.InvariantCulture);
            else if (parts[i].EndsWith('X'))
                width = int.Parse(parts[i + 1], CultureInfo.InvariantCulture);
        }

        if (width <= 0 || height <= 0)
            throw new InvalidDataException($"Invalid Radiance HDR dimensions in {fullPath}: '{resLine}'");

        float[] rgba = new float[width * height * 4];
        byte[] scanline = new byte[width * 4];

        for (int y = 0; y < height; y++)
        {
            int b0 = fs.ReadByte();
            int b1 = fs.ReadByte();
            int b2 = fs.ReadByte();
            int b3 = fs.ReadByte();
            if (b0 < 0 || b1 < 0 || b2 < 0 || b3 < 0) break;

            if (b0 != 2 || b1 != 2 || (b2 & 0x80) != 0)
            {
                // Uncompressed RGBE
                scanline[0] = (byte)b0;
                scanline[1] = (byte)b1;
                scanline[2] = (byte)b2;
                scanline[3] = (byte)b3;
                fs.ReadExactly(scanline.AsSpan(4, (width - 1) * 4));
                for (int x = 0; x < width; x++)
                {
                    RgbeToFloat(
                        scanline[x * 4 + 0],
                        scanline[x * 4 + 1],
                        scanline[x * 4 + 2],
                        scanline[x * 4 + 3],
                        rgba, (y * width + x) * 4);
                }
            }
            else
            {
                // New-style RLE per channel
                for (int ch = 0; ch < 4; ch++)
                {
                    int ptr = 0;
                    while (ptr < width)
                    {
                        int count = fs.ReadByte();
                        if (count > 128)
                        {
                            count -= 128;
                            byte val = (byte)fs.ReadByte();
                            for (int i = 0; i < count && ptr < width; i++)
                                scanline[ptr++ * 4 + ch] = val;
                        }
                        else
                        {
                            for (int i = 0; i < count && ptr < width; i++)
                                scanline[ptr++ * 4 + ch] = (byte)fs.ReadByte();
                        }
                    }
                }
                for (int x = 0; x < width; x++)
                {
                    RgbeToFloat(
                        scanline[x * 4 + 0],
                        scanline[x * 4 + 1],
                        scanline[x * 4 + 2],
                        scanline[x * 4 + 3],
                        rgba, (y * width + x) * 4);
                }
            }
        }

        return new CpuTextureData
        {
            Name = Path.GetFileName(fullPath),
            Width = width,
            Height = height,
            Rgba32FloatData = rgba
        };
    }

    private static string ReadAsciiLine(Stream s)
    {
        var sb = new StringBuilder();
        int b;
        while ((b = s.ReadByte()) >= 0)
        {
            if (b == '\n') break;
            if (b != '\r') sb.Append((char)b);
        }
        return sb.ToString();
    }

    private static void RgbeToFloat(byte r, byte g, byte b, byte e, float[] dst, int offset)
    {
        if (e == 0)
        {
            dst[offset + 0] = 0f;
            dst[offset + 1] = 0f;
            dst[offset + 2] = 0f;
            dst[offset + 3] = 1f;
        }
        else
        {
            float f = MathF.Pow(2f, e - (128 + 8));
            dst[offset + 0] = (r + 0.5f) * f;
            dst[offset + 1] = (g + 0.5f) * f;
            dst[offset + 2] = (b + 0.5f) * f;
            dst[offset + 3] = 1f;
        }
    }

    public static List<(CpuMesh Mesh, List<MaterialData> Materials, Matrix4x4 WorldMatrix)> ParseGltfFile(
        string fullPath,
        Matrix4x4 rootTransform,
        MaterialData? overrideMaterial)
    {
        string baseDir = Path.GetDirectoryName(fullPath) ?? "";
        byte[] jsonBytes;
        byte[]? glbBinChunk = null;

        if (fullPath.EndsWith(".glb", StringComparison.OrdinalIgnoreCase))
        {
            using var fs = File.OpenRead(fullPath);
            using var br = new BinaryReader(fs);
            uint magic = br.ReadUInt32();
            uint version = br.ReadUInt32();
            uint totalLen = br.ReadUInt32();
            if (magic != 0x46546C67) // 'glTF'
                throw new InvalidDataException($"Not a valid GLB file: {fullPath}");

            uint jsonLen = br.ReadUInt32();
            uint jsonType = br.ReadUInt32();
            jsonBytes = br.ReadBytes((int)jsonLen);

            if (fs.Position + 8 <= totalLen)
            {
                uint binLen = br.ReadUInt32();
                uint binType = br.ReadUInt32();
                glbBinChunk = br.ReadBytes((int)binLen);
            }
        }
        else
        {
            jsonBytes = File.ReadAllBytes(fullPath);
        }

        using var doc = JsonDocument.Parse(jsonBytes);
        JsonElement root = doc.RootElement;

        // 1. Load buffers
        var buffers = new List<byte[]>();
        if (root.TryGetProperty("buffers", out var buffersElem))
        {
            int idx = 0;
            foreach (var bufElem in buffersElem.EnumerateArray())
            {
                if (idx == 0 && glbBinChunk != null && !bufElem.TryGetProperty("uri", out _))
                {
                    buffers.Add(glbBinChunk);
                }
                else if (bufElem.TryGetProperty("uri", out var uriProp))
                {
                    string uri = uriProp.GetString() ?? "";
                    if (uri.StartsWith("data:", StringComparison.Ordinal))
                    {
                        int comma = uri.IndexOf(',');
                        buffers.Add(Convert.FromBase64String(uri.Substring(comma + 1)));
                    }
                    else
                    {
                        buffers.Add(File.ReadAllBytes(Path.Combine(baseDir, Uri.UnescapeDataString(uri))));
                    }
                }
                else
                {
                    buffers.Add(glbBinChunk ?? Array.Empty<byte>());
                }
                idx++;
            }
        }

        // 2. Load bufferViews
        var bufferViews = new List<(int Buffer, int ByteOffset, int ByteLength, int ByteStride)>();
        if (root.TryGetProperty("bufferViews", out var bvElem))
        {
            foreach (var bv in bvElem.EnumerateArray())
            {
                int b = bv.GetProperty("buffer").GetInt32();
                int off = bv.TryGetProperty("byteOffset", out var o) ? o.GetInt32() : 0;
                int len = bv.GetProperty("byteLength").GetInt32();
                int stride = bv.TryGetProperty("byteStride", out var s) ? s.GetInt32() : 0;
                bufferViews.Add((b, off, len, stride));
            }
        }

        // 3. Load accessors
        var accessors = new List<JsonElement>();
        if (root.TryGetProperty("accessors", out var accElem))
        {
            foreach (var acc in accElem.EnumerateArray())
                accessors.Add(acc);
        }

        // 4. Load images
        var images = new List<JsonElement>();
        if (root.TryGetProperty("images", out var imgElem))
        {
            foreach (var img in imgElem.EnumerateArray())
                images.Add(img);
        }

        // 5. Load textures
        var textures = new List<int>();
        if (root.TryGetProperty("textures", out var texElem))
        {
            foreach (var tex in texElem.EnumerateArray())
            {
                int src = tex.TryGetProperty("source", out var s) ? s.GetInt32() : -1;
                textures.Add(src);
            }
        }

        var decodedImages = new Dictionary<int, CpuTextureData>();
        CpuTextureData? GetDecodedImage(int texIdx)
        {
            if (texIdx < 0 || texIdx >= textures.Count) return null;
            int imgIdx = textures[texIdx];
            if (imgIdx < 0 || imgIdx >= images.Count) return null;
            if (decodedImages.TryGetValue(imgIdx, out var cached)) return cached;

            var imgJson = images[imgIdx];
            byte[]? rawBytes = null;
            string name = $"Image_{imgIdx}";

            if (imgJson.TryGetProperty("uri", out var uriProp))
            {
                string uri = uriProp.GetString() ?? "";
                name = uri;
                if (uri.StartsWith("data:", StringComparison.Ordinal))
                {
                    int comma = uri.IndexOf(',');
                    rawBytes = Convert.FromBase64String(uri.Substring(comma + 1));
                }
                else
                {
                    string imgPath = Path.Combine(baseDir, Uri.UnescapeDataString(uri));
                    if (File.Exists(imgPath))
                        rawBytes = File.ReadAllBytes(imgPath);
                }
            }
            else if (imgJson.TryGetProperty("bufferView", out var bvProp))
            {
                int bvIdx = bvProp.GetInt32();
                var bv = bufferViews[bvIdx];
                rawBytes = new byte[bv.ByteLength];
                Buffer.BlockCopy(buffers[bv.Buffer], bv.ByteOffset, rawBytes, 0, bv.ByteLength);
            }

            if (rawBytes == null) return null;
            using var ms = new MemoryStream(rawBytes);
            byte[] rgba = DecodeImageRgba8(ms, out int w, out int h);
            var texData = new CpuTextureData
            {
                Name = name,
                Width = w,
                Height = h,
                Rgba8Data = rgba
            };
            decodedImages[imgIdx] = texData;
            return texData;
        }

        // 6. Load materials
        var matMap = new Dictionary<int, MaterialData>();
        if (overrideMaterial == null && root.TryGetProperty("materials", out var matsElem))
        {
            int mIdx = 0;
            foreach (var matJson in matsElem.EnumerateArray())
            {
                var matData = new MaterialData
                {
                    Name = matJson.TryGetProperty("name", out var n) ? (n.GetString() ?? $"Mat_{mIdx}") : $"Mat_{mIdx}",
                    DoubleSided = matJson.TryGetProperty("doubleSided", out var ds) && ds.GetBoolean()
                };

                if (matJson.TryGetProperty("pbrMetallicRoughness", out var pbr))
                {
                    if (pbr.TryGetProperty("baseColorFactor", out var bcf))
                    {
                        var arr = bcf.EnumerateArray();
                        arr.MoveNext(); float r = arr.Current.GetSingle();
                        arr.MoveNext(); float g = arr.Current.GetSingle();
                        arr.MoveNext(); float b = arr.Current.GetSingle();
                        matData.BaseColorLinear = new Vector3(r, g, b);
                    }
                    if (pbr.TryGetProperty("baseColorTexture", out var bct) && bct.TryGetProperty("index", out var ti))
                    {
                        matData.DiffuseMap = GetDecodedImage(ti.GetInt32());
                    }
                }
                matMap[mIdx] = matData;
                mIdx++;
            }
        }

        Vector3[] ReadVec3Accessor(int accIdx)
        {
            var acc = accessors[accIdx];
            int count = acc.GetProperty("count").GetInt32();
            int bvIdx = acc.GetProperty("bufferView").GetInt32();
            int accOff = acc.TryGetProperty("byteOffset", out var ao) ? ao.GetInt32() : 0;
            var bv = bufferViews[bvIdx];
            int stride = bv.ByteStride > 0 ? bv.ByteStride : 12;
            byte[] buf = buffers[bv.Buffer];
            var res = new Vector3[count];
            for (int i = 0; i < count; i++)
            {
                int pos = bv.ByteOffset + accOff + i * stride;
                float x = BitConverter.ToSingle(buf, pos + 0);
                float y = BitConverter.ToSingle(buf, pos + 4);
                float z = BitConverter.ToSingle(buf, pos + 8);
                res[i] = new Vector3(x, y, z);
            }
            return res;
        }

        Vector2[] ReadVec2Accessor(int accIdx)
        {
            var acc = accessors[accIdx];
            int count = acc.GetProperty("count").GetInt32();
            int bvIdx = acc.GetProperty("bufferView").GetInt32();
            int accOff = acc.TryGetProperty("byteOffset", out var ao) ? ao.GetInt32() : 0;
            int compType = acc.GetProperty("componentType").GetInt32();
            var bv = bufferViews[bvIdx];
            byte[] buf = buffers[bv.Buffer];
            var res = new Vector2[count];

            if (compType == 5126) // FLOAT
            {
                int stride = bv.ByteStride > 0 ? bv.ByteStride : 8;
                for (int i = 0; i < count; i++)
                {
                    int pos = bv.ByteOffset + accOff + i * stride;
                    float u = BitConverter.ToSingle(buf, pos + 0);
                    float v = BitConverter.ToSingle(buf, pos + 4);
                    res[i] = new Vector2(u, v);
                }
            }
            else if (compType == 5123) // UNSIGNED_SHORT normalized
            {
                int stride = bv.ByteStride > 0 ? bv.ByteStride : 4;
                for (int i = 0; i < count; i++)
                {
                    int pos = bv.ByteOffset + accOff + i * stride;
                    float u = BitConverter.ToUInt16(buf, pos + 0) / 65535f;
                    float v = BitConverter.ToUInt16(buf, pos + 2) / 65535f;
                    res[i] = new Vector2(u, v);
                }
            }
            return res;
        }

        uint[] ReadIndicesAccessor(int accIdx)
        {
            var acc = accessors[accIdx];
            int count = acc.GetProperty("count").GetInt32();
            int bvIdx = acc.GetProperty("bufferView").GetInt32();
            int accOff = acc.TryGetProperty("byteOffset", out var ao) ? ao.GetInt32() : 0;
            int compType = acc.GetProperty("componentType").GetInt32();
            var bv = bufferViews[bvIdx];
            byte[] buf = buffers[bv.Buffer];
            var res = new uint[count];

            if (compType == 5125) // UNSIGNED_INT
            {
                int stride = bv.ByteStride > 0 ? bv.ByteStride : 4;
                for (int i = 0; i < count; i++)
                    res[i] = BitConverter.ToUInt32(buf, bv.ByteOffset + accOff + i * stride);
            }
            else if (compType == 5123) // UNSIGNED_SHORT
            {
                int stride = bv.ByteStride > 0 ? bv.ByteStride : 2;
                for (int i = 0; i < count; i++)
                    res[i] = BitConverter.ToUInt16(buf, bv.ByteOffset + accOff + i * stride);
            }
            else if (compType == 5121) // UNSIGNED_BYTE
            {
                int stride = bv.ByteStride > 0 ? bv.ByteStride : 1;
                for (int i = 0; i < count; i++)
                    res[i] = buf[bv.ByteOffset + accOff + i * stride];
            }
            return res;
        }

        var meshesJson = new List<JsonElement>();
        if (root.TryGetProperty("meshes", out var mElem))
        {
            foreach (var m in mElem.EnumerateArray())
                meshesJson.Add(m);
        }

        var nodesJson = new List<JsonElement>();
        if (root.TryGetProperty("nodes", out var nElem))
        {
            foreach (var n in nElem.EnumerateArray())
                nodesJson.Add(n);
        }

        var results = new List<(CpuMesh Mesh, List<MaterialData> Materials, Matrix4x4 WorldMatrix)>();

        void TraverseNode(int nodeIdx, Matrix4x4 parentWorld)
        {
            if (nodeIdx < 0 || nodeIdx >= nodesJson.Count) return;
            var node = nodesJson[nodeIdx];

            Matrix4x4 local = Matrix4x4.Identity;
            if (node.TryGetProperty("matrix", out var matProp))
            {
                float[] m = new float[16];
                int k = 0;
                foreach (var v in matProp.EnumerateArray()) m[k++] = v.GetSingle();
                local = new Matrix4x4(
                    m[0], m[1], m[2], m[3],
                    m[4], m[5], m[6], m[7],
                    m[8], m[9], m[10], m[11],
                    m[12], m[13], m[14], m[15]);
            }
            else
            {
                Vector3 t = Vector3.Zero;
                Quaternion r = Quaternion.Identity;
                Vector3 s = Vector3.One;
                if (node.TryGetProperty("translation", out var tp))
                {
                    var e = tp.EnumerateArray();
                    e.MoveNext(); float x = e.Current.GetSingle();
                    e.MoveNext(); float y = e.Current.GetSingle();
                    e.MoveNext(); float z = e.Current.GetSingle();
                    t = new Vector3(x, y, z);
                }
                if (node.TryGetProperty("rotation", out var rp))
                {
                    var e = rp.EnumerateArray();
                    e.MoveNext(); float x = e.Current.GetSingle();
                    e.MoveNext(); float y = e.Current.GetSingle();
                    e.MoveNext(); float z = e.Current.GetSingle();
                    e.MoveNext(); float w = e.Current.GetSingle();
                    r = new Quaternion(x, y, z, w);
                }
                if (node.TryGetProperty("scale", out var sp))
                {
                    var e = sp.EnumerateArray();
                    e.MoveNext(); float x = e.Current.GetSingle();
                    e.MoveNext(); float y = e.Current.GetSingle();
                    e.MoveNext(); float z = e.Current.GetSingle();
                    s = new Vector3(x, y, z);
                }
                local = Matrix4x4.CreateScale(s) * Matrix4x4.CreateFromQuaternion(r) * Matrix4x4.CreateTranslation(t);
            }

            Matrix4x4 world = local * parentWorld;

            if (node.TryGetProperty("mesh", out var meshProp))
            {
                int meshIdx = meshProp.GetInt32();
                var meshJson = meshesJson[meshIdx];
                var vertices = new List<MeshVertex>();
                var indices = new List<uint>();
                var groups = new List<SubMeshGroup>();
                var materials = new List<MaterialData>();

                foreach (var prim in meshJson.GetProperty("primitives").EnumerateArray())
                {
                    int mode = prim.TryGetProperty("mode", out var mp) ? mp.GetInt32() : 4;
                    if (mode != 4) continue; // TRIANGLES only

                    var attrs = prim.GetProperty("attributes");
                    if (!attrs.TryGetProperty("POSITION", out var posProp)) continue;
                    var positions = ReadVec3Accessor(posProp.GetInt32());

                    Vector3[]? normals = attrs.TryGetProperty("NORMAL", out var norProp)
                        ? ReadVec3Accessor(norProp.GetInt32())
                        : null;
                    Vector2[]? uvs = attrs.TryGetProperty("TEXCOORD_0", out var uvProp)
                        ? ReadVec2Accessor(uvProp.GetInt32())
                        : null;

                    uint baseVertex = (uint)vertices.Count;
                    for (int i = 0; i < positions.Length; i++)
                    {
                        Vector3 p = positions[i];
                        Vector3 n = normals != null && i < normals.Length ? normals[i] : Vector3.UnitY;
                        Vector2 uv = uvs != null && i < uvs.Length ? new Vector2(uvs[i].X, 1.0f - uvs[i].Y) : Vector2.Zero;
                        vertices.Add(new MeshVertex(p, n, uv));
                    }

                    int startIdx = indices.Count;
                    if (prim.TryGetProperty("indices", out var idxProp))
                    {
                        uint[] primIndices = ReadIndicesAccessor(idxProp.GetInt32());
                        for (int i = 0; i < primIndices.Length; i++)
                            indices.Add(baseVertex + primIndices[i]);
                    }
                    else
                    {
                        for (uint i = 0; i < (uint)positions.Length; i++)
                            indices.Add(baseVertex + i);
                    }

                    if (normals == null)
                    {
                        for (int i = startIdx; i + 2 < indices.Count; i += 3)
                        {
                            int i0 = (int)indices[i];
                            int i1 = (int)indices[i + 1];
                            int i2 = (int)indices[i + 2];
                            Vector3 n = Vector3.Cross(vertices[i1].Position - vertices[i0].Position, vertices[i2].Position - vertices[i0].Position);
                            n = n.LengthSquared() > 1e-12f ? Vector3.Normalize(n) : Vector3.UnitY;
                            var v0 = vertices[i0]; v0.Normal = n; vertices[i0] = v0;
                            var v1 = vertices[i1]; v1.Normal = n; vertices[i1] = v1;
                            var v2 = vertices[i2]; v2.Normal = n; vertices[i2] = v2;
                        }
                    }

                    MaterialData mat;
                    if (overrideMaterial != null)
                        mat = overrideMaterial;
                    else if (prim.TryGetProperty("material", out var matIdxProp) && matMap.TryGetValue(matIdxProp.GetInt32(), out var foundMat))
                        mat = foundMat;
                    else
                        mat = new MaterialData();

                    int matSlot = materials.IndexOf(mat);
                    if (matSlot < 0)
                    {
                        matSlot = materials.Count;
                        materials.Add(mat);
                    }

                    groups.Add(new SubMeshGroup
                    {
                        StartIndex = startIdx,
                        IndexCount = indices.Count - startIdx,
                        MaterialIndex = matSlot
                    });
                }

                if (vertices.Count > 0 && indices.Count > 0)
                {
                    var cpuMesh = new CpuMesh
                    {
                        Name = meshJson.TryGetProperty("name", out var mn) ? (mn.GetString() ?? "GltfMesh") : "GltfMesh",
                        Vertices = vertices.ToArray(),
                        Indices = indices.ToArray()
                    };
                    cpuMesh.Groups.AddRange(groups);
                    results.Add((cpuMesh, materials, world * rootTransform));
                }
            }

            if (node.TryGetProperty("children", out var childrenProp))
            {
                foreach (var child in childrenProp.EnumerateArray())
                    TraverseNode(child.GetInt32(), world);
            }
        }

        if (root.TryGetProperty("scenes", out var scenesElem))
        {
            int sceneIdx = root.TryGetProperty("scene", out var sp) ? sp.GetInt32() : 0;
            int curr = 0;
            foreach (var s in scenesElem.EnumerateArray())
            {
                if (curr == sceneIdx && s.TryGetProperty("nodes", out var rootNodes))
                {
                    foreach (var rn in rootNodes.EnumerateArray())
                        TraverseNode(rn.GetInt32(), Matrix4x4.Identity);
                }
                curr++;
            }
        }

        return results;
    }
}
