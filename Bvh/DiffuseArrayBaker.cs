using System.Numerics;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.Scene;

namespace SurfelGI.Bvh;

public static class DiffuseArrayBaker
{
    public static (GpuTexture DiffuseArrayTex, Dictionary<string, int> MaterialIdByUuid) BuildDiffuseArrayTexture(
        WebGpuContext ctx,
        SceneGraph scene,
        int layerSize = 1024)
    {
        var materialIdByUuid = new Dictionary<string, int>();
        var materials = new List<MaterialData>();

        foreach (var inst in scene.Meshes)
        {
            if (!inst.Visible) continue;
            foreach (var mat in inst.Materials)
            {
                if (!materialIdByUuid.ContainsKey(mat.Uuid))
                {
                    materialIdByUuid[mat.Uuid] = materials.Count;
                    materials.Add(mat);
                }
            }
        }

        int materialCount = Math.Max(1, materials.Count);
        int arrayLayerCount = Math.Max(2, materialCount);

        var arrayTex = new GpuTexture(
            ctx,
            layerSize,
            layerSize,
            TextureFormat.Rgba8UnormSrgb,
            TextureUsage.TextureBinding | TextureUsage.CopyDst,
            depthOrArrayLayers: arrayLayerCount,
            viewDimension: TextureViewDimension.Dimension2DArray,
            filter: FilterMode.Linear,
            addressMode: AddressMode.Repeat,
            label: "DiffuseArrayTex");

        byte[] layerPixels = new byte[layerSize * layerSize * 4];

        for (int layer = 0; layer < arrayLayerCount; layer++)
        {
            MaterialData? mat = layer < materials.Count ? materials[layer] : null;
            Vector3 baseLinear = mat?.BaseColorLinear ?? Vector3.One;
            CpuTextureData? map = mat?.DiffuseMap;

            if (map == null || map.Rgba8Data == null || map.Width <= 0 || map.Height <= 0)
            {
                byte r = LinearToSrgbByte(baseLinear.X);
                byte g = LinearToSrgbByte(baseLinear.Y);
                byte b = LinearToSrgbByte(baseLinear.Z);

                for (int i = 0; i < layerPixels.Length; i += 4)
                {
                    layerPixels[i + 0] = r;
                    layerPixels[i + 1] = g;
                    layerPixels[i + 2] = b;
                    layerPixels[i + 3] = 255;
                }
            }
            else
            {
                byte[] src = map.Rgba8Data;
                int srcW = map.Width;
                int srcH = map.Height;

                Parallel.For(0, layerSize, y =>
                {
                    // Flip V to match flippedUv = vec2(uv.x, 1.0 - uv.y) in diffuseArray.ts
                    float v = 1.0f - (y + 0.5f) / layerSize;
                    int sy = Math.Clamp((int)(v * srcH), 0, srcH - 1);

                    for (int x = 0; x < layerSize; x++)
                    {
                        float u = (x + 0.5f) / layerSize;
                        int sx = Math.Clamp((int)(u * srcW), 0, srcW - 1);

                        int srcIdx = (sy * srcW + sx) * 4;
                        float sr = MaterialData.SrgbToLinear(src[srcIdx + 0] / 255f) * baseLinear.X;
                        float sg = MaterialData.SrgbToLinear(src[srcIdx + 1] / 255f) * baseLinear.Y;
                        float sb = MaterialData.SrgbToLinear(src[srcIdx + 2] / 255f) * baseLinear.Z;

                        int dstIdx = (y * layerSize + x) * 4;
                        layerPixels[dstIdx + 0] = LinearToSrgbByte(sr);
                        layerPixels[dstIdx + 1] = LinearToSrgbByte(sg);
                        layerPixels[dstIdx + 2] = LinearToSrgbByte(sb);
                        layerPixels[dstIdx + 3] = 255;
                    }
                });
            }

            arrayTex.UploadLayer<byte>(layerPixels, 4, layer);
        }

        return (arrayTex, materialIdByUuid);
    }

    private static byte LinearToSrgbByte(float c)
    {
        c = Math.Clamp(c, 0f, 1f);
        float srgb = c <= 0.0031308f ? c * 12.92f : 1.055f * MathF.Pow(c, 1.0f / 2.4f) - 0.055f;
        return (byte)Math.Clamp((int)MathF.Round(srgb * 255f), 0, 255);
    }
}
