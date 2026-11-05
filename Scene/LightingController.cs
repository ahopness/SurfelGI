using System.Numerics;

namespace SurfelGI.Scene;

public class LightConfig
{
    public float AzimuthDeg = 35f;
    public float ElevationDeg = 50f;
    public float Intensity = 2.0f;
    public bool Animate = false;
    public float Speed = 0.5f;

    public LightConfig Clone() => new()
    {
        AzimuthDeg = AzimuthDeg,
        ElevationDeg = ElevationDeg,
        Intensity = Intensity,
        Animate = Animate,
        Speed = Speed
    };
}

public class LightingController
{
    public LightConfig Config { get; } = new();
    private readonly LightConfig _defaults = new();
    private DirectionalLightData _light;

    public LightingController(DirectionalLightData initialLight)
    {
        _light = initialLight;
        UpdateLightFromAngles();
    }

    public void SetLight(DirectionalLightData light)
    {
        _light = light;
        UpdateLightFromAngles();
    }

    public void ResetToDefaults()
    {
        Config.AzimuthDeg = _defaults.AzimuthDeg;
        Config.ElevationDeg = _defaults.ElevationDeg;
        Config.Intensity = _defaults.Intensity;
        Config.Animate = _defaults.Animate;
        Config.Speed = _defaults.Speed;
    }

    public void SetAnglesFromEnvMapSunUV(float u, float v)
    {
        Config.AzimuthDeg = (u - 0.5f) * 360f;
        Config.ElevationDeg = (0.5f - v) * 180f;
    }

    public void UpdateLightFromAngles()
    {
        if (Config.Animate) return;

        float az = Config.AzimuthDeg * (MathF.PI / 180f);
        float el = Config.ElevationDeg * (MathF.PI / 180f);
        const float r = 40f;
        float x = r * MathF.Cos(el) * MathF.Cos(az);
        float y = r * MathF.Sin(el);
        float z = r * MathF.Cos(el) * MathF.Sin(az);

        _light.Position = new Vector3(x, y, z);
        _light.Intensity = Config.Intensity;
    }

    public void UpdateAnimation(double timeSeconds)
    {
        if (!Config.Animate) return;

        const float r = 10f;
        const float lowY = 20.5f;
        float t = (float)timeSeconds * Config.Speed;

        _light.Position = new Vector3(MathF.Sin(t) * r, lowY, MathF.Cos(t) * r);
        _light.Intensity = Config.Intensity;
    }

    public static Vector2 FindSunPositionWeighted(CpuTextureData envTex, float threshold = 0.5f)
    {
        float[]? data = envTex.Rgba32FloatData;
        int width = envTex.Width;
        int height = envTex.Height;

        if (data == null || data.Length < width * height * 4)
            return new Vector2(0.5f, 0.25f);

        float maxLum = 0f;
        for (int i = 0; i < data.Length; i += 4)
        {
            float lum = 0.2126f * data[i] + 0.7152f * data[i + 1] + 0.0722f * data[i + 2];
            if (lum > maxLum) maxLum = lum;
        }

        float cutoff = maxLum * threshold;
        double sumX = 0, sumY = 0, sumWeight = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = (y * width + x) * 4;
                float lum = 0.2126f * data[i] + 0.7152f * data[i + 1] + 0.0722f * data[i + 2];
                if (lum > cutoff)
                {
                    sumX += x * lum;
                    sumY += y * lum;
                    sumWeight += lum;
                }
            }
        }

        if (sumWeight <= 1e-8)
            return new Vector2(0.5f, 0.25f);

        float sunU = (float)((sumX / sumWeight + 0.5) / width);
        float sunV = (float)((sumY / sumWeight + 0.5) / height);

        if (envTex.FlipY)
            sunV = 1.0f - sunV;

        return new Vector2(sunU, sunV);
    }
}
