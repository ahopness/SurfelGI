using System.Numerics;

namespace SurfelGI.Scene;

public enum GiOutputMode : uint
{
    Combined = 0,
    Direct = 1,
    Indirect = 2
}

public enum GiSourceMode : uint
{
    AmbientOnly = 0,
    ProbeOnly = 1,
    SurfelOnly = 2,
    SurfelAndProbe = 3
}

public class OcclusionSettings
{
    public float ShadowStrength = 1.2f;
    public float BleedReduction = 0.2f;
    public float GrazingBiasScale = 0.25f;
    public float VarianceBleedScale = 0.15f;

    public Vector4 ToVector4() => new(ShadowStrength, BleedReduction, GrazingBiasScale, VarianceBleedScale);

    public void ResetDefaults()
    {
        ShadowStrength = 1.2f;
        BleedReduction = 0.2f;
        GrazingBiasScale = 0.25f;
        VarianceBleedScale = 0.15f;
    }
}

public class GiTransportSettings
{
    public GiOutputMode Mode = GiOutputMode.Combined;
    public GiSourceMode SourceMode = GiSourceMode.SurfelAndProbe;
    public bool HalfResolutionGi = true;
    public bool StaggerUpdates = true;
    public float Exposure = 1.0f;
    public float IndirectIntensity = 1.0f;
    public int BaseSampleCount = 4;
    public float EnvIntensity = 1.0f;
    public float EnvLod = 4.0f;
    public float GiFromDirect = 1.0f;
    public float GiFromIndirect = 1.0f;
    public float AlbedoBoost = 1.0f;

    public void ResetDefaults()
    {
        Mode = GiOutputMode.Combined;
        SourceMode = GiSourceMode.SurfelAndProbe;
        Exposure = 1.0f;
        IndirectIntensity = 1.0f;
        BaseSampleCount = 4;
        EnvIntensity = 1.0f;
        EnvLod = 4.0f;
        GiFromDirect = 1.0f;
        GiFromIndirect = 1.0f;
        AlbedoBoost = 1.0f;
    }
}

public class ScenePresetDefinition
{
    public string Id { get; set; } = "";
    public string Label { get; set; } = "";
    public string HdrPath { get; set; } = "";
    public Vector3 CameraPosition { get; set; } = new(2f, 5f, 5f);
    public Vector3 CameraTarget { get; set; } = Vector3.Zero;
    public Action<SceneGraph, AssetLoader, OcclusionSettings, GiTransportSettings, LightConfig> Populate { get; set; } = null!;
}

public static class ScenePresets
{
    public static readonly List<ScenePresetDefinition> All = new()
    {
        new ScenePresetDefinition
        {
            Id = "cornell-box",
            Label = "Cornell Box",
            HdrPath = "exr/pizzo_pernice_puresky_2k.hdr",
            CameraPosition = new Vector3(0f, 2.3f, 11f),
            CameraTarget = new Vector3(0f, 2.3f, 1f),
            Populate = (scene, loader, occ, gi, light) =>
            {
                occ.ShadowStrength = 0.5f;
                PopulateCornellScene(scene, loader);
            }
        },
        new ScenePresetDefinition
        {
            Id = "leonardo",
            Label = "Leonardo",
            HdrPath = "exr/hay_bales_1k.exr",
            CameraPosition = new Vector3(0f, 3f, 13f),
            CameraTarget = new Vector3(0f, 2f, 0f),
            Populate = (scene, loader, occ, gi, light) =>
            {
                occ.ShadowStrength = 0.8f;
                PopulateLeonardoScene(scene, loader);
            }
        },
        new ScenePresetDefinition
        {
            Id = "occlusion",
            Label = "Occlusion test",
            HdrPath = "exr/hay_bales_1k.exr",
            CameraPosition = new Vector3(13f, 3f, 0f),
            CameraTarget = new Vector3(0f, 2f, 0f),
            Populate = (scene, loader, occ, gi, light) =>
            {
                occ.ShadowStrength = 1.2f;
                occ.BleedReduction = 0.1f;
                occ.GrazingBiasScale = 0f;
                occ.VarianceBleedScale = 1.0f;
                PopulateOcclusionScene(scene, loader);
            }
        },
        new ScenePresetDefinition
        {
            Id = "marble-bust",
            Label = "Marble Bust",
            HdrPath = "exr/kloppenheim_05_puresky_2k.hdr",
            CameraPosition = new Vector3(0f, -0.3f, 3f),
            CameraTarget = new Vector3(0f, 0f, 0f),
            Populate = (scene, loader, occ, gi, light) =>
            {
                occ.ShadowStrength = 0.5f;
                occ.BleedReduction = 0.1f;
                occ.GrazingBiasScale = 0f;
                occ.VarianceBleedScale = 0.2f;
                PopulateMarbleBustScene(scene, loader);
            }
        },
        new ScenePresetDefinition
        {
            Id = "sponza",
            Label = "Sponza (Heavy)",
            HdrPath = "exr/pizzo_pernice_puresky_2k.hdr",
            CameraPosition = new Vector3(5f, 4f, -0.5f),
            CameraTarget = new Vector3(-1.6f, 3.8f, -0.6f),
            Populate = (scene, loader, occ, gi, light) =>
            {
                light.Intensity = 10f;
                light.AzimuthDeg = 9.6f;
                light.ElevationDeg = 47.4f;
                occ.ShadowStrength = 0.6f;
                occ.BleedReduction = 0.25f;
                occ.GrazingBiasScale = 0.3f;
                occ.VarianceBleedScale = 0.2f;
                gi.IndirectIntensity = 1.7f;
                gi.GiFromDirect = 2.0f;
                gi.GiFromIndirect = 2.3f;
                gi.AlbedoBoost = 1.2f;
                PopulateSponzaScene(scene, loader);
            }
        },
        new ScenePresetDefinition
        {
            Id = "beast",
            Label = "Sponza with Beast",
            HdrPath = "exr/qwantani_noon_puresky_1k.exr",
            CameraPosition = new Vector3(4.2f, 0.5f, -0.5f),
            CameraTarget = new Vector3(-1.6f, 2.8f, -0.6f),
            Populate = (scene, loader, occ, gi, light) =>
            {
                light.Intensity = 10f;
                light.AzimuthDeg = -7.7f;
                light.ElevationDeg = 74.9f;
                occ.ShadowStrength = 0.6f;
                occ.BleedReduction = 0.25f;
                occ.GrazingBiasScale = 0.3f;
                occ.VarianceBleedScale = 0.2f;
                gi.IndirectIntensity = 1.4f;
                gi.GiFromDirect = 2.0f;
                gi.GiFromIndirect = 2.3f;
                gi.AlbedoBoost = 1.2f;
                PopulateBeastScene(scene, loader);
            }
        }
    };

    private static void PopulateCornellScene(SceneGraph scene, AssetLoader loader)
    {
        const float boxWidth = 2f;
        const float boxDepth = 2f;
        const float boxHeight = 1.5f;
        const float wallThickness = 0.02f;

        var redMat = new MaterialData { Name = "Red", BaseColorLinear = MaterialData.SrgbHexToLinear(0xff0000) };
        var greenMat = new MaterialData { Name = "Green", BaseColorLinear = MaterialData.SrgbHexToLinear(0x00ff00) };
        var whiteMat = new MaterialData { Name = "White", BaseColorLinear = MaterialData.SrgbHexToLinear(0xffffff) };

        Matrix4x4 groupXform = Matrix4x4.CreateScale(4f) * Matrix4x4.CreateTranslation(0f, -0.5f, 0f);

        // Floor
        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(boxWidth + wallThickness * 2f, wallThickness, boxDepth + wallThickness),
            whiteMat,
            Matrix4x4.CreateTranslation(0f, -wallThickness * 0.5f, -wallThickness * 0.5f) * groupXform));

        float wallOffsetX = boxWidth * 0.5f + wallThickness * 0.5f;
        // Left wall (red)
        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(wallThickness, boxHeight, boxDepth),
            redMat,
            Matrix4x4.CreateTranslation(-wallOffsetX, boxHeight * 0.5f, 0f) * groupXform));

        // Right wall (green)
        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(wallThickness, boxHeight, boxDepth),
            greenMat,
            Matrix4x4.CreateTranslation(wallOffsetX, boxHeight * 0.5f, 0f) * groupXform));

        // Back wall
        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(2f + 2f * wallThickness, 1.5f, wallThickness),
            whiteMat,
            Matrix4x4.CreateTranslation(0f, 0.75f, -boxDepth * 0.5f - wallThickness * 0.5f) * groupXform));

        // Two ceiling panels with center gap
        float ceilingPanelWidth = 0.75f + wallThickness;
        float ceilingPanelDepth = boxDepth + wallThickness;
        float ceilingY = boxHeight + wallThickness * 0.5f;
        float ceilingXOffset = boxWidth * 0.5f - ceilingPanelWidth * 0.5f + wallThickness;

        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(ceilingPanelWidth, wallThickness, ceilingPanelDepth),
            whiteMat,
            Matrix4x4.CreateTranslation(-ceilingXOffset, ceilingY, -0.5f * wallThickness) * groupXform));

        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(ceilingPanelWidth, wallThickness, ceilingPanelDepth),
            whiteMat,
            Matrix4x4.CreateTranslation(ceilingXOffset, ceilingY, -0.5f * wallThickness) * groupXform));

        // Tall box
        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(0.5f, 0.7f, 0.5f),
            whiteMat,
            Matrix4x4.CreateRotationY(MathF.PI * 0.25f) * Matrix4x4.CreateTranslation(-0.3f, 0.35f, -0.2f) * groupXform));

        // Short box
        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(0.4f, 0.4f, 0.4f),
            whiteMat,
            Matrix4x4.CreateRotationY(MathF.PI * -0.1f) * Matrix4x4.CreateTranslation(0.4f, 0.2f, 0.4f) * groupXform));

        scene.DirLight.OrthoExtent = 10f;
        scene.DirLight.ShadowBias = -0.00002f;
        scene.DirLight.ShadowMapSize = 4096;
        scene.DirLight.UseCsm = false;
    }

    private static void PopulateLeonardoScene(SceneGraph scene, AssetLoader loader)
    {
        Matrix4x4 groupXform = Matrix4x4.CreateScale(7f) * Matrix4x4.CreateTranslation(0f, -0.2f, 0f);
        scene.Meshes.AddRange(loader.LoadGltf("models/leonardo.glb", groupXform));

        scene.DirLight.ShadowMapSize = 2048;
        scene.DirLight.ShadowNear = 0.1f;
        scene.DirLight.ShadowFar = 200f;
        scene.DirLight.OrthoExtent = 5f;
        scene.DirLight.ShadowBias = -0.0002f;
        scene.DirLight.UseCsm = true;
        scene.DirLight.CsmMaxFar = 50f;
        scene.DirLight.CsmLightMargin = 15f;
    }

    private static void PopulateOcclusionScene(SceneGraph scene, AssetLoader loader)
    {
        Matrix4x4 groupXform = Matrix4x4.CreateTranslation(0f, 0.2f, 0f);
        scene.Meshes.AddRange(loader.LoadGltf("models/testocc.glb", groupXform));

        scene.DirLight.ShadowMapSize = 2048;
        scene.DirLight.ShadowNear = 0.1f;
        scene.DirLight.ShadowFar = 200f;
        scene.DirLight.OrthoExtent = 5f;
        scene.DirLight.ShadowBias = -0.0002f;
        scene.DirLight.UseCsm = true;
        scene.DirLight.CsmMaxFar = 50f;
        scene.DirLight.CsmLightMargin = 15f;
    }

    private static void PopulateMarbleBustScene(SceneGraph scene, AssetLoader loader)
    {
        Matrix4x4 groupXform = Matrix4x4.CreateScale(3f) * Matrix4x4.CreateTranslation(0f, -0.9f, 0f);
        scene.Meshes.AddRange(loader.LoadGltf("models/marble_bust/marble_bust_01_4k.gltf", groupXform));

        const float boxWidth = 1f;
        const float boxDepth = 1f;
        const float wallThickness = 0.02f;

        var redMat = new MaterialData { Name = "Red", BaseColorLinear = MaterialData.SrgbHexToLinear(0xff0000), DoubleSided = true };
        var blueMat = new MaterialData { Name = "Blue", BaseColorLinear = MaterialData.SrgbHexToLinear(0x0000ff), DoubleSided = true };
        var whiteMat = new MaterialData { Name = "White", BaseColorLinear = MaterialData.SrgbHexToLinear(0xaaaaaa), DoubleSided = true };

        // Floor
        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(boxWidth + wallThickness * 2f, wallThickness, boxDepth + wallThickness),
            whiteMat,
            Matrix4x4.CreateTranslation(0f, -wallThickness * 0.5f, 0f) * groupXform));

        // Back
        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(boxWidth + wallThickness * 2f, wallThickness, boxDepth / 1.5f),
            whiteMat,
            Matrix4x4.CreateRotationX(MathF.PI * 0.5f) * Matrix4x4.CreateTranslation(0f, 0.33f, -0.5f) * groupXform));

        // Left reflector (red)
        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(0.3f, wallThickness, 0.3f),
            redMat,
            Matrix4x4.CreateRotationZ(-MathF.PI * 0.25f) * Matrix4x4.CreateRotationX(MathF.PI * 0.5f) * Matrix4x4.CreateTranslation(-0.2f, 0.4f, 0f) * groupXform));

        // Right reflector (blue)
        scene.Meshes.Add(loader.CreateBoxInstance(
            new Vector3(0.3f, wallThickness, 0.3f),
            blueMat,
            Matrix4x4.CreateRotationZ(MathF.PI * 0.25f) * Matrix4x4.CreateRotationX(MathF.PI * 0.5f) * Matrix4x4.CreateTranslation(0.2f, 0.4f, 0f) * groupXform));

        scene.DirLight.OrthoExtent = 3f;
        scene.DirLight.ShadowBias = -0.00002f;
        scene.DirLight.ShadowMapSize = 4096;
        scene.DirLight.UseCsm = false;
    }

    private static void PopulateSponzaScene(SceneGraph scene, AssetLoader loader)
    {
        scene.Meshes.AddRange(loader.LoadGltf("models/Sponza-Draco.glb", Matrix4x4.Identity));

        scene.DirLight.ShadowMapSize = 4096;
        scene.DirLight.ShadowNear = 0.1f;
        scene.DirLight.ShadowFar = 100f;
        scene.DirLight.OrthoExtent = 15f;
        scene.DirLight.ShadowBias = -0.0003f;
        scene.DirLight.UseCsm = true;
        scene.DirLight.CsmMaxFar = 50f;
        scene.DirLight.CsmLightMargin = 30f;
    }

    private static void PopulateBeastScene(SceneGraph scene, AssetLoader loader)
    {
        var whiteMaterial = new MaterialData { Name = "BeastWhite", BaseColorLinear = MaterialData.SrgbHexToLinear(0x999999) };
        Matrix4x4 beastXform =
            Matrix4x4.CreateScale(2f) *
            Matrix4x4.CreateRotationY(MathF.PI * 0.5f) *
            Matrix4x4.CreateTranslation(0f, 1.63f + 0.2f, 0f);

        scene.Meshes.AddRange(loader.LoadGltf("models/inferno-beast-from-space-from-jurafjvs-cc0-2.glb", beastXform, whiteMaterial));
        scene.Meshes.AddRange(loader.LoadGltf("models/Sponza-Draco.glb", Matrix4x4.Identity));

        scene.DirLight.ShadowMapSize = 2048;
        scene.DirLight.ShadowNear = 0.1f;
        scene.DirLight.ShadowFar = 200f;
        scene.DirLight.OrthoExtent = 5f;
        scene.DirLight.ShadowBias = -0.0002f;
        scene.DirLight.UseCsm = true;
        scene.DirLight.CsmMaxFar = 50f;
        scene.DirLight.CsmLightMargin = 15f;
    }
}
