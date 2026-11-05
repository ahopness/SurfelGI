using System;
using System.Diagnostics;
using System.Numerics;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.WebGPU;
using Silk.NET.Windowing;
using SurfelGI.Bvh;
using SurfelGI.Core;
using SurfelGI.Debug;
using SurfelGI.GiPipeline;
using SurfelGI.Scene;

namespace SurfelGI;

internal static unsafe class Program
{
    public static void Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "--self-test")
        {
            RunSelfTest(args);
            return;
        }

        int maxSmokeFrames = 0;
        if (args.Length >= 2 && args[0] == "--smoke-frames" && int.TryParse(args[1], out int parsedFrames))
        {
            maxSmokeFrames = parsedFrames;
        }

        var windowOpts = WindowOptions.Default;
        windowOpts.Title = "WebGPU Surfel Global Illumination";
        windowOpts.Size = new Vector2D<int>(1600, 900);
        windowOpts.API = GraphicsAPI.None;
        windowOpts.VSync = true;

        using var window = Window.Create(windowOpts);

        WebGpuContext? ctx = null;
        ShaderManager? shaders = null;
        AssetLoader? assets = null;
        Camera? camera = null;
        OrbitControls? controls = null;
        SceneGraph? scene = null;
        LightingController? lighting = null;
        GiTransportSettings giSettings = new();
        OcclusionSettings occlusionSettings = new();
        CsmShadowSystem? csm = null;
        GBufferPass? gbuffer = null;
        SurfelPool? surfelPool = null;
        SurfelHashGrid? hashGrid = null;
        SurfelPreparePass? preparePass = null;
        SurfelFindMissingPass? findMissingPass = null;
        SurfelDispatchArgsPass? allocDispatchPass = null;
        SurfelDispatchArgsPass? integratorDispatchPass = null;
        SurfelAgePass? agePass = null;
        SurfelAllocatePass? allocatePass = null;
        SurfelIntegratePass? integratePass = null;
        SurfelGiResolvePass? resolvePass = null;
        CompositePass? compositePass = null;
        SurfelScreenDebugPass? debugPass = null;
        ImGuiController? imgui = null;
        SceneBvhBundle? bvhBundle = null;
        GpuTexture? blueNoiseTex = null;
        GpuTexture? currentEnvTex = null;

        IInputContext? input = null;
        Vector2 mousePos = Vector2.Zero;
        Vector2 prevMousePos = Vector2.Zero;
        bool leftMouseDown = false;
        bool rightMouseDown = false;
        float mouseWheelDelta = 0f;
        Vector3 prevCameraPos = Vector3.Zero;
        bool forceClearSurfels = true;

        var frameTimer = Stopwatch.StartNew();
        double lastFrameSec = 0.0;
        int fpsFrames = 0;
        double fpsAccum = 0.0;
        float currentFps = 60f;
        float currentFrameMs = 16.6f;

        void LoadSceneByIndex(int presetIndex)
        {
            if (ctx == null || assets == null || camera == null || controls == null ||
                scene == null || lighting == null || imgui == null)
                return;

            var preset = ScenePresets.All[Math.Clamp(presetIndex, 0, ScenePresets.All.Count - 1)];
            scene.ClearMeshes();
            lighting.ResetToDefaults();
            giSettings.ResetDefaults();
            occlusionSettings.ResetDefaults();

            var (envCpu, envGpu) = assets.LoadEnvironmentTexture(preset.HdrPath);
            currentEnvTex = envGpu;
            scene.EnvironmentTexture = envGpu;

            Vector2 sunUv = LightingController.FindSunPositionWeighted(envCpu);
            lighting.SetAnglesFromEnvMapSunUV(sunUv.X, sunUv.Y);

            preset.Populate(scene, assets, occlusionSettings, giSettings, lighting.Config);
            lighting.UpdateLightFromAngles();

            camera.Position = preset.CameraPosition;
            camera.Target = preset.CameraTarget;
            camera.UpdateMatrices();
            prevCameraPos = camera.Position;

            bvhBundle?.Dispose();
            bvhBundle = SceneBvhBundle.Create(ctx, scene);

            imgui.TriangleCount = bvhBundle.IndexBuffer.Count;
            imgui.BvhNodeCount = bvhBundle.BvhBuffer.Count;
            forceClearSurfels = true;
        }

        window.Load += () =>
        {
            ctx = new WebGpuContext(window);
            shaders = new ShaderManager(ctx);
            assets = new AssetLoader(ctx);
            blueNoiseTex = assets.LoadBlueNoiseTexture();

            camera = new Camera
            {
                Aspect = (float)ctx.Width / Math.Max(1, ctx.Height)
            };
            controls = new OrbitControls(camera);

            scene = new SceneGraph();
            lighting = new LightingController(scene.DirLight);

            csm = new CsmShadowSystem(ctx, shaders);
            gbuffer = new GBufferPass(ctx, shaders);
            surfelPool = new SurfelPool(ctx);
            hashGrid = new SurfelHashGrid(ctx, shaders);

            preparePass = new SurfelPreparePass(ctx, shaders);
            findMissingPass = new SurfelFindMissingPass(ctx, shaders);
            allocDispatchPass = new SurfelDispatchArgsPass(ctx, shaders, "AllocDispatchArgs");
            integratorDispatchPass = new SurfelDispatchArgsPass(ctx, shaders, "IntegratorDispatchArgs");
            agePass = new SurfelAgePass(ctx, shaders);
            allocatePass = new SurfelAllocatePass(ctx, shaders);
            integratePass = new SurfelIntegratePass(ctx, shaders, blueNoiseTex);
            resolvePass = new SurfelGiResolvePass(ctx, shaders);
            compositePass = new CompositePass(ctx, shaders);
            debugPass = new SurfelScreenDebugPass(ctx, shaders);
            imgui = new ImGuiController(ctx);

            imgui.OnSceneSelected = LoadSceneByIndex;
            imgui.OnResetSurfels = () => { forceClearSurfels = true; };

            input = window.CreateInput();
            controls.Attach(input);
            foreach (var mouse in input.Mice)
            {
                mouse.MouseMove += (_, pos) => mousePos = pos;
                mouse.MouseDown += (_, btn) =>
                {
                    if (btn == MouseButton.Left) leftMouseDown = true;
                    if (btn == MouseButton.Right) rightMouseDown = true;
                };
                mouse.MouseUp += (_, btn) =>
                {
                    if (btn == MouseButton.Left) leftMouseDown = false;
                    if (btn == MouseButton.Right) rightMouseDown = false;
                };
                mouse.Scroll += (_, wheel) => mouseWheelDelta += wheel.Y;
            }

            foreach (var kb in input.Keyboards)
            {
                kb.KeyDown += (_, key, _) =>
                {
                    if (imgui.WantsCaptureKeyboard) return;
                    if (key >= Key.Number1 && key <= Key.Number6)
                    {
                        int idx = (int)key - (int)Key.Number1;
                        if (idx >= 0 && idx < ScenePresets.All.Count)
                        {
                            imgui.SelectedSceneIndex = idx;
                            LoadSceneByIndex(idx);
                        }
                    }
                    else if (key == Key.R)
                    {
                        forceClearSurfels = true;
                    }
                    else if (key == Key.M)
                    {
                        debugPass.Mode = (ScreenDebugMode)(((int)debugPass.Mode + 1) % Enum.GetValues<ScreenDebugMode>().Length);
                    }
                };
            }

            LoadSceneByIndex(imgui.SelectedSceneIndex);
        };

        window.FramebufferResize += newSize =>
        {
            if (ctx == null || camera == null || gbuffer == null || resolvePass == null)
                return;

            if (newSize.X <= 0 || newSize.Y <= 0) return;

            ctx.ConfigureSurface(newSize.X, newSize.Y);
            camera.Aspect = (float)newSize.X / newSize.Y;
            camera.UpdateMatrices();

            gbuffer.Resize(newSize.X, newSize.Y);
            resolvePass.Resize(newSize.X, newSize.Y);
        };

        window.Render += _ =>
        {
            if (ctx == null || camera == null || controls == null || scene == null ||
                lighting == null || csm == null || gbuffer == null || surfelPool == null ||
                hashGrid == null || preparePass == null || findMissingPass == null ||
                allocDispatchPass == null || integratorDispatchPass == null ||
                agePass == null || allocatePass == null || integratePass == null ||
                resolvePass == null || compositePass == null || debugPass == null ||
                imgui == null || bvhBundle == null)
                return;

            double nowSec = frameTimer.Elapsed.TotalSeconds;
            float dt = (float)(nowSec - lastFrameSec);
            lastFrameSec = nowSec;

            fpsFrames++;
            fpsAccum += dt;
            if (fpsAccum >= 0.25)
            {
                currentFps = (float)(fpsFrames / fpsAccum);
                currentFrameMs = 1000f / Math.Max(0.1f, currentFps);
                fpsFrames = 0;
                fpsAccum = 0.0;
            }
            imgui.Fps = currentFps;
            imgui.FrameTimeMs = currentFrameMs;

            imgui.UpdateInput(dt, ctx.Width, ctx.Height, mousePos, leftMouseDown, rightMouseDown, mouseWheelDelta);
            prevMousePos = mousePos;
            mouseWheelDelta = 0f;

            controls.SuppressInput = imgui.WantsCaptureMouse;
            lighting.UpdateAnimation(nowSec);
            imgui.BuildUi(lighting, giSettings, occlusionSettings, debugPass);
            controls.Update();

            SurfaceTexture surfaceTex = default;
            ctx.Wgpu.SurfaceGetCurrentTexture(ctx.Surface, &surfaceTex);
            if ((surfaceTex.Status != SurfaceGetCurrentTextureStatus.SuccessOptimal &&
                 surfaceTex.Status != SurfaceGetCurrentTextureStatus.SuccessSuboptimal) ||
                surfaceTex.Texture == null)
            {
                ctx.ConfigureSurface(ctx.Width, ctx.Height);
                return;
            }

            var viewDesc = new TextureViewDescriptor
            {
                Format = ctx.SwapchainFormat,
                Dimension = TextureViewDimension.Dimension2D,
                BaseMipLevel = 0,
                MipLevelCount = 1,
                BaseArrayLayer = 0,
                ArrayLayerCount = 1,
                Aspect = TextureAspect.All
            };
            TextureView* swapchainView = ctx.Wgpu.TextureCreateView(surfaceTex.Texture, in viewDesc);

            var encDesc = new CommandEncoderDescriptor();
            CommandEncoder* encoder = ctx.Wgpu.DeviceCreateCommandEncoder(ctx.Device, in encDesc);

            GpuTexture envTex = currentEnvTex ?? blueNoiseTex!;

            // 1. 4-Cascade CSM Shadow Maps
            csm.Execute(encoder, scene, camera);

            // 2. MRT GBuffer Pass (Normal+Rough, Albedo, Depth)
            gbuffer.Execute(encoder, scene, camera);

            // 3. Surfel Prepare Pass (Init pool or reset per-frame counters)
            preparePass.Run(encoder, surfelPool, forceClearSurfels);
            forceClearSurfels = false;

            // 4. Surfel Find Missing Pass (8x8 screen tiles)
            int tileCount = findMissingPass.Run(encoder, camera, gbuffer, surfelPool, hashGrid, prevCameraPos);

            // 5. Indirect Dispatch Args for Surfel Age Pass
            allocDispatchPass.Run(encoder, surfelPool);

            // 6. Surfel Age & Recycling Pass
            agePass.Run(encoder, surfelPool, hashGrid, prevCameraPos, allocDispatchPass.IndirectArgsBuffer);

            // 7. Surfel Allocate Pass
            allocatePass.Run(encoder, surfelPool, findMissingPass, tileCount);

            // 8. Indirect Dispatch Args for Grid Build & Integrator
            integratorDispatchPass.Run(encoder, surfelPool);

            // 9. Spatial Hash Grid Build (Clear -> Count -> ScanSeg -> CollectSeg -> SegPrefix -> Merge -> Slot)
            hashGrid.Build(encoder, surfelPool, camera, integratorDispatchPass.IndirectArgsBuffer);

            // 10. Surfel BVH Ray Tracing & Multiscale Moments Integration
            integratePass.Run(
                encoder,
                surfelPool,
                bvhBundle,
                hashGrid,
                camera,
                scene.DirLight,
                envTex,
                giSettings,
                occlusionSettings,
                integratorDispatchPass.IndirectArgsBuffer);

            // 11. Screen-Space Surfel GI Resolve Pass
            resolvePass.Run(encoder, camera, gbuffer, surfelPool, hashGrid, occlusionSettings, giSettings.HalfResolutionGi);

            // 12. Direct CSM + Resolved GI + Skybox + Reinhard Composite
            compositePass.Execute(encoder, swapchainView, camera, scene, gbuffer, resolvePass, csm, envTex, giSettings);

            // 13. Optional surfel debug views
            debugPass.RunAndRenderOverlay(
                encoder,
                swapchainView,
                camera,
                gbuffer,
                surfelPool,
                hashGrid,
                findMissingPass,
                bvhBundle,
                occlusionSettings,
                prevCameraPos);

            // 14. ImGui UI Overlay
            imgui.Render(encoder, swapchainView, ctx.Width, ctx.Height);

            var cmdDesc = new CommandBufferDescriptor();
            CommandBuffer* cmdBuf = ctx.Wgpu.CommandEncoderFinish(encoder, in cmdDesc);
            if (ctx.LastError != null) throw new InvalidOperationException(ctx.LastError);
            ctx.Wgpu.QueueSubmit(ctx.Queue, 1, &cmdBuf);
            ctx.Wgpu.SurfacePresent(ctx.Surface);

            // 15. Ping-pong Surfel Moments & save camera position
            surfelPool.SwapMoments();
            prevCameraPos = camera.Position;
            ctx.FrameIndex++;

            ctx.Wgpu.CommandBufferRelease(cmdBuf);
            ctx.Wgpu.CommandEncoderRelease(encoder);
            ctx.Wgpu.TextureViewRelease(swapchainView);
            ctx.Wgpu.TextureRelease(surfaceTex.Texture);

            if (maxSmokeFrames > 0 && ctx.FrameIndex >= (uint)maxSmokeFrames)
            {
                Console.WriteLine($"[SmokeTest] Successfully rendered {ctx.FrameIndex} windowed frames ({ctx.Width}x{ctx.Height}, format={ctx.SwapchainFormat}). Closing window.");
                window.Close();
            }
        };

        window.Closing += () =>
        {
            imgui?.Dispose();
            debugPass?.Dispose();
            compositePass?.Dispose();
            resolvePass?.Dispose();
            integratePass?.Dispose();
            allocatePass?.Dispose();
            agePass?.Dispose();
            integratorDispatchPass?.Dispose();
            allocDispatchPass?.Dispose();
            findMissingPass?.Dispose();
            preparePass?.Dispose();
            hashGrid?.Dispose();
            surfelPool?.Dispose();
            gbuffer?.Dispose();
            csm?.Dispose();
            bvhBundle?.Dispose();
            blueNoiseTex?.Dispose();
            scene?.ClearMeshes();
            shaders?.Dispose();
            ctx?.Dispose();
        };

        window.Run();
    }

    private static void RunSelfTest(string[] args)
    {
        Console.WriteLine("=== NativeGiya Offline Self-Test ===");

        // 1. Verify struct sizes match WGSL std430 layout
        Console.WriteLine($"sizeof(SurfelPacked)  = {sizeof(SurfelPacked)} bytes (expected 32)");
        Console.WriteLine($"sizeof(SurfelMoments) = {sizeof(SurfelMoments)} bytes (expected 80)");
        Console.WriteLine($"sizeof(BvhNodeGpu)    = {sizeof(BvhNodeGpu)} bytes (expected 32)");
        if (sizeof(SurfelPacked) != 32 || sizeof(SurfelMoments) != 80 || sizeof(BvhNodeGpu) != 32)
            throw new InvalidOperationException("GPU struct size mismatch.");

        // 1b. Verify native wgpu library loads
        var api = Silk.NET.WebGPU.WebGPU.GetApi();
        var instDesc = new Silk.NET.WebGPU.InstanceDescriptor();
        var inst = api.CreateInstance(&instDesc);
        if (inst == null)
            throw new InvalidOperationException("wgpuCreateInstance returned null.");
        api.InstanceRelease(inst);
        Console.WriteLine("Native WebGPU library loaded and created Instance successfully.");

        // 2. Verify WGSL #include preprocessor on all 11 shaders
        string shaderDir = System.IO.Path.Combine(AppContext.BaseDirectory, "Shaders");
        string[] shaderFiles =
        {
            "gbuffer.wgsl",
            "surfel_prepare.wgsl",
            "surfel_find_missing.wgsl",
            "surfel_dispatch_args.wgsl",
            "surfel_age.wgsl",
            "surfel_allocate.wgsl",
            "surfel_grid_build.wgsl",
            "surfel_integrate.wgsl",
            "surfel_resolve.wgsl",
            "direct_csm_composite.wgsl",
            "surfel_schedule.wgsl",
            "surfel_screen_debug.wgsl"
        };
        foreach (string sf in shaderFiles)
        {
            string full = System.IO.Path.Combine(shaderDir, sf);
            if (!System.IO.File.Exists(full))
                throw new System.IO.FileNotFoundException($"Missing shader file: {full}");
            string src = System.IO.File.ReadAllText(full);
            if (string.IsNullOrWhiteSpace(src))
                throw new InvalidOperationException($"Empty shader file: {sf}");
        }
        Console.WriteLine($"Verified all {shaderFiles.Length} WGSL shader files in {shaderDir}.");

        // 3. Verify HDR & EXR environment loading + weighted sun extraction
        string pubDir = AssetLoader.FindPublicDirectory();
        var hdrData = GltfAndHdrReader.LoadRadianceHdr(System.IO.Path.Combine(pubDir, "exr/pizzo_pernice_puresky_2k.hdr"));
        Vector2 hdrSun = LightingController.FindSunPositionWeighted(hdrData);
        Console.WriteLine($"Loaded HDR '{hdrData.Name}': {hdrData.Width}x{hdrData.Height}, Sun UV=({hdrSun.X:F3}, {hdrSun.Y:F3})");

        var exrData = ExrReader.Load(System.IO.Path.Combine(pubDir, "exr/hay_bales_1k.exr"));
        Vector2 exrSun = LightingController.FindSunPositionWeighted(exrData);
        Console.WriteLine($"Loaded EXR '{exrData.Name}': {exrData.Width}x{exrData.Height}, Sun UV=({exrSun.X:F3}, {exrSun.Y:F3})");

        // 4. Verify glTF 2.0 (.glb / .gltf) parsing & SAH BVH construction
        var sw = Stopwatch.StartNew();
        var leonardoMeshes = GltfAndHdrReader.ParseGltfFile(
            System.IO.Path.Combine(pubDir, "models/leonardo.glb"),
            Matrix4x4.CreateScale(7f),
            null);

        var allPos = new System.Collections.Generic.List<Vector4>();
        var allTri = new System.Collections.Generic.List<Vector4>();
        foreach (var (mesh, _, world) in leonardoMeshes)
        {
            for (int i = 0; i + 2 < mesh.Indices.Length; i += 3)
            {
                Vector3 p0 = Vector3.Transform(mesh.Vertices[mesh.Indices[i]].Position, world);
                Vector3 p1 = Vector3.Transform(mesh.Vertices[mesh.Indices[i + 1]].Position, world);
                Vector3 p2 = Vector3.Transform(mesh.Vertices[mesh.Indices[i + 2]].Position, world);
                uint b = (uint)allPos.Count;
                allPos.Add(new Vector4(p0, 0f));
                allPos.Add(new Vector4(p1, 0f));
                allPos.Add(new Vector4(p2, 0f));
                allTri.Add(new Vector4(
                    BitConverter.UInt32BitsToSingle(b),
                    BitConverter.UInt32BitsToSingle(b + 1),
                    BitConverter.UInt32BitsToSingle(b + 2),
                    0f));
            }
        }
        var (nodes, tris) = MeshBvhBuilder.BuildSah(allPos.ToArray(), allTri.ToArray());
        sw.Stop();
        Console.WriteLine($"Built SAH BVH for 'leonardo.glb': {tris.Length:N0} triangles -> {nodes.Length:N0} BVH nodes in {sw.ElapsedMilliseconds} ms.");

        // 5. Verify full WebGPU Adapter, Device, WGSL Shader Compilation, Pipelines, and GPU Frame Execution
        using var ctx = new WebGpuContext();
        using var shaders = new ShaderManager(ctx);
        ScheduleValidation.Run(ctx, shaders);
        SurfelResolveValidation.Run(ctx, shaders);
        var assets = new AssetLoader(ctx);
        using var blueNoiseTex = assets.LoadBlueNoiseTexture();
        var camera = new Camera { Aspect = (float)ctx.Width / ctx.Height };
        var scene = new SceneGraph();
        var lighting = new LightingController(scene.DirLight);
        var giSettings = new GiTransportSettings();
        var occlusionSettings = new OcclusionSettings();

        using var csm = new CsmShadowSystem(ctx, shaders);
        using var gbuffer = new GBufferPass(ctx, shaders);
        using var surfelPool = new SurfelPool(ctx);
        using var hashGrid = new SurfelHashGrid(ctx, shaders);
        using var preparePass = new SurfelPreparePass(ctx, shaders);
        using var findMissingPass = new SurfelFindMissingPass(ctx, shaders);
        using var allocDispatchPass = new SurfelDispatchArgsPass(ctx, shaders, "AllocDispatchArgs");
        using var integratorDispatchPass = new SurfelDispatchArgsPass(ctx, shaders, "IntegratorDispatchArgs");
        using var agePass = new SurfelAgePass(ctx, shaders);
        using var allocatePass = new SurfelAllocatePass(ctx, shaders);
        using var integratePass = new SurfelIntegratePass(ctx, shaders, blueNoiseTex);
        using var resolvePass = new SurfelGiResolvePass(ctx, shaders);
        using var compositePass = new CompositePass(ctx, shaders);
        using var debugPass = new SurfelScreenDebugPass(ctx, shaders);
        using var imgui = new ImGuiController(ctx);

        int sceneIndex = args.Length > 1 ? int.Parse(args[1]) : 0;
        imgui.SelectedSceneIndex = Math.Clamp(sceneIndex, 0, ScenePresets.All.Count-1);
        var preset = ScenePresets.All[imgui.SelectedSceneIndex];
        var (envCpu, envGpu) = assets.LoadEnvironmentTexture(preset.HdrPath);
        scene.EnvironmentTexture = envGpu;
        preset.Populate(scene, assets, occlusionSettings, giSettings, lighting.Config);
        lighting.UpdateLightFromAngles();
        camera.Position = preset.CameraPosition;
        camera.Target = preset.CameraTarget;
        camera.UpdateMatrices();
        using var bvhBundle = SceneBvhBundle.Create(ctx, scene);

        using var offscreenTarget = new GpuTexture(
            ctx, ctx.Width, ctx.Height, ctx.SwapchainFormat,
            TextureUsage.RenderAttachment | TextureUsage.CopySrc,
            label: "OffscreenTarget");

        if (args.Contains("full-resolution")) giSettings.HalfResolutionGi = false;
        if (args.Contains("indirect")) giSettings.Mode = GiOutputMode.Indirect;
        if (args.Contains("direct")) giSettings.Mode = GiOutputMode.Direct;
        if (args.Contains("reference")) { giSettings.StaggerUpdates = false; }
        var benchmark = Stopwatch.StartNew();
        double steadyMilliseconds = 0;
        Vector3 initialCameraOffset = camera.Position - camera.Target;
        int frameCount = args.Length > 2 ? int.Parse(args[2]) : 8;
        for (int testFrame = 0; testFrame < frameCount; testFrame++) {
            var frameWatch = Stopwatch.StartNew();
            Vector3 previousPosition = camera.Position;
            if (args.Contains("camera-sweep")) {
                camera.Position = camera.Target + Vector3.Transform(initialCameraOffset,
                    Matrix4x4.CreateRotationY(MathF.Sin(testFrame*0.2f)*0.8f));
                camera.UpdateMatrices();
            } else if (testFrame == 4) { camera.Position += new Vector3(0.25f,0,0); camera.UpdateMatrices(); }
            if (testFrame == 16) { lighting.Config.AzimuthDeg += 20f; lighting.UpdateLightFromAngles(); }
            if (testFrame == 24) {
                gbuffer.Resize(960,540); resolvePass.Resize(960,540); offscreenTarget.Resize(960,540);
                camera.Aspect = 960f/540f; camera.UpdateMatrices();
            }
            if (testFrame == 40 && !args.Contains("static") && !args.Contains("camera-sweep")) { camera.Position = camera.Target + new Vector3(0,20,150); camera.UpdateMatrices(); }
            var encDesc = new CommandEncoderDescriptor();
            CommandEncoder* encoder = ctx.Wgpu.DeviceCreateCommandEncoder(ctx.Device, in encDesc);
            csm.Execute(encoder, scene, camera);
            gbuffer.Execute(encoder, scene, camera);
            preparePass.Run(encoder, surfelPool, testFrame == 0);
            int tileCount = findMissingPass.Run(encoder, camera, gbuffer, surfelPool, hashGrid, previousPosition);
            allocDispatchPass.Run(encoder, surfelPool);
            agePass.Run(encoder, surfelPool, hashGrid, previousPosition, allocDispatchPass.IndirectArgsBuffer);
            allocatePass.Run(encoder, surfelPool, findMissingPass, tileCount);
            integratorDispatchPass.Run(encoder, surfelPool);
            hashGrid.Build(encoder, surfelPool, camera, integratorDispatchPass.IndirectArgsBuffer);
            integratePass.Run(encoder, surfelPool, bvhBundle, hashGrid, camera, scene.DirLight, envGpu, giSettings, occlusionSettings, integratorDispatchPass.IndirectArgsBuffer);
            resolvePass.Run(encoder, camera, gbuffer, surfelPool, hashGrid, occlusionSettings, giSettings.HalfResolutionGi);
            compositePass.Execute(encoder, offscreenTarget.View, camera, scene, gbuffer, resolvePass, csm, envGpu, giSettings);
            imgui.UpdateInput(1f / 60f, gbuffer.Width, gbuffer.Height, Vector2.Zero, false, false, 0f);
            imgui.BuildUi(lighting, giSettings, occlusionSettings, debugPass);
            imgui.Render(encoder, offscreenTarget.View, gbuffer.Width, gbuffer.Height);

            var cmdDesc = new CommandBufferDescriptor();
            CommandBuffer* cmdBuf = ctx.Wgpu.CommandEncoderFinish(encoder, in cmdDesc);
            if (ctx.LastError != null) throw new InvalidOperationException(ctx.LastError);
            ctx.Wgpu.QueueSubmit(ctx.Queue, 1, &cmdBuf);
            ctx.Wgpu.CommandBufferRelease(cmdBuf);
            ctx.Wgpu.CommandEncoderRelease(encoder);
            ctx.Wgpu.DeviceWaitIdle(ctx.Device);
            if (ctx.LastError != null) throw new InvalidOperationException(ctx.LastError);
            if (args.Contains("surfel-stats") && (testFrame < 8 || testFrame >= frameCount-4)) {
                var live = FrameCapture.ReadBuffer(ctx,surfelPool.SurfelBuffer).Where(s => s.Age < Constants.SURFEL_TTL).ToArray();
                Console.WriteLine($"[Surfels] frame={testFrame} live={live.Length} born={live.Count(s => s.PosB.W == testFrame)}");
                FrameCapture.Save(ctx,offscreenTarget,$"/tmp/nativegiya-stability-{testFrame}.png");
            }
            if (testFrame >= 8) steadyMilliseconds += frameWatch.Elapsed.TotalMilliseconds;
            surfelPool.SwapMoments(); ctx.FrameIndex++;
        }
        if (args.Contains("capture")) FrameCapture.Save(ctx, offscreenTarget, $"/tmp/nativegiya-{sceneIndex}.png");
        Console.WriteLine($"[Validation] {preset.Label}: {frameCount} completed GPU frames, {benchmark.Elapsed.TotalMilliseconds/frameCount:F2} ms/frame wall time (includes CPU and cold initialization), steady {steadyMilliseconds / Math.Max(1,frameCount-8):F2} ms/frame.");

        Console.WriteLine("=== All Self-Tests Passed ===");
    }
}
