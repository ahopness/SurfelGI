# SurfelGI

<img width="1590" height="891" alt="Screenshot From 2026-11-05 09-38-11" src="https://github.com/user-attachments/assets/8d08d78f-67a7-4749-811d-c13bda16cc43" />
<img width="1590" height="891" alt="Screenshot From 2026-11-05 09-37-41" src="https://github.com/user-attachments/assets/8654332b-ec82-4b98-9283-42d11bdf3fbb" />


An experimental native C# renderer for real-time surfel global illumination. It uses WebGPU compute shaders through `wgpu-native`, with rasterized direct lighting and ray-traced indirect lighting cached on surface elements (surfels).

This repository is a native fork of [Webgiya](https://github.com/jure/webgiya), a WebGPU and Three.js experiment exploring surfel-based global illumination on the web. The original project’s [technical write-up](https://juretriglav.si/surfel-based-global-illumination-on-the-web) explains the approach, and its [live demo](https://jure.github.io/webgiya/) remains the web version. This fork replaces the browser application with a standalone C# renderer and includes its own scene, BVH, rendering, and diagnostic code.

## Rendering pipeline

Each frame renders directional shadows (cascaded where configured) and a normal, albedo, and depth G-buffer, then runs GPU passes to find missing surfel coverage, age and allocate surfels, and build a camera-centered spatial grid. Rays are traced through a CPU-built scene BVH to integrate indirect lighting with guided sampling and temporal history. The renderer gathers surfel lighting back onto the screen, applies visibility from radial depth moments, and composites direct and indirect lighting with the environment.

| Stage | Purpose | Source |
| --- | --- | --- |
| G-buffer and shadows | Rasterize scene depth, normals, albedo, and directional shadows. | [`GiPipeline/GBufferPass.cs`](GiPipeline/GBufferPass.cs), [`Scene/CsmShadowSystem.cs`](Scene/CsmShadowSystem.cs) |
| Surfel coverage and lifecycle | Find under-sampled screen tiles, schedule candidates, allocate surfels, and recycle old ones. | [`GiPipeline/SurfelFindMissingPass.cs`](GiPipeline/SurfelFindMissingPass.cs), [`GiPipeline/SurfelAllocatePass.cs`](GiPipeline/SurfelAllocatePass.cs), [`GiPipeline/SurfelAgePass.cs`](GiPipeline/SurfelAgePass.cs) |
| Spatial grid | Organize surfels into camera-centered cascades for lookup. | [`GiPipeline/SurfelHashGrid.cs`](GiPipeline/SurfelHashGrid.cs) |
| Integration | Trace rays through the scene BVH, guide sampling, and update temporal lighting and radial depth moments. | [`GiPipeline/SurfelIntegratePass.cs`](GiPipeline/SurfelIntegratePass.cs), [`Bvh/`](Bvh/) |
| Resolve and composite | Gather surfel irradiance at screen pixels and combine it with direct lighting and the environment. | [`GiPipeline/SurfelGiResolvePass.cs`](GiPipeline/SurfelGiResolvePass.cs), [`GiPipeline/CompositePass.cs`](GiPipeline/CompositePass.cs) |

The surfel pool supports up to 262,144 surfels. Grid entries use atomic writes, so their order is not stable; lighting and coverage queries traverse the full stored cell list. CPU and WGSL GPU layouts share constants in [`Constants.cs`](Constants.cs) and [`Shaders/Common/constants.wgsl`](Shaders/Common/constants.wgsl), which must stay in sync when GPU data layouts change.

## Requirements

- .NET 10 SDK.
- Linux x64 with a working GPU driver and a WebGPU-compatible adapter. This checkout includes `runtimes/linux-x64/native/libwgpu_native.so`.
- Scene assets under the included `public/` directory.

The project uses the `ImGui.NET` and `SixLabors.ImageSharp` NuGet packages. ImGui.NET supplies the native `cimgui` runtime for its supported platforms. The windowing and WebGPU interop shim is in [`Core/SilkWebGpuShim.cs`](Core/SilkWebGpuShim.cs).

## Build and run

Run from the repository root:

```bash
dotnet restore
dotnet build
dotnet run
```

### Assets

The repository includes the required blue-noise texture, models, and HDR/EXR environment maps:

```text
public/
├── draco/
├── exr/
├── models/
└── textures/
```

Assets load at runtime and are not downloaded during build. The loader searches upward from the current working directory for a `public/` directory, then checks beside the executable. Running from the repository root finds the included assets. The Sponza preset uses `models/sponza.glb` when available in place of `Sponza-Draco.glb`.

### Controls

| Input | Action |
| --- | --- |
| Left mouse drag | Orbit the camera |
| Right mouse drag | Pan the camera |
| Mouse wheel | Zoom |
| `1`–`6` | Select a scene preset |
| `R` | Reset and regenerate surfels |
| `M` | Cycle screen debug views |

The presets, in keyboard order, are Cornell Box, Leonardo, Occlusion test, Marble Bust, Sponza (Heavy), and Sponza with Beast. The overlay also provides combined, direct-only, and indirect-only output; half-resolution GI; staggered updates; sun controls; integrator settings; and radial occlusion controls.

## GPU self-test

The self-test does not open the interactive window, but it still requires a WebGPU adapter, native libraries, and the scene assets:

```bash
# Default scene, eight frames
dotnet run --project SurfelGI.csproj -- --self-test

# Sponza, twelve frames, and an image capture
dotnet run --project SurfelGI.csproj -- --self-test 4 12 capture

# Exercise the windowed renderer, then close automatically
dotnet run --project SurfelGI.csproj -- --smoke-frames 8
```

For `--self-test`, the optional positional arguments are the zero-based scene index and frame count. Additional options include `capture`, `surfel-stats`, `full-resolution`, `indirect`, `direct`, `reference` (disable staggered updates), and `camera-sweep`. Captures are saved to `/tmp/nativegiya-<sceneIndex>.png`; `surfel-stats` also saves selected frames to `/tmp/nativegiya-stability-<frame>.png`.

The self-test checks GPU struct sizes, shader availability, asset decoding, BVH construction, surfel scheduling, resolve behavior in crowded grid cells, and GPU frame execution. It changes camera, lighting, and resolution during the run, so it is not a fixed-camera image comparison test.

## Source layout

| Directory or file | Purpose |
| --- | --- |
| `Core/` | WebGPU context, native interop, buffers, textures, and shader loading |
| `GiPipeline/` | G-buffer, surfel lifecycle, grid, integration, resolve, and composition |
| `Shaders/` | WGSL compute and render passes plus shared shader helpers |
| `Bvh/` | CPU BVH construction and GPU scene data |
| `Scene/` | Camera, scene presets, asset loading, environment and sun lighting |
| `Debug/` | ImGui controls, diagnostic views, captures, and GPU validation |
| `Program.cs` | Interactive and self-test entry points; connects the rendering passes |

The build copies WGSL shaders to the output directory. Edit the source shaders under `Shaders/`.

## Current limitations

Indirect lighting depends on surfel coverage and takes time to converge. Areas without surfels have no indirect-lighting fallback, so they can remain black in indirect-only mode while coverage builds. Dense grid cells require more lookup work, and larger scenes increase BVH construction and ray-tracing costs.
