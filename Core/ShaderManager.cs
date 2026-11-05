using System.Text;
using Silk.NET.Core.Native;
using Silk.NET.WebGPU;

namespace SurfelGI.Core;

public unsafe class ShaderManager : IDisposable
{
    private readonly WebGpuContext _ctx;
    private readonly string _shaderRootDir;
    private readonly Dictionary<string, IntPtr> _modules = new();
    private readonly Dictionary<string, IntPtr> _computePipelines = new();

    public ShaderManager(WebGpuContext ctx, string? shaderRootDir = null)
    {
        _ctx = ctx;
        _shaderRootDir = shaderRootDir ?? FindShaderDirectory();
    }

    private static string FindShaderDirectory()
    {
        string baseDir = AppContext.BaseDirectory;
        string candidate1 = Path.Combine(baseDir, "Shaders");
        if (Directory.Exists(candidate1)) return candidate1;

        string? curr = Directory.GetCurrentDirectory();
        while (curr != null)
        {
            string candidate = Path.Combine(curr, "Shaders");
            if (Directory.Exists(candidate)) return candidate;
            string candidateSrc = Path.Combine(curr, "src", "NativeGiya", "Shaders");
            if (Directory.Exists(candidateSrc)) return candidateSrc;
            string candidateNative = Path.Combine(curr, "nativegiya", "src", "NativeGiya", "Shaders");
            if (Directory.Exists(candidateNative)) return candidateNative;
            curr = Directory.GetParent(curr)?.FullName;
        }
        return candidate1;
    }

    public string PreprocessWgsl(string relativePath, HashSet<string>? visited = null)
    {
        visited ??= new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string fullPath = Path.GetFullPath(Path.Combine(_shaderRootDir, relativePath));
        if (!visited.Add(fullPath))
            return string.Empty;

        string[] lines = File.ReadAllLines(fullPath);
        var sb = new StringBuilder();
        string dir = Path.GetDirectoryName(fullPath)!;

        foreach (string line in lines)
        {
            string trimmed = line.Trim();
            if (trimmed.StartsWith("#include", StringComparison.Ordinal))
            {
                int firstQuote = trimmed.IndexOf('"');
                int lastQuote = trimmed.LastIndexOf('"');
                if (firstQuote >= 0 && lastQuote > firstQuote)
                {
                    string incFile = trimmed.Substring(firstQuote + 1, lastQuote - firstQuote - 1);
                    string incFull = Path.GetFullPath(Path.Combine(dir, incFile));
                    string relToRoot = Path.GetRelativePath(_shaderRootDir, incFull);
                    sb.AppendLine(PreprocessWgsl(relToRoot, visited));
                    continue;
                }
            }
            sb.AppendLine(line);
        }
        return sb.ToString();
    }

    public ShaderModule* GetOrCreateModule(string relativePath)
    {
        if (_modules.TryGetValue(relativePath, out IntPtr existing))
            return (ShaderModule*)existing;

        string source = PreprocessWgsl(relativePath);
        return CreateModuleFromSource(relativePath, source);
    }

    public ShaderModule* CreateModuleFromSource(string label, string wgslSource)
    {
        byte* codePtr = (byte*)SilkMarshal.StringToPtr(wgslSource);
        byte* labelPtr = (byte*)SilkMarshal.StringToPtr(label);
        try
        {
            var wgslChain = new ShaderModuleWGSLDescriptor
            {
                Chain = new ChainedStruct
                {
                    SType = SType.ShaderSourceWGSL
                },
                Code = codePtr
            };
            var desc = new ShaderModuleDescriptor
            {
                Label = labelPtr,
                NextInChain = (ChainedStruct*)&wgslChain
            };
            ShaderModule* mod = _ctx.Wgpu.DeviceCreateShaderModule(_ctx.Device, &desc);
            _modules[label] = (IntPtr)mod;
            return mod;
        }
        finally
        {
            SilkMarshal.Free((nint)codePtr);
            SilkMarshal.Free((nint)labelPtr);
        }
    }

    public ComputePipeline* GetOrCreateComputePipeline(string relativePath, string entryPoint = "main")
    {
        string key = $"{relativePath}::{entryPoint}";
        if (_computePipelines.TryGetValue(key, out IntPtr existing))
            return (ComputePipeline*)existing;

        ShaderModule* module = GetOrCreateModule(relativePath);
        byte* entryPtr = (byte*)SilkMarshal.StringToPtr(entryPoint);
        byte* labelPtr = (byte*)SilkMarshal.StringToPtr(key);
        try
        {
            var desc = new ComputePipelineDescriptor
            {
                Label = labelPtr,
                Layout = null, // Auto layout
                Compute = new ProgrammableStageDescriptor
                {
                    Module = module,
                    EntryPoint = entryPtr
                }
            };
            ComputePipeline* pipeline = _ctx.Wgpu.DeviceCreateComputePipeline(_ctx.Device, &desc);
            _computePipelines[key] = (IntPtr)pipeline;
            return pipeline;
        }
        finally
        {
            SilkMarshal.Free((nint)entryPtr);
            SilkMarshal.Free((nint)labelPtr);
        }
    }

    public BindGroup* CreateBindGroup(ComputePipeline* pipeline, uint groupIndex, ReadOnlySpan<BindGroupEntry> entries, string label = "")
    {
        BindGroupLayout* layout = _ctx.Wgpu.ComputePipelineGetBindGroupLayout(pipeline, groupIndex);
        byte* labelPtr = (byte*)SilkMarshal.StringToPtr(label);
        try
        {
            fixed (BindGroupEntry* entriesPtr = entries)
            {
                var desc = new BindGroupDescriptor
                {
                    Label = labelPtr,
                    Layout = layout,
                    EntryCount = (nuint)entries.Length,
                    Entries = entriesPtr
                };
                return _ctx.Wgpu.DeviceCreateBindGroup(_ctx.Device, &desc);
            }
        }
        finally
        {
            SilkMarshal.Free((nint)labelPtr);
            _ctx.Wgpu.BindGroupLayoutRelease(layout);
        }
    }

    public BindGroup* CreateBindGroup(RenderPipeline* pipeline, uint groupIndex, ReadOnlySpan<BindGroupEntry> entries, string label = "")
    {
        BindGroupLayout* layout = _ctx.Wgpu.RenderPipelineGetBindGroupLayout(pipeline, groupIndex);
        byte* labelPtr = (byte*)SilkMarshal.StringToPtr(label);
        try
        {
            fixed (BindGroupEntry* entriesPtr = entries)
            {
                var desc = new BindGroupDescriptor
                {
                    Label = labelPtr,
                    Layout = layout,
                    EntryCount = (nuint)entries.Length,
                    Entries = entriesPtr
                };
                return _ctx.Wgpu.DeviceCreateBindGroup(_ctx.Device, &desc);
            }
        }
        finally
        {
            SilkMarshal.Free((nint)labelPtr);
            _ctx.Wgpu.BindGroupLayoutRelease(layout);
        }
    }

    public void Dispose()
    {
        foreach (var kv in _computePipelines)
            _ctx.Wgpu.ComputePipelineRelease((ComputePipeline*)kv.Value);
        _computePipelines.Clear();

        foreach (var kv in _modules)
            _ctx.Wgpu.ShaderModuleRelease((ShaderModule*)kv.Value);
        _modules.Clear();
    }
}
