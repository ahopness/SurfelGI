using Silk.NET.Core.Native;
using Silk.NET.WebGPU;

namespace SurfelGI.Core;

public unsafe class GpuTexture : IDisposable
{
    private readonly WebGpuContext _ctx;
    public Texture* Handle { get; private set; }
    public TextureView* View { get; private set; }
    public Sampler* Sampler { get; private set; }
    public int Width { get; private set; }
    public int Height { get; private set; }
    public int DepthOrArrayLayers { get; private set; }
    public uint MipLevelCount { get; private set; }
    public TextureFormat Format { get; }
    public TextureUsage Usage { get; }
    public TextureViewDimension ViewDimension { get; }
    public string Label { get; }

    public GpuTexture(
        WebGpuContext ctx,
        int width,
        int height,
        TextureFormat format,
        TextureUsage usage,
        int depthOrArrayLayers = 1,
        uint mipLevelCount = 1,
        TextureViewDimension viewDimension = TextureViewDimension.Dimension2D,
        FilterMode filter = FilterMode.Linear,
        AddressMode addressMode = AddressMode.ClampToEdge,
        string label = "")
    {
        _ctx = ctx;
        Format = format;
        Usage = usage;
        ViewDimension = viewDimension;
        Label = label;

        Allocate(width, height, depthOrArrayLayers, mipLevelCount);
        CreateSampler(filter, addressMode);
    }

    private void Allocate(int width, int height, int depthOrArrayLayers, uint mipLevelCount)
    {
        if (View != null) { _ctx.Wgpu.TextureViewRelease(View); View = null; }
        if (Handle != null) { _ctx.Wgpu.TextureDestroy(Handle); _ctx.Wgpu.TextureRelease(Handle); Handle = null; }

        Width = Math.Max(1, width);
        Height = Math.Max(1, height);
        DepthOrArrayLayers = Math.Max(1, depthOrArrayLayers);
        MipLevelCount = Math.Max(1u, mipLevelCount);

        byte* labelPtr = (byte*)SilkMarshal.StringToPtr(Label);
        try
        {
            var desc = new TextureDescriptor
            {
                Label = labelPtr,
                Dimension = TextureDimension.Dimension2D,
                Size = new Extent3D((uint)Width, (uint)Height, (uint)DepthOrArrayLayers),
                Format = Format,
                MipLevelCount = MipLevelCount,
                SampleCount = 1,
                Usage = Usage
            };
            Handle = _ctx.Wgpu.DeviceCreateTexture(_ctx.Device, &desc);

            var viewDesc = new TextureViewDescriptor
            {
                Label = labelPtr,
                Format = Format,
                Dimension = ViewDimension,
                BaseMipLevel = 0,
                MipLevelCount = MipLevelCount,
                BaseArrayLayer = 0,
                ArrayLayerCount = (uint)DepthOrArrayLayers,
                Aspect = Format == TextureFormat.Depth32Float ? TextureAspect.DepthOnly : TextureAspect.All
            };
            View = _ctx.Wgpu.TextureCreateView(Handle, &viewDesc);
        }
        finally
        {
            SilkMarshal.Free((nint)labelPtr);
        }
    }

    public TextureView* CreateLayerView(uint layerIndex, uint mipLevel = 0)
    {
        var viewDesc = new TextureViewDescriptor
        {
            Format = Format,
            Dimension = TextureViewDimension.Dimension2D,
            BaseMipLevel = mipLevel,
            MipLevelCount = 1,
            BaseArrayLayer = layerIndex,
            ArrayLayerCount = 1,
            Aspect = Format == TextureFormat.Depth32Float ? TextureAspect.DepthOnly : TextureAspect.All
        };
        return _ctx.Wgpu.TextureCreateView(Handle, &viewDesc);
    }

    private void CreateSampler(FilterMode filter, AddressMode addressMode)
    {
        var samplerDesc = new SamplerDescriptor
        {
            AddressModeU = addressMode,
            AddressModeV = addressMode,
            AddressModeW = addressMode,
            MagFilter = filter,
            MinFilter = filter,
            MipmapFilter = filter == FilterMode.Linear ? MipmapFilterMode.Linear : MipmapFilterMode.Nearest,
            LodMinClamp = 0f,
            LodMaxClamp = 32f,
            MaxAnisotropy = 1
        };
        Sampler = _ctx.Wgpu.DeviceCreateSampler(_ctx.Device, &samplerDesc);
    }

    public bool Resize(int width, int height)
    {
        if (Width == width && Height == height && Handle != null)
            return false;

        Allocate(width, height, DepthOrArrayLayers, MipLevelCount);
        return true;
    }

    public void UploadLayer<T>(ReadOnlySpan<T> data, int bytesPerPixel, int layer = 0, uint mipLevel = 0, int mipWidth = 0, int mipHeight = 0) where T : unmanaged
    {
        int w = mipWidth > 0 ? mipWidth : Width;
        int h = mipHeight > 0 ? mipHeight : Height;
        var copyTex = new ImageCopyTexture
        {
            Texture = Handle,
            MipLevel = mipLevel,
            Origin = new Origin3D(0, 0, (uint)layer),
            Aspect = TextureAspect.All
        };
        var layout = new TextureDataLayout
        {
            Offset = 0,
            BytesPerRow = (uint)(w * bytesPerPixel),
            RowsPerImage = (uint)h
        };
        var extent = new Extent3D((uint)w, (uint)h, 1);

        fixed (T* ptr = data)
        {
            _ctx.Wgpu.QueueWriteTexture(_ctx.Queue, &copyTex, ptr, (nuint)(data.Length * sizeof(T)), &layout, &extent);
        }
    }

    public BindGroupEntry AsTextureEntry(uint binding)
    {
        return new BindGroupEntry
        {
            Binding = binding,
            TextureView = View
        };
    }

    public BindGroupEntry AsSamplerEntry(uint binding)
    {
        return new BindGroupEntry
        {
            Binding = binding,
            Sampler = Sampler
        };
    }

    public void Dispose()
    {
        if (Sampler != null) { _ctx.Wgpu.SamplerRelease(Sampler); Sampler = null; }
        if (View != null) { _ctx.Wgpu.TextureViewRelease(View); View = null; }
        if (Handle != null) { _ctx.Wgpu.TextureDestroy(Handle); _ctx.Wgpu.TextureRelease(Handle); Handle = null; }
    }
}
