using System.Runtime.InteropServices;

namespace SurfelGI.Scene;

public static unsafe class ExrReader
{
    private static IntPtr _libHandle;
    private static bool _initialized;

    private static delegate* unmanaged[Cdecl]<byte*, IntPtr> _imfOpenInputFile;
    private static delegate* unmanaged[Cdecl]<IntPtr, int> _imfCloseInputFile;
    private static delegate* unmanaged[Cdecl]<IntPtr, IntPtr> _imfInputHeader;
    private static delegate* unmanaged[Cdecl]<IntPtr, int*, int*, int*, int*, void> _imfHeaderDataWindow;
    private static delegate* unmanaged[Cdecl]<IntPtr, ushort*, nuint, nuint, int> _imfInputSetFrameBuffer;
    private static delegate* unmanaged[Cdecl]<IntPtr, int, int, int> _imfInputReadPixels;

    private static bool EnsureLoaded()
    {
        if (_initialized) return _libHandle != IntPtr.Zero;
        _initialized = true;

        string[] candidates =
        {
            "libOpenEXR-3_2.so.31",
            "libOpenEXR.so",
            "OpenEXR",
            "/usr/lib64/libOpenEXR-3_2.so.31"
        };

        foreach (string name in candidates)
        {
            if (NativeLibrary.TryLoad(name, out _libHandle))
            {
                _imfOpenInputFile = (delegate* unmanaged[Cdecl]<byte*, IntPtr>)NativeLibrary.GetExport(_libHandle, "ImfOpenInputFile");
                _imfCloseInputFile = (delegate* unmanaged[Cdecl]<IntPtr, int>)NativeLibrary.GetExport(_libHandle, "ImfCloseInputFile");
                _imfInputHeader = (delegate* unmanaged[Cdecl]<IntPtr, IntPtr>)NativeLibrary.GetExport(_libHandle, "ImfInputHeader");
                _imfHeaderDataWindow = (delegate* unmanaged[Cdecl]<IntPtr, int*, int*, int*, int*, void>)NativeLibrary.GetExport(_libHandle, "ImfHeaderDataWindow");
                _imfInputSetFrameBuffer = (delegate* unmanaged[Cdecl]<IntPtr, ushort*, nuint, nuint, int>)NativeLibrary.GetExport(_libHandle, "ImfInputSetFrameBuffer");
                _imfInputReadPixels = (delegate* unmanaged[Cdecl]<IntPtr, int, int, int>)NativeLibrary.GetExport(_libHandle, "ImfInputReadPixels");
                return true;
            }
        }
        return false;
    }

    public static CpuTextureData Load(string path)
    {
        if (EnsureLoaded())
        {
            byte[] pathBytes = System.Text.Encoding.UTF8.GetBytes(path + "\0");
            fixed (byte* pPath = pathBytes)
            {
                IntPtr file = _imfOpenInputFile(pPath);
                if (file != IntPtr.Zero)
                {
                    try
                    {
                        IntPtr header = _imfInputHeader(file);
                        int xMin, yMin, xMax, yMax;
                        _imfHeaderDataWindow(header, &xMin, &yMin, &xMax, &yMax);

                        int width = xMax - xMin + 1;
                        int height = yMax - yMin + 1;
                        ushort[] halfPixels = new ushort[width * height * 4];

                        fixed (ushort* pBuf = halfPixels)
                        {
                            ushort* basePtr = pBuf - (yMin * width + xMin) * 4;
                            _imfInputSetFrameBuffer(file, basePtr, 1, (nuint)width);
                            _imfInputReadPixels(file, yMin, yMax);
                        }

                        float[] floatPixels = new float[width * height * 4];
                        for (int i = 0; i < halfPixels.Length; i++)
                        {
                            float val = (float)BitConverter.UInt16BitsToHalf(halfPixels[i]);
                            floatPixels[i] = float.IsFinite(val) ? val : 0f;
                        }

                        return new CpuTextureData
                        {
                            Name = Path.GetFileName(path),
                            Width = width,
                            Height = height,
                            Rgba32FloatData = floatPixels,
                            FlipY = false
                        };
                    }
                    finally
                    {
                        _imfCloseInputFile(file);
                    }
                }
            }
        }

        // Fallback procedural sky if native OpenEXR is unavailable
        const int fallbackW = 512;
        const int fallbackH = 256;
        float[] fallback = new float[fallbackW * fallbackH * 4];
        for (int y = 0; y < fallbackH; y++)
        {
            float v = (float)y / fallbackH;
            for (int x = 0; x < fallbackW; x++)
            {
                int idx = (y * fallbackW + x) * 4;
                fallback[idx + 0] = MathF.Max(0.05f, 1.0f - v);
                fallback[idx + 1] = MathF.Max(0.08f, 1.2f - v * 0.9f);
                fallback[idx + 2] = MathF.Max(0.15f, 1.8f - v * 0.7f);
                fallback[idx + 3] = 1.0f;
            }
        }
        return new CpuTextureData
        {
            Name = Path.GetFileName(path),
            Width = fallbackW,
            Height = fallbackH,
            Rgba32FloatData = fallback,
            FlipY = false
        };
    }
}
