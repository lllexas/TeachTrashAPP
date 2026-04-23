using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using OpenCvSharp;

namespace TrashAPP.Core.Helpers;

/// <summary>
/// OpenCvSharp Mat 与 WPF BitmapSource 之间的高效转换
/// </summary>
public static class MatToBitmapSourceConverter
{
    /// <summary>
    /// 将 Mat 转换为冻结的 BitmapSource，可直接用于 WPF 绑定（线程安全）
    /// </summary>
    public static BitmapSource? Convert(Mat mat)
    {
        if (mat == null || mat.Empty())
            return null;

        try
        {
            PixelFormat pixelFormat;
            int channels = mat.Channels();
            int bytesPerPixel;

            switch (channels)
            {
                case 3:
                    pixelFormat = PixelFormats.Bgr24;
                    bytesPerPixel = 3;
                    break;
                case 4:
                    pixelFormat = PixelFormats.Bgra32;
                    bytesPerPixel = 4;
                    break;
                case 1:
                    pixelFormat = PixelFormats.Gray8;
                    bytesPerPixel = 1;
                    break;
                default:
                    // 不支持的格式，先转 3 通道 BGR
                    using (var converted = new Mat())
                    {
                        if (channels == 2)
                            Cv2.CvtColor(mat, converted, ColorConversionCodes.GRAY2BGR);
                        else
                            Cv2.CvtColor(mat, converted, ColorConversionCodes.BGRA2BGR);
                        return Convert(converted);
                    }
            }

            int width = mat.Width;
            int height = mat.Height;
            int stride = checked((int)mat.Step()); // 每行字节数（含 padding）

            var bitmap = new WriteableBitmap(width, height, 96, 96, pixelFormat, null);
            bitmap.Lock();
            try
            {
                unsafe
                {
                    byte* srcPtr = (byte*)mat.Ptr(0).ToPointer();
                    byte* dstPtr = (byte*)bitmap.BackBuffer;
                    int dstStride = width * bytesPerPixel;

                    if (stride == dstStride)
                    {
                        // 无 padding，一次性拷贝
                        Buffer.MemoryCopy(srcPtr, dstPtr, stride * height, stride * height);
                    }
                    else
                    {
                        // 有 stride padding，逐行拷贝
                        for (int row = 0; row < height; row++)
                        {
                            byte* rowSrc = (byte*)mat.Ptr(row).ToPointer();
                            byte* rowDst = dstPtr + row * dstStride;
                            Buffer.MemoryCopy(rowSrc, rowDst, dstStride, dstStride);
                        }
                    }
                }
                bitmap.AddDirtyRect(new Int32Rect(0, 0, width, height));
            }
            finally
            {
                bitmap.Unlock();
            }

            bitmap.Freeze(); // 冻结后可跨线程使用
            return bitmap;
        }
        catch
        {
            return null;
        }
    }
}
