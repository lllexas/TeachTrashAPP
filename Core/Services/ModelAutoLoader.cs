using System.IO;

namespace TrashAPP.Core.Services;

/// <summary>
/// 自动扫描并加载模型文件
/// </summary>
public static class ModelAutoLoader
{
    /// <summary>
    /// YOLO 检测模型目录（相对于程序根目录）
    /// </summary>
    public const string YoloModelDir = "models/YOLO";

    /// <summary>
    /// ResNet 分类模型目录（相对于程序根目录）
    /// </summary>
    public const string ResNetModelDir = "models/ResNet";

    /// <summary>
    /// 自动加载检测模型路径
    /// </summary>
    /// <returns>模型路径，未找到则返回 null</returns>
    public static string? FindDetectModel()
    {
        return FindFirstFile(YoloModelDir);
    }

    /// <summary>
    /// 自动加载分类模型路径
    /// </summary>
    /// <returns>模型路径，未找到则返回 null</returns>
    public static string? FindClassifyModel()
    {
        return FindFirstFile(ResNetModelDir);
    }

    private static string? FindFirstFile(string relativeDir)
    {
        // 优先在程序输出目录查找（exe 同级）
        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
        var path = Path.Combine(baseDir, relativeDir);

        if (Directory.Exists(path))
        {
            var files = Directory.GetFiles(path);
            if (files.Length > 0)
                return files[0];
        }

        // 回退到项目目录（开发时）
        var projectDir = Path.GetDirectoryName(typeof(ModelAutoLoader).Assembly.Location);
        if (projectDir != null)
        {
            path = Path.Combine(projectDir, relativeDir);
            if (Directory.Exists(path))
            {
                var files = Directory.GetFiles(path);
                if (files.Length > 0)
                    return files[0];
            }
        }

        return null;
    }

    /// <summary>
    /// 检查模型是否就绪
    /// </summary>
    public static (bool Ready, string? DetectModel, string? ClassifyModel, string? ErrorMessage) CheckModels()
    {
        var detect = FindDetectModel();
        var classify = FindClassifyModel();

        if (detect == null && classify == null)
            return (false, null, null, $"未找到模型文件。请将模型放入:\n  {YoloModelDir}/\n  {ResNetModelDir}/");

        if (detect == null)
            return (false, null, classify, $"未找到检测模型。请将模型放入: {YoloModelDir}/");

        if (classify == null)
            return (false, null, null, $"未找到分类模型。请将模型放入: {ResNetModelDir}/");

        return (true, detect, classify, null);
    }
}
