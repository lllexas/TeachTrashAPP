using System.Collections.Generic;

namespace TrashAPP.Core.Models;

/// <summary>
/// 单张图片的推理结果
/// </summary>
public class ImageInferenceResult
{
    /// <summary></summary>
    public string ImagePath { get; set; } = string.Empty;

    /// <summary></summary>
    public string? OutputImagePath { get; set; }

    /// <summary></summary>
    public List<DetectionItem> Detections { get; set; } = new();

    /// <summary></summary>
    public int KeptCount { get; set; }

    /// <summary></summary>
    public int SkippedBigCount { get; set; }

    /// <summary></summary>
    public bool HasError { get; set; }

    /// <summary></summary>
    public string? ErrorMessage { get; set; }
}

/// <summary>
/// 单个检测目标
/// </summary>
public class DetectionItem
{
    /// <summary></summary>
    public int X1 { get; set; }

    /// <summary></summary>
    public int Y1 { get; set; }

    /// <summary></summary>
    public int X2 { get; set; }

    /// <summary></summary>
    public int Y2 { get; set; }

    /// <summary></summary>
    public float BoxWidth { get; set; }

    /// <summary></summary>
    public float BoxHeight { get; set; }

    /// <summary></summary>
    public float BoxAreaRatio { get; set; }

    /// <summary></summary>
    public float DetectionConfidence { get; set; }

    /// <summary></summary>
    public string ClassName { get; set; } = string.Empty;

    /// <summary></summary>
    public float ClassConfidence { get; set; }

    /// <summary></summary>
    public bool IsBigBox { get; set; }

    /// <summary></summary>
    public bool IsFiltered { get; set; }

    /// <summary>
    /// 位置字符串，如 "(100, 200) - (300, 400)"
    /// </summary>
    public string BoxPosition => $"({X1}, {Y1}) - ({X2}, {Y2})";

    /// <summary>
    /// 状态文本
    /// </summary>
    public string StatusText => IsFiltered ? "已过滤" : "保留";
}
