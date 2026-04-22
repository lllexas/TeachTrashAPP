namespace TrashAPP.Core.Models;

/// <summary>
/// 推理参数配置
/// </summary>
public class InferenceParameters
{
    /// <summary>
    /// 检测置信度阈值（默认 0.25）
    /// </summary>
    public float DetectConfidence { get; set; } = 0.25f;

    /// <summary>
    /// 检测输入尺寸（默认 768）
    /// </summary>
    public int DetectImageSize { get; set; } = 768;

    /// <summary>
    /// 大框面积比例阈值（默认 0.30）
    /// </summary>
    public float MaxBoxAreaRatio { get; set; } = 0.30f;

    /// <summary>
    /// 大框最小分类置信度（默认 0.55）
    /// </summary>
    public float BigBoxMinClassConfidence { get; set; } = 0.55f;

    /// <summary>
    /// 分类模型输入尺寸（默认 224）
    /// </summary>
    public int ClassifyImageSize { get; set; } = 224;

    /// <summary>
    /// 检测模型路径
    /// </summary>
    public string DetectModelPath { get; set; } = string.Empty;

    /// <summary>
    /// 分类模型路径
    /// </summary>
    public string ClassifyModelPath { get; set; } = string.Empty;
}
