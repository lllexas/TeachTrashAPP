namespace TrashAPP.Core.Models;

/// <summary>
/// 输出模式
/// </summary>
public enum OutputMode
{
    /// <summary>保存为文件（图片/视频）</summary>
    SaveToFile,

    /// <summary>仅预览，不保存（用于摄像头）</summary>
    PreviewOnly,

    /// <summary>周期性截图（用于摄像头）</summary>
    PeriodicSnapshot,

    /// <summary>录制为 MKV 视频（用于摄像头）</summary>
    RecordMkv
}
