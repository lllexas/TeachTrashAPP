namespace TrashAPP.Core.Models;

/// <summary>
/// 输入源类型
/// </summary>
public enum InputMode
{
    /// <summary>单张图片</summary>
    SingleImage,

    /// <summary>单个视频</summary>
    SingleVideo,

    /// <summary>文件夹（所有图片）</summary>
    FolderImages,

    /// <summary>文件夹（所有视频）</summary>
    FolderVideos,

    /// <summary>摄像头流式输入（TODO）</summary>
    CameraStream
}
