using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TrashAPP.Core.Models;

namespace TrashAPP.Core.Services;

/// <summary>
/// 推理服务接口
/// </summary>
public interface IInferenceService
{
    /// <summary>
    /// 对单张图片进行推理
    /// </summary>
    Task<ImageInferenceResult> InferSingleAsync(
        string imagePath,
        InferenceParameters parameters,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// 对文件夹中的所有图片进行批量推理
    /// </summary>
    IAsyncEnumerable<ImageInferenceResult> InferBatchAsync(
        string folderPath,
        InferenceParameters parameters,
        CancellationToken cancellationToken = default);
}
