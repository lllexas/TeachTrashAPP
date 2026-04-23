using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using TrashAPP.Core.Models;

namespace TrashAPP.Core.Services;

/// <summary>
/// 通过 HTTP 调用 FastAPI 后端推理服务
/// </summary>
public class HttpInferenceService : IInferenceService
{
    private readonly HttpClient _httpClient;
    private string _baseUrl;

    public HttpInferenceService(string? baseUrl = null)
    {
        _baseUrl = baseUrl?.TrimEnd('/') ?? "http://127.0.0.1:14785";
        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(60)
        };
    }

    public void UpdateBaseUrl(string? baseUrl)
    {
        _baseUrl = baseUrl?.TrimEnd('/') ?? "http://127.0.0.1:14785";
    }

    /// <summary>
    /// 检查后端服务是否在线
    /// </summary>
    public async Task<bool> HealthCheckAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var response = await _httpClient.GetAsync(
                $"{_baseUrl}/health",
                cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task<ImageInferenceResult> InferSingleAsync(
        string imagePath,
        InferenceParameters parameters,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(imagePath))
        {
            return new ImageInferenceResult
            {
                ImagePath = imagePath,
                HasError = true,
                ErrorMessage = "图片文件不存在"
            };
        }

        try
        {
            using var content = new MultipartFormDataContent();
            using var fileStream = File.OpenRead(imagePath);
            var fileContent = new StreamContent(fileStream);
            fileContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(fileContent, "file", Path.GetFileName(imagePath));

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/predict/image",
                content,
                cancellationToken);

            var json = await response.Content.ReadAsStringAsync(cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                return new ImageInferenceResult
                {
                    ImagePath = imagePath,
                    HasError = true,
                    ErrorMessage = $"HTTP {response.StatusCode}: {json}"
                };
            }

            // 解析后端返回的 JSON
            var backendResult = JsonSerializer.Deserialize<BackendImageResult>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                NumberHandling = JsonNumberHandling.AllowReadingFromString
            });

            if (backendResult == null)
            {
                return new ImageInferenceResult
                {
                    ImagePath = imagePath,
                    HasError = true,
                    ErrorMessage = "无法解析后端返回的 JSON"
                };
            }

            if (!backendResult.Success)
            {
                return new ImageInferenceResult
                {
                    ImagePath = imagePath,
                    HasError = true,
                    ErrorMessage = backendResult.Detail ?? "后端处理失败"
                };
            }

            // 转换为前端使用的格式
            var detections = new List<DetectionItem>();
            if (backendResult.Detections != null)
            {
                foreach (var d in backendResult.Detections)
                {
                    detections.Add(new DetectionItem
                    {
                        X1 = d.Bbox[0],
                        Y1 = d.Bbox[1],
                        X2 = d.Bbox[2],
                        Y2 = d.Bbox[3],
                        BoxWidth = d.Bbox[2] - d.Bbox[0],
                        BoxHeight = d.Bbox[3] - d.Bbox[1],
                        DetectionConfidence = d.DetConf,
                        ClassName = d.Label,
                        ClassConfidence = d.Confidence,
                        IsBigBox = d.IsBigBox,
                        IsFiltered = false,
                    });
                }
            }

            return new ImageInferenceResult
            {
                ImagePath = imagePath,
                OutputImagePath = backendResult.SavePath,
                HasError = false,
                Detections = detections,
                KeptCount = backendResult.DetectionCount,
                SkippedBigCount = 0, // 后端已过滤，前端不再统计
            };
        }
        catch (TaskCanceledException)
        {
            return new ImageInferenceResult
            {
                ImagePath = imagePath,
                HasError = true,
                ErrorMessage = "请求超时，后端处理时间过长"
            };
        }
        catch (HttpRequestException ex)
        {
            return new ImageInferenceResult
            {
                ImagePath = imagePath,
                HasError = true,
                ErrorMessage = $"无法连接后端服务: {ex.Message}\n请确认后端已启动 ({_baseUrl})"
            };
        }
        catch (Exception ex)
        {
            return new ImageInferenceResult
            {
                ImagePath = imagePath,
                HasError = true,
                ErrorMessage = $"请求异常: {ex.Message}"
            };
        }
    }

    public async IAsyncEnumerable<ImageInferenceResult> InferBatchAsync(
        string folderPath,
        InferenceParameters parameters,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var images = GetImageFiles(folderPath);
        foreach (var img in images)
        {
            if (cancellationToken.IsCancellationRequested)
                yield break;

            yield return await InferSingleAsync(img, parameters, cancellationToken);
        }
    }

    private static List<string> GetImageFiles(string folderPath)
    {
        var result = new List<string>();
        var extensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };

        foreach (var file in Directory.GetFiles(folderPath))
        {
            var ext = Path.GetExtension(file).ToLower();
            if (Array.Exists(extensions, e => e == ext))
                result.Add(file);
        }

        return result;
    }

    #region 后端 JSON 模型

    private class BackendImageResult
    {
        [JsonPropertyName("success")]
        public bool Success { get; set; }

        [JsonPropertyName("filename")]
        public string Filename { get; set; } = string.Empty;

        [JsonPropertyName("inference_time_ms")]
        public int InferenceTimeMs { get; set; }

        [JsonPropertyName("detections")]
        public List<BackendDetection>? Detections { get; set; }

        [JsonPropertyName("detection_count")]
        public int DetectionCount { get; set; }

        [JsonPropertyName("save_path")]
        public string? SavePath { get; set; }

        [JsonPropertyName("detail")]
        public string? Detail { get; set; }
    }

    private class BackendDetection
    {
        [JsonPropertyName("label")]
        public string Label { get; set; } = string.Empty;

        [JsonPropertyName("confidence")]
        public float Confidence { get; set; }

        [JsonPropertyName("bbox")]
        public List<int> Bbox { get; set; } = new();

        [JsonPropertyName("det_conf")]
        public float DetConf { get; set; }

        [JsonPropertyName("is_big_box")]
        public bool IsBigBox { get; set; }

        [JsonPropertyName("risk_level")]
        public string RiskLevel { get; set; } = string.Empty;
    }

    #endregion
}
