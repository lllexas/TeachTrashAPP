using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using TrashAPP.Core.Models;

namespace TrashAPP.Core.Services;

/// <summary>
/// 通过调用 Python 进程执行推理的服务
/// </summary>
public class PythonProcessInferenceService : IInferenceService
{
    private readonly string _pythonExecutable;
    private readonly string _inferenceScriptPath;

    public PythonProcessInferenceService(
        string? pythonExecutable = null,
        string? inferenceScriptPath = null)
    {
        _pythonExecutable = pythonExecutable ?? FindPython();
        _inferenceScriptPath = inferenceScriptPath ?? FindInferenceScript();
    }

    public async Task<ImageInferenceResult> InferSingleAsync(
        string imagePath,
        InferenceParameters parameters,
        CancellationToken cancellationToken = default)
    {
        var args = BuildArguments(imagePath, parameters);
        return await RunPythonAsync(args, cancellationToken);
    }

    public async IAsyncEnumerable<ImageInferenceResult> InferBatchAsync(
        string folderPath,
        InferenceParameters parameters,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var args = BuildArguments(folderPath, parameters);
        var result = await RunPythonAsync(args, cancellationToken);

        // Python 脚本批量处理时会输出多张图片的结果
        // 这里假设输出包含多个结果，或者逐张处理
        // 为了简化，先逐张调用单图推理
        var images = Directory.GetFiles(folderPath, "*.*", SearchOption.TopDirectoryOnly);
        foreach (var img in images)
        {
            var ext = Path.GetExtension(img).ToLower();
            if (ext is ".jpg" or ".jpeg" or ".png" or ".bmp" or ".webp")
            {
                if (cancellationToken.IsCancellationRequested)
                    yield break;

                yield return await InferSingleAsync(img, parameters, cancellationToken);
            }
        }
    }

    public async Task<ImageInferenceResult> InferFromBytesAsync(
        byte[] imageBytes,
        string filename,
        InferenceParameters parameters,
        CancellationToken cancellationToken = default)
    {
        var extension = Path.GetExtension(filename);
        if (string.IsNullOrWhiteSpace(extension))
        {
            extension = ".jpg";
        }

        var tempFilePath = Path.Combine(
            Path.GetTempPath(),
            $"{Path.GetFileNameWithoutExtension(filename)}_{Guid.NewGuid():N}{extension}");

        try
        {
            await File.WriteAllBytesAsync(tempFilePath, imageBytes, cancellationToken);
            var result = await InferSingleAsync(tempFilePath, parameters, cancellationToken);
            if (string.IsNullOrWhiteSpace(result.ImagePath))
            {
                result.ImagePath = $"[frame] {filename}";
            }

            return result;
        }
        finally
        {
            try
            {
                if (File.Exists(tempFilePath))
                {
                    File.Delete(tempFilePath);
                }
            }
            catch
            {
            }
        }
    }

    private async Task<ImageInferenceResult> RunPythonAsync(
        string arguments,
        CancellationToken cancellationToken)
    {
        var psi = new ProcessStartInfo
        {
            FileName = _pythonExecutable,
            Arguments = $"\"{_inferenceScriptPath}\" {arguments}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = System.Text.Encoding.UTF8,
            StandardErrorEncoding = System.Text.Encoding.UTF8
        };

        using var process = new Process { StartInfo = psi };
        var tcs = new TaskCompletionSource<bool>();

        process.EnableRaisingEvents = true;
        process.Exited += (_, _) => tcs.TrySetResult(true);

        process.Start();

        using var _ = cancellationToken.Register(() =>
        {
            try { process.Kill(); } catch { /* ignore */ }
            tcs.TrySetCanceled(cancellationToken);
        });

        var outputTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
        var errorTask = process.StandardError.ReadToEndAsync(cancellationToken);

        await Task.WhenAll(tcs.Task, outputTask, errorTask);

        var output = await outputTask;
        var error = await errorTask;

        if (process.ExitCode != 0)
        {
            return new ImageInferenceResult
            {
                HasError = true,
                ErrorMessage = $"Python 进程退出码 {process.ExitCode}: {error}"
            };
        }

        // 尝试解析 JSON 输出
        try
        {
            var result = JsonSerializer.Deserialize<ImageInferenceResult>(output, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                NumberHandling = JsonNumberHandling.AllowReadingFromString
            });

            return result ?? new ImageInferenceResult
            {
                HasError = true,
                ErrorMessage = "无法解析 Python 输出"
            };
        }
        catch (JsonException ex)
        {
            return new ImageInferenceResult
            {
                HasError = true,
                ErrorMessage = $"JSON 解析错误: {ex.Message}"
            };
        }
    }

    private static string BuildArguments(string sourcePath, InferenceParameters p)
    {
        return $"\"{sourcePath}\" " +
               $"--detect-model \"{p.DetectModelPath}\" " +
               $"--cls-model \"{p.ClassifyModelPath}\" " +
               $"--conf {p.DetectConfidence} " +
               $"--imgsz {p.DetectImageSize} " +
               $"--max-ratio {p.MaxBoxAreaRatio} " +
               $"--cls-conf {p.BigBoxMinClassConfidence} " +
               $"--cls-imgsz {p.ClassifyImageSize}";
    }

    private static string FindPython()
    {
        // 尝试常见路径
        var candidates = new[]
        {
            "python",
            "python3",
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs", "Python", "Python311", "python.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs", "Python", "Python310", "python.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs", "Python", "Python39", "python.exe"),
            @"C:\Python311\python.exe",
            @"C:\Python310\python.exe",
            @"C:\Python39\python.exe",
        };

        foreach (var candidate in candidates)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = candidate,
                    Arguments = "--version",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                using var p = Process.Start(psi);
                if (p != null)
                {
                    p.WaitForExit(2000);
                    if (p.ExitCode == 0)
                        return candidate;
                }
            }
            catch { /* ignore */ }
        }

        return "python";
    }

    private static string FindInferenceScript()
    {
        // 优先使用当前目录下的 inference.py
        var local = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "inference.py");
        if (File.Exists(local))
            return local;

        // 或者项目目录
        var projectDir = Path.GetDirectoryName(typeof(PythonProcessInferenceService).Assembly.Location);
        if (projectDir != null)
        {
            var proj = Path.Combine(projectDir, "inference.py");
            if (File.Exists(proj))
                return proj;
        }

        return "inference.py";
    }
}
