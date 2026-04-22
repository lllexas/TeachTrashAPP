using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace TrashAPP.Core.Services;

/// <summary>
/// 本地后端服务启动器
/// </summary>
public class BackendLauncher
{
    private Process? _backendProcess;
    private readonly HttpClient _httpClient;

    public BackendLauncher()
    {
        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(3)
        };
    }

    /// <summary>
    /// 后端进程是否正在运行
    /// </summary>
    public bool IsRunning => _backendProcess != null && !_backendProcess.HasExited;

    /// <summary>
    /// 自动查找后端代码路径
    /// </summary>
    public static string? FindBackendEntry()
    {
        // 1. 相对于 WPF 输出目录的 ../total/main.py
        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
        var candidates = new[]
        {
            Path.Combine(baseDir, "..", "..", "..", "total", "main.py"),
            Path.Combine(baseDir, "..", "..", "total", "main.py"),
            Path.Combine(baseDir, "..", "total", "main.py"),
            Path.Combine(baseDir, "total", "main.py"),
            Path.Combine(Directory.GetCurrentDirectory(), "total", "main.py"),
        };

        foreach (var candidate in candidates)
        {
            var fullPath = Path.GetFullPath(candidate);
            if (File.Exists(fullPath))
                return fullPath;
        }

        return null;
    }

    /// <summary>
    /// 自动查找 Python 解释器
    /// </summary>
    public static string FindPython()
    {
        var candidates = new[]
        {
            "python",
            "python3",
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs", "Python", "Python311", "python.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs", "Python", "Python310", "python.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs", "Python", "Python312", "python.exe"),
            @"C:\Python311\python.exe",
            @"C:\Python310\python.exe",
            @"C:\Python312\python.exe",
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

    /// <summary>
    /// 启动后端服务
    /// </summary>
    /// <param name="onProgress">进度回调 (message)</param>
    /// <param name="cancellationToken">取消令牌</param>
    public async Task<bool> StartAsync(
        Action<string>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        var entryPath = FindBackendEntry();
        if (entryPath == null)
        {
            onProgress?.Invoke("错误: 找不到后端入口文件 (total/main.py)");
            return false;
        }

        var python = FindPython();
        var workingDir = Path.GetDirectoryName(entryPath)!;
        var port = 8000;

        onProgress?.Invoke($"正在启动后端...");
        onProgress?.Invoke($"Python: {python}");
        onProgress?.Invoke($"入口: {entryPath}");

        // 如果已经有后端在跑，先杀掉
        if (IsRunning)
        {
            try { _backendProcess?.Kill(); } catch { }
            _backendProcess = null;
            await Task.Delay(1000, cancellationToken);
        }

        var psi = new ProcessStartInfo
        {
            FileName = python,
            Arguments = $"-m uvicorn main:app --host 0.0.0.0 --port {port} --no-access-log",
            WorkingDirectory = workingDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = System.Text.Encoding.UTF8,
            StandardErrorEncoding = System.Text.Encoding.UTF8
        };

        _backendProcess = new Process { StartInfo = psi };

        _backendProcess.OutputDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                onProgress?.Invoke($"[OUT] {e.Data}");
        };
        _backendProcess.ErrorDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                onProgress?.Invoke($"[ERR] {e.Data}");
        };

        _backendProcess.Start();
        _backendProcess.BeginOutputReadLine();
        _backendProcess.BeginErrorReadLine();

        // 等待后端就绪（轮询 /health）
        onProgress?.Invoke("等待后端就绪...");
        var ready = await WaitForBackendAsync(port, maxRetries: 30, cancellationToken);

        if (ready)
        {
            onProgress?.Invoke("后端启动成功!");
            return true;
        }
        else
        {
            onProgress?.Invoke("后端启动超时，请检查日志");
            try { _backendProcess?.Kill(); } catch { }
            _backendProcess = null;
            return false;
        }
    }

    /// <summary>
    /// 停止后端服务
    /// </summary>
    public void Stop()
    {
        if (_backendProcess != null && !_backendProcess.HasExited)
        {
            try
            {
                _backendProcess.Kill();
                _backendProcess.WaitForExit(3000);
            }
            catch { /* ignore */ }
            finally
            {
                _backendProcess = null;
            }
        }
    }

    /// <summary>
    /// 轮询等待后端就绪
    /// </summary>
    private async Task<bool> WaitForBackendAsync(int port, int maxRetries, CancellationToken cancellationToken)
    {
        for (int i = 0; i < maxRetries; i++)
        {
            if (cancellationToken.IsCancellationRequested)
                return false;

            try
            {
                var response = await _httpClient.GetAsync(
                    $"http://127.0.0.1:{port}/health",
                    cancellationToken);
                if (response.IsSuccessStatusCode)
                    return true;
            }
            catch { /* 还没准备好 */ }

            await Task.Delay(1000, cancellationToken);
        }
        return false;
    }
}
