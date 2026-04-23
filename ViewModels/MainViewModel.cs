using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using TrashAPP.Core.Models;
using TrashAPP.Core.Services;

namespace TrashAPP.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly HttpInferenceService _httpService;
    private readonly BackendLauncher _backendLauncher;
    private CancellationTokenSource? _cts;

    // ========== 后端连接 ==========
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    private string _backendUrl = "http://127.0.0.1:14785";

    [ObservableProperty]
    private bool _backendConnected;

    [ObservableProperty]
    private string? _backendStatusMessage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowStartBackendButton))]
    private bool _isStartingBackend;

    public bool ShowStartBackendButton => !BackendConnected && !IsStartingBackend;

    // ========== 输入配置 ==========
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    [NotifyPropertyChangedFor(nameof(IsSingleInput))]
    [NotifyPropertyChangedFor(nameof(IsFolderInput))]
    [NotifyPropertyChangedFor(nameof(IsCameraInput))]
    private InputMode _inputMode = InputMode.SingleImage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    private string _sourcePath = string.Empty;

    // ========== 输出配置 ==========
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    [NotifyPropertyChangedFor(nameof(ShowOutputDir))]
    [NotifyPropertyChangedFor(nameof(ShowPeriodicOptions))]
    private OutputMode _outputMode = OutputMode.SaveToFile;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    private string _outputDir = string.Empty;

    [ObservableProperty]
    private int _snapshotIntervalSeconds = 5;

    // ========== 运行状态 ==========
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    private bool _isRunning;

    [ObservableProperty]
    private string _statusMessage = "就绪 - 正在检测后端...";

    [ObservableProperty]
    private double _progressValue;

    [ObservableProperty]
    private int _totalImages;

    [ObservableProperty]
    private int _processedImages;

    // ========== 预览 ==========
    [ObservableProperty]
    private BitmapSource? _previewImage;

    [ObservableProperty]
    private BitmapSource? _resultImage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasResult))]
    private ImageInferenceResult? _currentResult;

    // ========== 计算属性 ==========
    public bool HasResult => CurrentResult != null && !CurrentResult.HasError;

    public bool IsSingleInput => InputMode is InputMode.SingleImage or InputMode.SingleVideo;

    public bool IsFolderInput => InputMode is InputMode.FolderImages or InputMode.FolderVideos;

    public bool IsCameraInput => InputMode == InputMode.CameraStream;

    public bool ShowOutputDir => OutputMode != OutputMode.PreviewOnly;

    public bool ShowPeriodicOptions => OutputMode == OutputMode.PeriodicSnapshot;

    public bool CanStartInference =>
        !IsRunning
        && BackendConnected
        && !string.IsNullOrWhiteSpace(SourcePath)
        && (OutputMode == OutputMode.PreviewOnly || !string.IsNullOrWhiteSpace(OutputDir));

    // ========== 结果集合 ==========
    public ObservableCollection<ImageInferenceResult> Results { get; } = new();
    public ObservableCollection<DetectionItem> CurrentDetections { get; } = new();

    // ========== 构造函数 ==========
    public MainViewModel()
    {
        _httpService = new HttpInferenceService(BackendUrl);
        _backendLauncher = new BackendLauncher();
        _ = InitializeAsync();
    }

    private async Task InitializeAsync()
    {
        // 启动时自动检测后端
        await TestBackendConnection();
    }

    // ========== 一键启动后端 ==========
    [RelayCommand]
    private async Task StartBackend()
    {
        if (IsStartingBackend)
            return;

        IsStartingBackend = true;
        StatusMessage = "正在启动后端服务...";

        try
        {
            var success = await _backendLauncher.StartAsync(
                onProgress: msg => StatusMessage = msg,
                cancellationToken: CancellationToken.None
            );

            if (success)
            {
                await TestBackendConnection();
            }
            else
            {
                StatusMessage = "后端启动失败，请检查 Python 环境和依赖";
                MessageBox.Show(
                    "后端启动失败。请确认:\n1. Python 已安装\n2. 依赖已安装 (pip install -r requirements.txt)\n3. 模型文件已放入 models/ 目录",
                    "启动失败",
                    MessageBoxButton.OK,
                    MessageBoxImage.Warning
                );
            }
        }
        catch (Exception ex)
        {
            StatusMessage = $"启动异常: {ex.Message}";
        }
        finally
        {
            IsStartingBackend = false;
        }
    }

    // ========== 后端连接测试 ==========
    [RelayCommand]
    private async Task TestBackendConnection()
    {
        StatusMessage = $"正在连接 {BackendUrl}...";
        _httpService.UpdateBaseUrl(BackendUrl);

        try
        {
            var isHealthy = await _httpService.HealthCheckAsync();
            BackendConnected = isHealthy;
            BackendStatusMessage = isHealthy ? "后端连接正常" : "后端无响应";
            StatusMessage = isHealthy
                ? $"后端连接成功: {BackendUrl}"
                : $"后端未运行: {BackendUrl}";
        }
        catch (Exception ex)
        {
            BackendConnected = false;
            BackendStatusMessage = "连接异常";
            StatusMessage = $"连接异常: {ex.Message}";
        }
    }

    // ========== 输入选择 ==========
    [RelayCommand]
    private void SelectSource()
    {
        switch (InputMode)
        {
            case InputMode.SingleImage:
                SelectSingleImage();
                break;
            case InputMode.SingleVideo:
                SelectSingleVideo();
                break;
            case InputMode.FolderImages:
                SelectFolder("选择包含图片的文件夹");
                break;
            case InputMode.FolderVideos:
                SelectFolder("选择包含视频的文件夹");
                break;
            case InputMode.CameraStream:
                SourcePath = "摄像头输入（TODO）";
                break;
        }
    }

    private void SelectSingleImage()
    {
        var dialog = new OpenFileDialog
        {
            Filter = "图片文件|*.jpg;*.jpeg;*.png;*.bmp;*.webp|所有文件|*.*",
            Title = "选择要识别的图片"
        };

        if (dialog.ShowDialog() == true)
        {
            SourcePath = dialog.FileName;
            LoadPreview(SourcePath);
        }
    }

    private void SelectSingleVideo()
    {
        var dialog = new OpenFileDialog
        {
            Filter = "视频文件|*.mp4;*.avi;*.mkv;*.mov;*.wmv|所有文件|*.*",
            Title = "选择要识别的视频"
        };

        if (dialog.ShowDialog() == true)
        {
            SourcePath = dialog.FileName;
            PreviewImage = null;
        }
    }

    private void SelectFolder(string title)
    {
        var dialog = new OpenFolderDialog { Title = title };

        if (dialog.ShowDialog() == true)
        {
            SourcePath = dialog.FolderName;
            PreviewImage = null;
        }
    }

    // ========== 输出目录 ==========
    [RelayCommand]
    private void SelectOutputDir()
    {
        var dialog = new OpenFolderDialog { Title = "选择输出目录" };

        if (dialog.ShowDialog() == true)
        {
            OutputDir = dialog.FolderName;
        }
    }

    // ========== 推理 ==========
    [RelayCommand]
    private async Task StartInference()
    {
        if (!CanStartInference)
            return;

        if (InputMode == InputMode.CameraStream)
        {
            MessageBox.Show("摄像头流式输入功能尚未实现", "TODO", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        if (InputMode == InputMode.SingleVideo || InputMode == InputMode.FolderVideos)
        {
            MessageBox.Show("视频输入功能尚未实现", "TODO", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        _cts = new CancellationTokenSource();
        IsRunning = true;
        Results.Clear();
        CurrentDetections.Clear();
        ProgressValue = 0;
        ProcessedImages = 0;
        CurrentResult = null;
        ResultImage = null;

        try
        {
            var parameters = new InferenceParameters();

            if (IsFolderInput)
            {
                var images = GetImageFiles(SourcePath);
                TotalImages = images.Count;

                foreach (var imgPath in images)
                {
                    if (_cts.Token.IsCancellationRequested)
                        break;

                    StatusMessage = $"正在识别: {Path.GetFileName(imgPath)}";
                    var result = await _httpService.InferSingleAsync(imgPath, parameters, _cts.Token);

                    ProcessResult(result);
                    ProcessedImages++;
                    ProgressValue = (double)ProcessedImages / TotalImages * 100;
                }

                StatusMessage = $"批量识别完成，共处理 {ProcessedImages} 张图片";
            }
            else
            {
                TotalImages = 1;
                var result = await _httpService.InferSingleAsync(SourcePath, parameters, _cts.Token);
                ProcessResult(result);
                ProcessedImages = 1;
                ProgressValue = 100;
                StatusMessage = result.HasError
                    ? $"识别失败: {result.ErrorMessage}"
                    : $"识别完成，检测到 {result.KeptCount} 个目标";
            }
        }
        catch (OperationCanceledException)
        {
            StatusMessage = "已取消";
        }
        catch (Exception ex)
        {
            StatusMessage = $"错误: {ex.Message}";
            MessageBox.Show($"推理过程中发生错误:\n{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            IsRunning = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    [RelayCommand]
    private void CancelInference()
    {
        _cts?.Cancel();
        StatusMessage = "正在取消...";
    }

    // ========== 结果处理 ==========
    [RelayCommand]
    private void SelectResultItem(ImageInferenceResult? result)
    {
        if (result == null) return;

        CurrentResult = result;
        CurrentDetections.Clear();
        foreach (var det in result.Detections)
            CurrentDetections.Add(det);

        // 尝试从后端返回的 save_path 加载结果图
        if (!string.IsNullOrEmpty(result.OutputImagePath))
        {
            try
            {
                if (File.Exists(result.OutputImagePath))
                {
                    LoadResultImage(result.OutputImagePath);
                    return;
                }
            }
            catch { }
        }

        // 如果无法加载结果图，显示原图
        if (!string.IsNullOrEmpty(result.ImagePath) && File.Exists(result.ImagePath))
        {
            LoadResultImage(result.ImagePath);
        }
    }

    private void ProcessResult(ImageInferenceResult result)
    {
        Results.Add(result);
        if (!result.HasError)
            SelectResultItem(result);
    }

    // ========== 图片加载 ==========
    private void LoadPreview(string path)
    {
        try
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = new Uri(path);
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.EndInit();
            bitmap.Freeze();
            PreviewImage = bitmap;
        }
        catch
        {
            PreviewImage = null;
        }
    }

    private void LoadResultImage(string path)
    {
        try
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = new Uri(path);
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.EndInit();
            bitmap.Freeze();
            ResultImage = bitmap;
        }
        catch
        {
            ResultImage = null;
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
}
