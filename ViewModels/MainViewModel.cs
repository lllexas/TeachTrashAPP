using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using OpenCvSharp;
using TrashAPP.Core.Helpers;
using TrashAPP.Core.Models;
using TrashAPP.Core.Services;

namespace TrashAPP.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly HttpInferenceService _httpService;
    private readonly BackendLauncher _backendLauncher;
    private CancellationTokenSource? _cts;

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

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    [NotifyPropertyChangedFor(nameof(IsSingleInput))]
    [NotifyPropertyChangedFor(nameof(IsFolderInput))]
    [NotifyPropertyChangedFor(nameof(IsCameraInput))]
    private InputMode _inputMode = InputMode.SingleImage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    private string _sourcePath = string.Empty;

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

    [ObservableProperty]
    private BitmapSource? _previewImage;

    [ObservableProperty]
    private BitmapSource? _resultImage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasResult))]
    private ImageInferenceResult? _currentResult;

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

    public ObservableCollection<ImageInferenceResult> Results { get; } = new();
    public ObservableCollection<DetectionItem> CurrentDetections { get; } = new();

    public MainViewModel()
    {
        _httpService = new HttpInferenceService(BackendUrl);
        _backendLauncher = new BackendLauncher();
        _ = InitializeAsync();
    }

    private async Task InitializeAsync()
    {
        await TestBackendConnection();
    }

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
                cancellationToken: CancellationToken.None);

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
                    MessageBoxImage.Warning);
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
            ResultImage = null;
        }
    }

    private void SelectFolder(string title)
    {
        var dialog = new OpenFolderDialog { Title = title };

        if (dialog.ShowDialog() == true)
        {
            SourcePath = dialog.FolderName;
            PreviewImage = null;
            ResultImage = null;
        }
    }

    [RelayCommand]
    private void SelectOutputDir()
    {
        var dialog = new OpenFolderDialog { Title = "选择输出目录" };

        if (dialog.ShowDialog() == true)
        {
            OutputDir = dialog.FolderName;
        }
    }

    [RelayCommand]
    private async Task StartInference()
    {
        if (!CanStartInference)
            return;

        _cts = new CancellationTokenSource();
        IsRunning = true;
        Results.Clear();
        CurrentDetections.Clear();
        ProgressValue = 0;
        ProcessedImages = 0;
        TotalImages = 0;
        CurrentResult = null;
        ResultImage = null;

        try
        {
            var parameters = new InferenceParameters();

            switch (InputMode)
            {
                case InputMode.SingleImage:
                    await RunSingleImageInference(parameters);
                    break;
                case InputMode.FolderImages:
                    await RunFolderImagesInference(parameters);
                    break;
                case InputMode.SingleVideo:
                    await RunSingleVideoInference(parameters);
                    break;
                case InputMode.FolderVideos:
                    await RunFolderVideosInference(parameters);
                    break;
                case InputMode.CameraStream:
                    await RunCameraInference(parameters);
                    break;
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

    [RelayCommand]
    private void SelectResultItem(ImageInferenceResult? result)
    {
        if (result == null)
            return;

        CurrentResult = result;
        CurrentDetections.Clear();
        foreach (var det in result.Detections)
        {
            det.BoxAreaRatio = ComputeBoxAreaRatio(det, PreviewImage);
            CurrentDetections.Add(det);
        }

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
            catch
            {
            }
        }

        if (!string.IsNullOrEmpty(result.ImagePath) && File.Exists(result.ImagePath))
        {
            LoadResultImage(result.ImagePath);
        }
    }

    private async Task RunSingleImageInference(InferenceParameters parameters)
    {
        StatusMessage = "正在识别图片...";
        ProgressValue = 0;
        TotalImages = 1;

        if (File.Exists(SourcePath))
        {
            LoadPreview(SourcePath);
        }

        var result = await _httpService.InferSingleAsync(SourcePath, parameters, _cts!.Token);
        await ProcessSingleResultAsync(result, SourcePath);

        ProcessedImages = 1;
        ProgressValue = 100;
        StatusMessage = result.HasError ? $"识别失败: {result.ErrorMessage}" : "图片识别完成";
    }

    private async Task RunFolderImagesInference(InferenceParameters parameters)
    {
        var images = GetImageFiles(SourcePath);
        TotalImages = images.Count;

        if (TotalImages == 0)
        {
            StatusMessage = "所选文件夹中没有可识别的图片";
            return;
        }

        StatusMessage = $"开始批量识别，共 {TotalImages} 张图片";

        for (var i = 0; i < images.Count; i++)
        {
            _cts!.Token.ThrowIfCancellationRequested();

            var path = images[i];
            LoadPreview(path);
            StatusMessage = $"正在处理第 {i + 1}/{TotalImages} 张";

            var result = await _httpService.InferSingleAsync(path, parameters, _cts.Token);
            await ProcessSingleResultAsync(result, path);

            ProcessedImages = i + 1;
            ProgressValue = TotalImages == 0 ? 0 : (double)ProcessedImages / TotalImages * 100;
        }

        StatusMessage = $"批量图片识别完成，共处理 {ProcessedImages} 张";
    }

    private async Task RunSingleVideoInference(InferenceParameters parameters)
    {
        StatusMessage = "正在播放视频并按最新帧识别...";
        TotalImages = 1;
        await ProcessVideoAsync(SourcePath, parameters, _cts!.Token);
        ProcessedImages = 1;
        ProgressValue = 100;
        StatusMessage = "视频播放结束";
    }

    private async Task RunFolderVideosInference(InferenceParameters parameters)
    {
        var videos = GetVideoFiles(SourcePath);
        TotalImages = videos.Count;

        if (TotalImages == 0)
        {
            StatusMessage = "所选文件夹中没有可识别的视频";
            return;
        }

        for (var i = 0; i < videos.Count; i++)
        {
            _cts!.Token.ThrowIfCancellationRequested();
            StatusMessage = $"正在处理视频 {i + 1}/{TotalImages}: {Path.GetFileName(videos[i])}";
            await ProcessVideoAsync(videos[i], parameters, _cts.Token);
            ProcessedImages = i + 1;
            ProgressValue = (double)ProcessedImages / TotalImages * 100;
        }

        StatusMessage = $"批量视频处理完成，共处理 {ProcessedImages} 个视频";
    }

    private Task RunCameraInference(InferenceParameters parameters)
    {
        StatusMessage = "摄像头模式尚未实现，目前已打通图片与视频的单帧异步识别链路";
        return Task.CompletedTask;
    }

    private async Task ProcessVideoAsync(string videoPath, InferenceParameters parameters, CancellationToken cancellationToken)
    {
        using var capture = new VideoCapture(videoPath);
        if (!capture.IsOpened())
        {
            throw new InvalidOperationException($"无法打开视频: {videoPath}");
        }

        var fps = capture.Fps;
        if (fps <= 0 || double.IsNaN(fps))
        {
            fps = 25;
        }

        var delayMs = Math.Max(1, (int)Math.Round(1000d / fps));
        var frameIndex = 0;
        Task? pendingInference = null;

        using var frame = new Mat();
        while (!cancellationToken.IsCancellationRequested)
        {
            if (!capture.Read(frame) || frame.Empty())
            {
                break;
            }

            frameIndex++;

            using var displayFrame = frame.Clone();
            await UpdatePreviewAsync(displayFrame);

            if (pendingInference == null || pendingInference.IsCompleted)
            {
                using var inferenceFrame = frame.Clone();
                var bytes = EncodeFrame(inferenceFrame);
                var inferenceFrameIndex = frameIndex;
                pendingInference = HandleFrameInferenceAsync(bytes, videoPath, inferenceFrameIndex, parameters, cancellationToken);
            }

            await Task.Delay(delayMs, cancellationToken);
        }

        if (pendingInference != null)
        {
            await pendingInference;
        }
    }

    private async Task HandleFrameInferenceAsync(
        byte[] imageBytes,
        string sourceName,
        int frameIndex,
        InferenceParameters parameters,
        CancellationToken cancellationToken)
    {
        var frameLabel = $"{Path.GetFileNameWithoutExtension(sourceName)}_frame_{frameIndex:D6}.jpg";
        var result = await _httpService.InferFromBytesAsync(imageBytes, frameLabel, parameters, cancellationToken);
        await ProcessFrameResultAsync(result, imageBytes, frameLabel);
    }

    private async Task ProcessSingleResultAsync(ImageInferenceResult result, string imagePath)
    {
        using var source = Cv2.ImRead(imagePath);
        if (!source.Empty())
        {
            using var annotated = DrawDetections(source, result.Detections);
            await UpdateResultAsync(annotated);
        }

        await UiAsync(() =>
        {
            ProcessResult(result);
        });
    }

    private async Task ProcessFrameResultAsync(ImageInferenceResult result, byte[] imageBytes, string imageLabel)
    {
        using var frame = Cv2.ImDecode(imageBytes, ImreadModes.Color);
        if (!frame.Empty())
        {
            using var annotated = DrawDetections(frame, result.Detections);
            await UpdateResultAsync(annotated);
        }

        await UiAsync(() =>
        {
            result.ImagePath = imageLabel;
            ProcessResult(result);
            StatusMessage = result.HasError
                ? $"帧识别失败: {result.ErrorMessage}"
                : $"已更新识别结果: {Path.GetFileName(imageLabel)}";
        });
    }

    private void ProcessResult(ImageInferenceResult result)
    {
        Results.Add(result);
        if (!result.HasError)
        {
            SelectResultItem(result);
        }
    }

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

    private async Task UpdatePreviewAsync(Mat frame)
    {
        var bitmap = MatToBitmapSourceConverter.Convert(frame);
        await UiAsync(() => PreviewImage = bitmap);
    }

    private async Task UpdateResultAsync(Mat frame)
    {
        var bitmap = MatToBitmapSourceConverter.Convert(frame);
        await UiAsync(() => ResultImage = bitmap);
    }

    private static byte[] EncodeFrame(Mat frame)
    {
        return frame.ImEncode(".jpg", new[] { (int)ImwriteFlags.JpegQuality, 90 });
    }

    private static Mat DrawDetections(Mat source, IReadOnlyCollection<DetectionItem> detections)
    {
        var canvas = source.Clone();
        foreach (var det in detections)
        {
            var rect = new OpenCvSharp.Rect(det.X1, det.Y1, Math.Max(1, det.X2 - det.X1), Math.Max(1, det.Y2 - det.Y1));
            Cv2.Rectangle(canvas, rect, Scalar.LimeGreen, 2);

            var label = $"{det.ClassName}  det:{det.DetectionConfidence:F2}  cls:{det.ClassConfidence:F2}";
            var labelPoint = new OpenCvSharp.Point(det.X1, Math.Max(24, det.Y1 - 8));
            Cv2.PutText(canvas, label, labelPoint, HersheyFonts.HersheySimplex, 0.6, Scalar.Yellow, 2);
        }

        return canvas;
    }

    private static float ComputeBoxAreaRatio(DetectionItem det, BitmapSource? image)
    {
        if (image == null || image.PixelWidth <= 0 || image.PixelHeight <= 0)
        {
            return 0;
        }

        var imageArea = image.PixelWidth * image.PixelHeight;
        if (imageArea <= 0)
        {
            return 0;
        }

        var boxArea = Math.Max(0, det.X2 - det.X1) * Math.Max(0, det.Y2 - det.Y1);
        return (float)boxArea / imageArea;
    }

    private static async Task UiAsync(Action action)
    {
        var dispatcher = Application.Current?.Dispatcher;
        if (dispatcher == null || dispatcher.CheckAccess())
        {
            action();
            return;
        }

        await dispatcher.InvokeAsync(action);
    }

    private static List<string> GetImageFiles(string folderPath)
    {
        var extensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
        return Directory
            .GetFiles(folderPath)
            .Where(file => extensions.Contains(Path.GetExtension(file).ToLowerInvariant()))
            .OrderBy(file => file, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static List<string> GetVideoFiles(string folderPath)
    {
        var extensions = new[] { ".mp4", ".avi", ".mkv", ".mov", ".wmv" };
        return Directory
            .GetFiles(folderPath)
            .Where(file => extensions.Contains(Path.GetExtension(file).ToLowerInvariant()))
            .OrderBy(file => file, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }
}
