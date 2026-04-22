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
    private readonly IInferenceService _inferenceService;
    private CancellationTokenSource? _cts;

    // ========== 模型路径（自动加载，只读显示）==========
    [ObservableProperty]
    private string _detectModelPath = "加载中...";

    [ObservableProperty]
    private string _classifyModelPath = "加载中...";

    [ObservableProperty]
    private bool _modelsReady;

    [ObservableProperty]
    private string? _modelStatusMessage;

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
    private string _statusMessage = "就绪";

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
        && ModelsReady
        && !string.IsNullOrWhiteSpace(SourcePath)
        && (OutputMode == OutputMode.PreviewOnly || !string.IsNullOrWhiteSpace(OutputDir));

    // ========== 结果集合 ==========
    public ObservableCollection<ImageInferenceResult> Results { get; } = new();
    public ObservableCollection<DetectionItem> CurrentDetections { get; } = new();

    // ========== 构造函数 ==========
    public MainViewModel()
    {
        _inferenceService = new PythonProcessInferenceService();
        LoadModels();
    }

    public MainViewModel(IInferenceService inferenceService)
    {
        _inferenceService = inferenceService;
        LoadModels();
    }

    // ========== 模型自动加载 ==========
    private void LoadModels()
    {
        var (ready, detect, classify, error) = ModelAutoLoader.CheckModels();
        ModelsReady = ready;
        DetectModelPath = detect ?? "未找到";
        ClassifyModelPath = classify ?? "未找到";
        ModelStatusMessage = ready
            ? "模型已自动加载"
            : error ?? "模型加载失败";

        if (!ready)
        {
            StatusMessage = $"模型未就绪: {error}";
        }
    }

    [RelayCommand]
    private void RefreshModels()
    {
        LoadModels();
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
                // TODO: 摄像头
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
            var parameters = new InferenceParameters
            {
                DetectModelPath = DetectModelPath,
                ClassifyModelPath = ClassifyModelPath,
                DetectConfidence = 0.25f,
                MaxBoxAreaRatio = 0.30f,
                BigBoxMinClassConfidence = 0.55f
            };

            if (IsFolderInput)
            {
                var images = GetImageFiles(SourcePath);
                TotalImages = images.Count;

                foreach (var imgPath in images)
                {
                    if (_cts.Token.IsCancellationRequested)
                        break;

                    StatusMessage = $"正在识别: {Path.GetFileName(imgPath)}";
                    var result = await _inferenceService.InferSingleAsync(imgPath, parameters, _cts.Token);

                    ProcessResult(result);
                    ProcessedImages++;
                    ProgressValue = (double)ProcessedImages / TotalImages * 100;
                }

                StatusMessage = $"批量识别完成，共处理 {ProcessedImages} 张图片";
            }
            else
            {
                TotalImages = 1;
                var result = await _inferenceService.InferSingleAsync(SourcePath, parameters, _cts.Token);
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

        if (!string.IsNullOrEmpty(result.OutputImagePath) && File.Exists(result.OutputImagePath))
            LoadResultImage(result.OutputImagePath);
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
