using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
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

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    private string _sourcePath = string.Empty;

    [ObservableProperty]
    private bool _isFolder;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    private string _detectModelPath = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    private string _classifyModelPath = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStartInference))]
    private string _pythonPath = "python";

    [ObservableProperty]
    private float _detectConfidence = 0.25f;

    [ObservableProperty]
    private float _maxBoxAreaRatio = 0.30f;

    [ObservableProperty]
    private float _bigBoxMinClassConfidence = 0.55f;

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

    [ObservableProperty]
    private BitmapSource? _previewImage;

    [ObservableProperty]
    private BitmapSource? _resultImage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasResult))]
    private ImageInferenceResult? _currentResult;

    public bool HasResult => CurrentResult != null && !CurrentResult.HasError;

    public bool CanStartInference =>
        !IsRunning
        && !string.IsNullOrWhiteSpace(SourcePath)
        && !string.IsNullOrWhiteSpace(DetectModelPath)
        && !string.IsNullOrWhiteSpace(ClassifyModelPath);

    public ObservableCollection<ImageInferenceResult> Results { get; } = new();
    public ObservableCollection<DetectionItem> CurrentDetections { get; } = new();

    public MainViewModel()
    {
        _inferenceService = new PythonProcessInferenceService();
    }

    public MainViewModel(IInferenceService inferenceService)
    {
        _inferenceService = inferenceService;
    }

    [RelayCommand]
    private void SelectImage()
    {
        var dialog = new OpenFileDialog
        {
            Filter = "图片文件|*.jpg;*.jpeg;*.png;*.bmp;*.webp|所有文件|*.*",
            Title = "选择要识别的图片"
        };

        if (dialog.ShowDialog() == true)
        {
            SourcePath = dialog.FileName;
            IsFolder = false;
            LoadPreview(SourcePath);
        }
    }

    [RelayCommand]
    private void SelectFolder()
    {
        var dialog = new OpenFolderDialog
        {
            Title = "选择包含图片的文件夹"
        };

        if (dialog.ShowDialog() == true)
        {
            SourcePath = dialog.FolderName;
            IsFolder = true;
            PreviewImage = null;
        }
    }

    [RelayCommand]
    private void SelectDetectModel()
    {
        var dialog = new OpenFileDialog
        {
            Filter = "YOLO 模型|*.pt|所有文件|*.*",
            Title = "选择检测模型"
        };

        if (dialog.ShowDialog() == true)
        {
            DetectModelPath = dialog.FileName;
        }
    }

    [RelayCommand]
    private void SelectClassifyModel()
    {
        var dialog = new OpenFileDialog
        {
            Filter = "分类模型|*.pth;*.pt|所有文件|*.*",
            Title = "选择分类模型"
        };

        if (dialog.ShowDialog() == true)
        {
            ClassifyModelPath = dialog.FileName;
        }
    }

    [RelayCommand]
    private async Task StartInference()
    {
        if (!CanStartInference)
            return;

        _cts = new CancellationTokenSource();
        IsRunning = true;
        StatusMessage = IsFolder ? "开始批量识别..." : "开始识别...";
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
                DetectConfidence = DetectConfidence,
                MaxBoxAreaRatio = MaxBoxAreaRatio,
                BigBoxMinClassConfidence = BigBoxMinClassConfidence
            };

            if (IsFolder)
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

    [RelayCommand]
    private void SelectResultItem(ImageInferenceResult? result)
    {
        if (result == null)
            return;

        CurrentResult = result;
        CurrentDetections.Clear();
        foreach (var det in result.Detections)
        {
            CurrentDetections.Add(det);
        }

        if (!string.IsNullOrEmpty(result.OutputImagePath) && File.Exists(result.OutputImagePath))
        {
            LoadResultImage(result.OutputImagePath);
        }
    }

    private void ProcessResult(ImageInferenceResult result)
    {
        Results.Add(result);

        if (!result.HasError)
        {
            // 自动显示最新结果
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

    private static List<string> GetImageFiles(string folderPath)
    {
        var result = new List<string>();
        var extensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };

        foreach (var file in Directory.GetFiles(folderPath))
        {
            var ext = Path.GetExtension(file).ToLower();
            if (Array.Exists(extensions, e => e == ext))
            {
                result.Add(file);
            }
        }

        return result;
    }
}
