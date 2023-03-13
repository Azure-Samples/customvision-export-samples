# Sample script for Custom Vision's ONNX Object Detection model (ML.NET)

| Task | Domain | Export Platform | Export Flavor |
|------|--------|-----------------|---------------|
| Object Detection | General (compact) [S1] | ONNX | null |

## Description

This sample is a C# console application that uses a CustomVision ONNX object detection model in an ML.NET pipeline.

For more information on ML.NET, see [What is ML.NET?](https://learn.microsoft.com/dotnet/machine-learning/how-does-mldotnet-work).

## Set up

See [ML.NET samples README](../README.md) for prerequisites

## How to use

```bash
dotnet run --image_path <image_filepath> --model_path <model_filepath> --labels_path <labels_filepath> --confidence <confidence_value> 
```

- *image_path*: The file path to the image you want to run inference on
- *model_path*: The file path of the exported ONNX model file
- *labels_path*: The file path of the exported labels file
- (Optional) *confidence*: A float value between 0.0 an 1.0 to determine the confidence score used to filter detected bounding boxes
