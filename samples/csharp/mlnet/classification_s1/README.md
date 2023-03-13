# Sample script for Custom Vision's ONNX classification model

| Task | Domain | Export Platform | Export Flavor |
|------|--------|-----------------|---------------|
| Classification | General (compact) [S1] | ONNX | null |

For the detail of the model export features, please visit [Custom Vision's official documents](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/).

## Description

This sample is a C# console application that uses a CustomVision ONNX classification model in an ML.NET pipeline.

For more information on ML.NET, see [What is ML.NET?](https://learn.microsoft.com/dotnet/machine-learning/how-does-mldotnet-work).

## Set up

See [ML.NET samples README](../README.md) for prerequisites

## How to use

```bash
dotnet run --image_path <image_filepath> --model_path <model_filepath> --labels_path <labels_filepath> 
```

- *image_path*: The file path to the image you want to run inference on
- *model_path*: The file path of the exported ONNX model file
- *labels_path*: The file path of the exported labels file
