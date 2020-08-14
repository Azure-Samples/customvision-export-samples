# Sample script for CustomVision's ONNX Object Detection model

| Task | Domain | Export Platform | Export Flavor |
|------|--------|-----------------|---------------|
| Object Detection | General (compact) | ONNX | null |
| Object Detection | General (compact) | ONNX | OnnxFloat16 |


For the detail of the model export features, please visit [Custom Vision's official documents](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/).

## Set up
- Follow [README](../README.md) to set up ONNX environment.

## How to use
```
python predict.py <model_filepath> <image_filepath>
```

## Notes
There is a slight difference in image preprocessing logic and this script cannot get identical results with Custom Vision's Cloud API.
