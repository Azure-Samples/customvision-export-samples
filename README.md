# Sample scripts for exported models from Custom Vision Service.

This repository contains samples scripts to use exported models from [Custom Vision Service](https://customvision.ai).


| Language | Model type | Link |
| -------- | -------- | ---- |
| C#       | ONNX     | [README](samples/csharp/onnx) |
| Javascript | TensorFlow.js | [README](samples/javascript/tensorflowjs) |
| Python   | CoreML   | [README](samples/python/coreml) |
| Python   | ONNX     | [README](samples/python/onnx) |
| Python   | OpenVino | [README](samples/python/openvino) |
| Python   | TensorFlow (Frozen Graph) [^1] | [README](samples/python/tensorflow) |
| Python   | TensorFlow (Saved Model) | [README](samples/python/tensorflow_saved_model) |
| Python   | TensorFlow Lite | [README](samples/python/tensorflow_lite) |

[^1]: This is the default export flavor for TensorFlow platform.

## How to export a model from Custom Vision Service?
Please see this [document](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/export-your-model).


## Notes
Those sample scripts are not aiming to get identical results with Custom Vision's prediction APIs. There are slight differences in the pre-processing logic, which cause small difference in the inference results.


## Related sample projects
| Language | Platform | Repository |
| -------- | -------- | ---------- |
| Java     | Android  | https://github.com/Azure-Samples/cognitive-services-android-customvision-sample |
| Swift, Objective-C | iOS | https://github.com/Azure-Samples/cognitive-services-ios-customvision-sample |
| C#       | WinML    | https://github.com/Azure-Samples/cognitive-services-onnx-customvision-sample |

## Resources
* [Custom Vision Service documents](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/)

