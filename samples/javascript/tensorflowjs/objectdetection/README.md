# Sample script for Custom Vision's TensorFlow.js Object Detection model

| Task | Domain | Export Platform | Export Flavor |
| ---- | ------ | --------------- | ------------- |
| Object Detection | General (compact) | TensorFlow | TensorFlowJs |
| Object Detection | General (compact) [S1] | TensorFlow | TensorFlowJs |

This script loads the customvision-tfjs library from CDN.

## Setup
1. Extract models in this directory
```bash
unzip <model_zip_filepath>
```

2. Serve this directory over HTTP.

```bash
# If you have python,
python -m http.server 8080

# If you have Node.js,
npx http-server -p 8080
```

## Usage
1. Open http://localhost:8080/ with your favorite browser.
2. Choose a test image.
