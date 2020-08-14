# Sample script for Custom Vision's TensorFlow.js object detection model

| Task | Domain | Export Platform | Export Flavor |
| ---- | ------ | --------------- | ------------- |
| Object Detection | General (compact) | TensorFlow | TensorFlowJs |
| Object Detection | General (compact) [S1] | TensorFlow | TensorFlowJs |


## Setup
### Use Docker (Recommended)
```bash
docker run -it --rm -v <path_to_this_dir>:/work node bash

# Inside the docker,
cd /work
npm install .
```

### Install Node.js manually
Please follow the [Node.js official documents](https://nodejs.org/en/download/).


## Usage
```bash
unzip <model_zip_filepath>
node . <image_filepath>
```

The prediction results will be shown on the stdout.
