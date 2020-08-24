import argparse
import pathlib
import coremltools
import numpy as np
import PIL.Image

PROB_THRESHOLD = 0.01  # Minimum probably to show results.


class Model:
    def __init__(self, model_filepath):
        self.model = coremltools.models.MLModel(str(model_filepath))
        spec = self.model.get_spec()
        input_description = spec.description.input[0]
        self.input_name = input_description.name
        self.input_shape = (input_description.type.imageType.width, input_description.type.imageType.height)

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        return self.model.predict({self.input_name: image})


def print_outputs(outputs):
    assert set(outputs.keys()) == set(['coordinates', 'confidence'])
    for box, scores in zip(outputs['coordinates'], outputs['confidence']):
        class_id = np.argmax(scores)
        score = np.max(scores)
        box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        if score > PROB_THRESHOLD:
            print(f"Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filepath', type=pathlib.Path)
    parser.add_argument('image_filepath', type=pathlib.Path)

    args = parser.parse_args()

    model = Model(args.model_filepath)
    outputs = model.predict(args.image_filepath)
    print_outputs(outputs)


if __name__ == '__main__':
    main()
