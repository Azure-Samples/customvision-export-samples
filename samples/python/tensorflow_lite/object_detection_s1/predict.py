import argparse
import pathlib
import numpy as np
import PIL.Image
import tensorflow

PROB_THRESHOLD = 0.01  # Minimum probably to show results.


class Model:
    def __init__(self, model_filepath):
        self.interpreter = tensorflow.lite.Interpreter(model_path=str(model_filepath))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        assert len(self.input_details) == 1 and len(self.output_details) == 3
        self.input_shape = self.input_details[0]['shape'][1:3]

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]

        self.interpreter.set_tensor(self.input_details[0]['index'], input_array)
        self.interpreter.invoke()

        return {detail['name']: self.interpreter.get_tensor(detail['index'])[np.newaxis, ...] for detail in self.output_details}


def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
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
