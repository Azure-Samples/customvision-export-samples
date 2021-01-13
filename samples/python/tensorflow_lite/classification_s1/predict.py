import argparse
import pathlib
import numpy as np
import tensorflow
import PIL.Image


class Model:
    def __init__(self, model_filepath):
        self.interpreter = tensorflow.lite.Interpreter(model_path=str(model_filepath))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        assert len(self.input_details) == 1 and len(self.output_details) == 1
        self.input_shape = self.input_details[0]['shape'][1:3]

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]

        self.interpreter.set_tensor(self.input_details[0]['index'], input_array)
        self.interpreter.invoke()

        return {detail['name']: self.interpreter.get_tensor(detail['index']) for detail in self.output_details}


def print_outputs(outputs):
    outputs = list(outputs.values())[0]
    for index, score in enumerate(outputs[0]):
        print(f"Label: {index}, score: {score:.5f}")


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
