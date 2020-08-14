import argparse
import pathlib
import numpy as np
import tensorflow
import PIL.Image


class Model:
    def __init__(self, model_dirpath):
        model = tensorflow.saved_model.load(str(model_dirpath))
        self.serve = model.signatures['serving_default']
        self.input_shape = self.serve.inputs[0].shape[1:3]

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array[:, :, :, (2, 1, 0)]  # => BGR

        input_tensor = tensorflow.convert_to_tensor(input_array)
        return self.serve(input_tensor)


def print_outputs(outputs):
    outputs = list(outputs.values())[0]
    for index, score in enumerate(outputs[0]):
        print(f"Label: {index}, score: {score:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dirpath', type=pathlib.Path)
    parser.add_argument('image_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if args.model_dirpath.is_file():
        args.model_dirpath = args.model_dirpath.parent

    model = Model(args.model_dirpath)
    outputs = model.predict(args.image_filepath)
    print_outputs(outputs)


if __name__ == '__main__':
    main()
