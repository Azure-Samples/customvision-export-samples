import argparse
import pathlib
import coremltools
import PIL.Image


class Model:
    def __init__(self, model_filepath):
        self.model = coremltools.models.MLModel(str(model_filepath))
        spec = self.model.get_spec()
        assert len(spec.description.input) == 1
        input_description = spec.description.input[0]
        self.input_name = input_description.name
        self.input_shape = (input_description.type.imageType.width, input_description.type.imageType.height)

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        return self.model.predict({self.input_name: image})


def print_outputs(outputs):
    print(outputs)


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
