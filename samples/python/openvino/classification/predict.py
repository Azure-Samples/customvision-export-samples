import argparse
import pathlib
import numpy as np
import PIL.Image
from openvino.inference_engine import IECore


class Model:
    def __init__(self, xml_filepath, bin_filepath):
        ie = IECore()
        net = ie.read_network(str(xml_filepath), str(bin_filepath))
        assert len(net.input_info) == 1

        self.exec_net = ie.load_network(network=net, device_name='CPU')
        self.input_name = list(net.input_info.keys())[0]
        self.input_shape = net.input_info[self.input_name].input_data.shape[2:]
        self.output_names = list(net.outputs.keys())

    def predict(self, image_filepath):
        # The model requires RGB[0-1] NCHW input.
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        return self.exec_net.infer(inputs={self.input_name: input_array})


def print_outputs(outputs):
    outputs = list(outputs.values())[0]
    for index, score in enumerate(outputs[0]):
        print(f"Label: {index}, score: {score:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_filepath', type=pathlib.Path)
    parser.add_argument('bin_filepath', type=pathlib.Path)
    parser.add_argument('image_filepath', type=pathlib.Path)

    args = parser.parse_args()

    model = Model(args.xml_filepath, args.bin_filepath)
    outputs = model.predict(args.image_filepath)
    print_outputs(outputs)


if __name__ == '__main__':
    main()
