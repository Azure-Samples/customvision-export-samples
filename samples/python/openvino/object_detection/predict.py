import argparse
import pathlib
import numpy as np
import PIL.Image
from openvino.inference_engine import IECore

PROB_THRESHOLD = 0.01  # Minimum probably to show results.


class Model:
    def __init__(self, xml_filepath, bin_filepath):
        ie = IECore()
        net = ie.read_network(str(xml_filepath), str(bin_filepath))
        assert len(net.inputs) == 1

        self.exec_net = ie.load_network(network=net, device_name='CPU')
        self.input_name = list(net.inputs.keys())[0]
        self.input_shape = net.inputs[self.input_name].shape[2:]
        self.output_names = list(net.outputs.keys())

    def predict(self, image_filepath):
        # The model requires RGB[0-1] NCHW input.
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        return self.exec_net.infer(inputs={self.input_name: input_array})


def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            print(f"Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")


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
