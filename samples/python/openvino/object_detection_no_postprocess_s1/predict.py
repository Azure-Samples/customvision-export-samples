import argparse
import pathlib
import numpy as np
import PIL.Image
from openvino.inference_engine import IECore

MAX_DETECTIONS = 64  # Max number of boxes to detect.
PROB_THRESHOLD = 0.01  # Minimum probably to show results.
IOU_THRESHOLD = 0.50


class NonMaxSuppression:
    def __init__(self, max_detections, prob_threshold, iou_threshold):
        self.max_detections = max_detections
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, boxes, class_probs):
        """
        Args:
            boxes (np.array with shape [-1, 4]): bounding boxes. [x, y, x2, y2]
            class_probs: (np.array with shape[-1, num_classes]): probabilities for each boxes and classes.
        """
        assert len(boxes.shape) == 2 and boxes.shape[1] == 4
        assert len(class_probs.shape) == 2
        assert len(boxes) == len(class_probs)
        classes = np.argmax(class_probs, axis=1)
        probs = class_probs[np.arange(len(class_probs)), classes]
        valid_indices = probs > self.prob_threshold
        boxes, classes, probs = boxes[valid_indices, :], classes[valid_indices], probs[valid_indices]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        selected_boxes = []
        selected_classes = []
        selected_probs = []
        max_detections = min(self.max_detections, len(boxes))

        while len(selected_boxes) < max_detections:
            i = np.argmax(probs)
            if probs[i] < self.prob_threshold:
                break

            # Save the selected prediction
            selected_boxes.append(boxes[i])
            selected_classes.append(classes[i])
            selected_probs.append(probs[i])

            box = boxes[i]
            other_indices = np.concatenate((np.arange(i), np.arange(i + 1, len(boxes))))
            other_boxes = boxes[other_indices]

            # Get overlap between the 'box' and 'other_boxes'
            xy = np.maximum(box[0:2], other_boxes[:, 0:2])
            xy2 = np.minimum(box[2:4], other_boxes[:, 2:4])
            wh = np.maximum(0, xy2 - xy)

            # Calculate Intersection Over Union (IOU)
            overlap_area = wh[:, 0] * wh[:, 1]
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)

            # Find the overlapping predictions
            overlapping_indices = other_indices[np.where(iou > self.iou_threshold)[0]]
            overlapping_indices = np.append(overlapping_indices, i)

            probs[overlapping_indices] = 0

        return np.array(selected_boxes), np.array(selected_classes), np.array(selected_probs)


class Model:
    OUTPUT_SIZE = 13  # Output Height/Width.

    def __init__(self, xml_filepath, bin_filepath):
        ie = IECore()
        net = ie.read_network(str(xml_filepath), str(bin_filepath))
        assert len(net.input_info) == 1 and len(net.outputs) == 2
        self.nms = NonMaxSuppression(MAX_DETECTIONS, PROB_THRESHOLD, IOU_THRESHOLD)
        self.exec_net = ie.load_network(network=net, device_name='CPU')
        self.input_name = list(net.input_info.keys())[0]
        self.input_shape = net.input_info[self.input_name].input_data.shape[2:]
        assert set(net.outputs.keys()) == set(['raw_probs', 'raw_boxes'])

    def predict(self, image_filepath):
        # The model requires RGB[0-1] NCHW input.
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.exec_net.infer(inputs={self.input_name: input_array})
        return self._postprocess(outputs)

    def _postprocess(self, outputs):
        raw_probs = outputs['raw_probs'][0][:, 1:]
        raw_boxes = outputs['raw_boxes'][0]
        center_xy = raw_boxes[:, :2]
        wh = raw_boxes[:, 2:]
        xy = center_xy - wh / 2
        boxes = np.concatenate((xy, xy + wh), axis=1)
        detected_boxes, detected_classes, detected_scores = self.nms(boxes, raw_probs)
        return {'detected_boxes': detected_boxes.reshape(1, -1, 4), 'detected_classes': detected_classes.reshape(1, -1), 'detected_scores': detected_scores.reshape(1, -1)}


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
