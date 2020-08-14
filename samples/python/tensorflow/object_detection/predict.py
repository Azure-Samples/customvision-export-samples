import argparse
import pathlib
import numpy as np
import PIL.Image
import tensorflow

MAX_DETECTIONS = 64  # Max number of boxes to detect.
PROB_THRESHOLD = 0.01  # Minimum probably to show results.
IOU_THRESHOLD = 0.45


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
    ANCHORS = np.array([[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]])

    def __init__(self, model_filepath):
        self.graph_def = tensorflow.compat.v1.GraphDef()
        self.graph_def.ParseFromString(model_filepath.read_bytes())

        self.nms = NonMaxSuppression(MAX_DETECTIONS, PROB_THRESHOLD, IOU_THRESHOLD)
        input_names, self.output_names = self._get_graph_inout(self.graph_def)
        assert len(input_names) == 1 and len(self.output_names) == 1
        self.input_name = input_names[0]
        self.input_shape = self._get_input_shape(self.graph_def, self.input_name)

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array[:, :, :, (2, 1, 0)]  # => BGR

        with tensorflow.compat.v1.Session() as sess:
            tensorflow.import_graph_def(self.graph_def, name='')
            out_tensors = [sess.graph.get_tensor_by_name(o + ':0') for o in self.output_names]
            outputs = sess.run(out_tensors, {self.input_name + ':0': input_array})
        return self._postprocess(outputs[0], self.ANCHORS)

    @staticmethod
    def _get_graph_inout(graph_def):
        input_names = []
        inputs_set = set()
        outputs_set = set()

        for node in graph_def.node:
            if node.op == 'Placeholder':
                input_names.append(node.name)

            for i in node.input:
                inputs_set.add(i.split(':')[0])
            outputs_set.add(node.name)

        output_names = list(outputs_set - inputs_set)
        return input_names, output_names

    @staticmethod
    def _get_input_shape(graph_def, input_name):
        for node in graph_def.node:
            if node.name == input_name:
                return [dim.size for dim in node.attr['shape'].shape.dim][1:3]

    def _postprocess(self, outputs, anchors):
        assert len(outputs.shape) == 4 and outputs.shape[0] == 1
        outputs = outputs[0]
        assert len(anchors.shape) == 2
        num_anchors = anchors.shape[0]
        height, width, channels = outputs.shape
        assert channels % num_anchors == 0
        num_classes = channels // num_anchors - 5
        outputs = outputs.reshape((height, width, num_anchors, -1))

        x = (outputs[..., 0] + np.arange(width)[np.newaxis, :, np.newaxis]) / width
        y = (outputs[..., 1] + np.arange(height)[:, np.newaxis, np.newaxis]) / height
        w = np.exp(outputs[..., 2]) * anchors[:, 0][np.newaxis, np.newaxis, :] / width
        h = np.exp(outputs[..., 3]) * anchors[:, 1][np.newaxis, np.newaxis, :] / height

        x = x - w / 2
        y = y - h / 2
        boxes = np.stack((x, y, x + w, y + h), axis=-1).reshape(-1, 4)

        objectness = self._logistic(outputs[..., 4, np.newaxis])
        class_probs = outputs[..., 5:]
        class_probs = np.exp(class_probs - np.amax(class_probs, axis=3)[..., np.newaxis])
        class_probs = (class_probs / np.sum(class_probs, axis=3)[..., np.newaxis] * objectness).reshape(-1, num_classes)

        detected_boxes, detected_classes, detected_scores = self.nms(boxes, class_probs)
        return {'detected_boxes': detected_boxes.reshape(1, -1, 4), 'detected_classes': detected_classes.reshape(1, -1), 'detected_scores': detected_scores.reshape(1, -1)}

    def _logistic(self, x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


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
