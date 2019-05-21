from tensorflow.python import keras
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow.python as tf
from train import create_model, get_anchors, get_classes
from tensorflow import py_function
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
import numpy as np


def test_model_graph():
    """ tensorflow.keras 中load weights不支持那个跳过不匹配的层，所以必须手动控制权重 """
    yolo = keras.models.load_model('model_data/yolo_weights.h5')  # type:keras.models.Model
    tbcback = TensorBoard()
    tbcback.set_model(yolo)

    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    h, w = input_shape
    image_input = keras.Input(shape=(h, w, 3))
    num_anchors = len(anchors)

    y_true = [keras.Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                                 num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    yolo_weight = yolo.get_weights()
    for i, w in enumerate(yolo_weight):
        if w.shape == (1, 1, 1024, 255):
            yolo_weight[i] = w[..., :(num_anchors // 3) * (num_classes + 5)]
        if w.shape == (1, 1, 512, 255):
            yolo_weight[i] = w[..., :(num_anchors // 3) * (num_classes + 5)]
        if w.shape == (1, 1, 256, 255):
            yolo_weight[i] = w[..., :(num_anchors // 3) * (num_classes + 5)]
        if w.shape == (255,):
            yolo_weight[i] = w[:(num_anchors // 3) * (num_classes + 5)]
    model_body.set_weights(yolo_weight)


def test_dict_dataset():
    """ 尝试输出字典形式的dataset """
    annotation_path = 'train.txt'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    val_split = 0.1
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(annotation_lines)
    np.random.seed(None)
    num_val = int(len(annotation_lines) * val_split)
    num_train = len(annotation_lines) - num_val

    batch_size = 32
    input_shape = (416, 416)

    num = len(annotation_lines)
    if num == 0 or batch_size <= 0:
        raise ValueError

    def parser(lines):
        image_data = []
        box_data = []
        for line in lines:
            image, box = get_random_data(line, input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        return {'input_1': image_data, 'input_2': y_true[0], 'input_3': y_true[1], 'input_4': y_true[2]}

    # x_set = (tf.data.Dataset.from_tensor_slices(annotation_lines).
    #          apply(tf.data.experimental.shuffle_and_repeat(batch_size * 300, seed=66)).
    #          batch(batch_size, drop_remainder=True).
    #          map(lambda lines: py_function(parser, [lines], ({'input_1': tf.float32, 'input_2': tf.float32, 'input_3': tf.float32, 'input_4': tf.float32}))))
    # y_set = tf.data.Dataset.from_tensors(tf.zeros(batch_size, tf.float32)).repeat()
    # dataset = tf.data.Dataset.zip((x_set, y_set))
    # dataset_iter = dataset.make_one_shot_iterator()
    # dataset_iter.get_next()


def test_parser():
    """ 测试parser函数以支持eager tensor """
    annotation_path = 'train.txt'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    val_split = 0.1
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(annotation_lines)
    np.random.seed(None)
    num_val = int(len(annotation_lines) * val_split)
    num_train = len(annotation_lines) - num_val

    batch_size = 32
    input_shape = (416, 416)

    num = len(annotation_lines)
    if num == 0 or batch_size <= 0:
        raise ValueError

    lines = tf.convert_to_tensor(annotation_lines[:10], tf.string)
    """ start parser """
    image_data = []
    box_data = []
    for line in lines:
        image, box = get_random_data(line.numpy().decode(), input_shape, random=True)
        image_data.append(image)
        box_data.append(box)

    image_data = np.array(image_data)
    box_data = np.array(box_data)

    y_true = [tf.convert_to_tensor(y, tf.float32) for y in preprocess_true_boxes(box_data, input_shape, anchors, num_classes)]
    image_data = tf.convert_to_tensor(image_data, tf.float32)
    return (image_data, *y_true)


def test_zip_dataset():
    """ 尝试zip dataset，但还是失败了 """
    annotation_path = 'train.txt'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    val_split = 0.1
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(annotation_lines)
    np.random.seed(None)
    num_val = int(len(annotation_lines) * val_split)
    num_train = len(annotation_lines) - num_val

    batch_size = 32
    input_shape = (416, 416)

    num = len(annotation_lines)
    if num == 0 or batch_size <= 0:
        raise ValueError

    def parser(lines):
        image_data = []
        box_data = []
        for line in lines:
            image, box = get_random_data(line.numpy().decode(), input_shape, random=True)
            image_data.append(image)
            box_data.append(box)

        image_data = np.array(image_data)
        box_data = np.array(box_data)

        y_true = [tf.convert_to_tensor(y, tf.float32) for y in preprocess_true_boxes(box_data, input_shape, anchors, num_classes)]
        image_data = tf.convert_to_tensor(image_data, tf.float32)
        return (image_data, *y_true)

    x_set = (tf.data.Dataset.from_tensor_slices(annotation_lines).
             apply(tf.data.experimental.shuffle_and_repeat(batch_size * 300, seed=66)).
             batch(batch_size, drop_remainder=True).
             map(lambda lines: py_function(parser, [lines], [tf.float32] * (1 + len(anchors) // 3))))
    y_set = tf.data.Dataset.from_tensors(tf.zeros(batch_size, tf.float32)).repeat()
    dataset = tf.data.Dataset.zip((x_set, y_set))

    sample = next(iter(dataset))


# NOTE 使用了Sequence但是数据加载速度还是不行
#  class YOLOSequence(Sequence):
#     def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
#         self.num = len(annotation_lines)
#         self.annotation_lines = annotation_lines
#         self.batch_size = batch_size
#         self.input_shape = input_shape
#         self.anchors = anchors
#         self.num_classes = num_classes
#         if self.num == 0 or self.batch_size <= 0:
#             raise ValueError

#     def __len__(self):
#         return self.num // self.batch_size

#     def __getitem__(self, idx):
#         image_data = []
#         box_data = []
#         for b in range(self.batch_size):
#             image, box = get_random_data(self.annotation_lines[idx * self.batch_size + b],
#                                          self.input_shape, random=True)
#             image_data.append(image)
#             box_data.append(box)
#         image_data = np.array(image_data)
#         box_data = np.array(box_data)
#         y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
#         return [image_data, *y_true], np.zeros(self.batch_size)
#     def on_epoch_end(self):
#         np.random.shuffle(self.annotation_lines)
