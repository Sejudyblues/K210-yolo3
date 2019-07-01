"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow.python as tf
from tensorflow.contrib.data import assert_element_shape
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model, load_model, save_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.utils import Sequence
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, mobile_yolo_body
from yolo3.utils import get_random_data
from tensorflow import py_function
from pathlib import Path
from datetime import datetime
from keras_mobilenet import MobileNet

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    h, w = input_shape
    image_input = Input(shape=(h, w, 3))
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        yolo_weight = load_model(weights_path).get_weights()
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
        print('Load weights {}.'.format(weights_path))
        # freeze_body = 2
        # if freeze_body in [1, 2]:
        #     # Freeze the darknet body or freeze all but 2 output layers.
        #     num = (20, len(model_body.layers) - 2)[freeze_body - 1]
        #     for i in range(num):
        #         model_body.layers[i].trainable = False
        #     print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    h, w = input_shape
    image_input = Input(shape=(h, w, 3))
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l],
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        yolo_weight = load_model(weights_path).get_weights()
        for i, w in enumerate(yolo_weight):
            if w.shape == (1, 1, 1024, 255):
                yolo_weight[i] = w[..., :(num_anchors // 2) * (num_classes + 5)]
            if w.shape == (1, 1, 512, 255):
                yolo_weight[i] = w[..., :(num_anchors // 2) * (num_classes + 5)]
            if w.shape == (1, 1, 256, 255):
                yolo_weight[i] = w[..., :(num_anchors // 2) * (num_classes + 5)]
            if w.shape == (255,):
                yolo_weight[i] = w[:(num_anchors // 2) * (num_classes + 5)]
        model_body.set_weights(yolo_weight)
        print('Load weights {}.'.format(weights_path))
        # freeze_body = 2
        # if freeze_body in [1, 2]:
        #     # Freeze the darknet body or freeze all but 2 output layers.
        #     num = (20, len(model_body.layers) - 2)[freeze_body - 1]
        #     for i in range(num):
        #         model_body.layers[i].trainable = False
        #     print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7, 'print_loss': True})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_mobile_yolo(input_shape, anchors, num_classes, load_pretrained=True, weights_path=None):
    '''create the training model, for mobilenetv1 YOLOv3'''
    K.clear_session()  # get a new session
    h, w = input_shape
    image_input = Input(shape=(h, w, 3))
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l],
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = mobile_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Mobilenet YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if isinstance(load_pretrained, str):
        model_body.load_weights(weights_path)
        print('Load weights {}.'.format(weights_path))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7, 'print_loss': True})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_dataset(annotation_lines: np.ndarray, batch_size: int,
                   input_shape: list, anchors: np.ndarray, num_classes: int, random=True) -> tf.data.Dataset:
    num = len(annotation_lines)
    if num == 0 or batch_size <= 0:
        raise ValueError

    def parser(lines):
        image_data = []
        box_data = []
        for line in lines:
            image, box = get_random_data(line.numpy().decode(), input_shape, random=random)
            image_data.append(image)
            box_data.append(box)

        image_data = np.array(image_data)
        box_data = np.array(box_data)

        y_true = [tf.convert_to_tensor(y, tf.float32) for y in preprocess_true_boxes(box_data, input_shape, anchors, num_classes)]
        image_data = tf.convert_to_tensor(image_data, tf.float32)
        return (image_data, *y_true)

    x_set = (tf.data.Dataset.from_tensor_slices(annotation_lines).
             apply(tf.data.experimental.shuffle_and_repeat(batch_size * 100, seed=66)).
             batch(batch_size, drop_remainder=True).
             map(lambda lines: py_function(parser, [lines], [tf.float32] * (1 + len(anchors) // 3)),
                 num_parallel_calls=tf.data.experimental.AUTOTUNE))
    y_set = tf.data.Dataset.from_tensors(tf.zeros(batch_size, tf.float32)).repeat()
    dataset = tf.data.Dataset.zip((x_set, y_set))
    return dataset


# NOTE 使用了Sequence但是数据加载速度还是不行
class YOLOSequence(Sequence):
    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        self.num = len(annotation_lines)
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        if self.num == 0 or self.batch_size <= 0:
            raise ValueError

    def __len__(self):
        return self.num // self.batch_size

    def __getitem__(self, idx):
        image_data = []
        box_data = []
        for b in range(self.batch_size):
            image, box = get_random_data(self.annotation_lines[idx * self.batch_size + b],
                                         self.input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.annotation_lines)


if __name__ == '__main__':
    annotation_path = 'train.txt'
    log_dir = Path('test_logs')
    log_dir = log_dir / datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/tiny_yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (224, 320)  # multiple of 32, hw
    batch_size = 16

    """ Set the Model """
    # model = create_tiny_model(input_shape, anchors, num_classes, weights_path='model_data/tiny_yolo_weights.h5')
    # model = create_model(input_shape, anchors, num_classes, weights_path='model_data/yolo_weights.h5')  # make sure you know what you freeze
    model = create_mobile_yolo(input_shape, anchors, num_classes)  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(str(log_dir) + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    tf.set_random_seed(10101)
    num_train = len(lines) - int(len(lines) * val_split)
    num_val = int(len(lines) * val_split)
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.

    model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    train_set = create_dataset(lines[:num_train], batch_size, input_shape, anchors, num_classes)
    vail_set = create_dataset(lines[num_train:], batch_size, input_shape, anchors, num_classes)

    shapes = (tuple([ins.shape for ins in model.input]), tuple(tf.TensorShape([batch_size, ])))

    train_set = train_set.apply(assert_element_shape(shapes))
    vail_set = vail_set.apply(assert_element_shape(shapes))

    try:
        model.fit(train_set,
                  epochs=10,
                  validation_data=vail_set, validation_steps=40,
                  steps_per_epoch=max(1, num_train // batch_size),
                  callbacks=[logging, checkpoint],
                  verbose=0)
    except KeyboardInterrupt:
        pass

    # train_set = YOLOSequence(lines[:num_train], batch_size, input_shape, anchors, num_classes)
    # model.fit_generator(train_set,
    #                     epochs=20,
    #                     steps_per_epoch=max(1, num_train // batch_size),
    #                     callbacks=[logging, checkpoint],
    #                     use_multiprocessing=True)
    # model.save_weights(log_dir + 'trained_weights_stage_1.h5')
    save_model(model, str(log_dir / 'yolo_model.h5'))
