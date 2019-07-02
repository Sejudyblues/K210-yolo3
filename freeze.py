import tensorflow.python as tf
from tensorflow.python import keras
from tensorflow import lite
import argparse
import sys
from yolo3.model import mobile_yolo_body


def main(model_path):
    yolo_model = mobile_yolo_body(keras.Input((224, 320, 3)), 3, 20)  # type: keras.Model
    yolo_model.load_weights(model_path)
    keras.models.save_model(yolo_model, 'mobile_yolo.h5')

    converter = lite.TFLiteConverter.from_keras_model_file('mobile_yolo.h5')
    tflite_model = converter.convert()
    with open('mobile_yolo.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument('--model', type=str, help='path to model model file')
    args = parser.parse_args(sys.argv[1:])
    main(args.model)
