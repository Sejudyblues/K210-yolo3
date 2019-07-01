import tensorflow.python as tf
from tensorflow.python import keras
from tensorflow import lite
import argparse
from yolo3.model import mobile_yolo_body

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument('--model', type=str, help='path to model model file')
    args = parser.parse_args()
    main(args.model)


def main(model):
    yolo_model = mobile_yolo_body(keras.Input((224, 320, 3)), 6, 20)
    yolo_model.load_weights('logs/20190701-110820/yolo_model.h5')    
    lite.TFLiteConverter.from_keras_model_file()
