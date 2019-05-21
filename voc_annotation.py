import xml.etree.ElementTree as ET
import tensorflow.python as tf
import argparse
import sys
from pathlib import Path

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2012', 'train'), ('2012', 'val')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(root: Path, year, image_id, list_file):
    in_file = open(f'{str(root)}/VOCdevkit/VOC{year}/Annotations/{image_id}.xml')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def main(root: Path):
    for year, image_set in sets:
        image_ids = (root / f'VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt').open().read().strip().split()
        list_file = open('%s_%s.txt' % (year, image_set), 'w')
        for image_id in image_ids:
            list_file.write(f'{str(root)}/VOCdevkit/VOC{year}/JPEGImages/{image_id}.jpg')
            convert_annotation(root, year, image_id, list_file)
            list_file.write('\n')
        list_file.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('voc_path', type=str, help='voc dataset path')

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(Path(args.voc_path))
