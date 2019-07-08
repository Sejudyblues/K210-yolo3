alpha=1.
annotation_path=train.txt
weights_path=None
learning_rate==0.0005
classes_path=model_data/voc_classes.txt
anchors_path=model_data/tiny_yolo_anchors.txt
epochs=10
augment=True
tflite=mobile_yolo.tflite
kmodel=yolo.kmodel

train:
	python3 ./train.py \
		--alpha ${alpha} \
		--annotation_path ${annotation_path} \
		--weights_path ${weights_path} \
		--learning_rate ${learning_rate} \
		--classes_path ${classes_path} \
		--anchors_path ${anchors_path} \
		--epochs ${epochs} \
		--augment ${augment}

inference:
	python3 ./yolo_video.py --model ${weights_path}  --anchors ${anchors_path} --classes ${classes_path} --image

freeze:
	toco --output_file ${tflite} --keras_model_file ${weights_path}

convert:
	/media/zqh/ProJects/ncc-linux-x86_64/ncc -i tflite -o k210model --dataset /home/zqh/workspace/images ${tflite}  ${kmodel}

emulator:
	/media/zqh/ProJects/ncc-linux-x86_64/ncc -i k210model -o inference --dataset test_logs ${kmodel} ./output