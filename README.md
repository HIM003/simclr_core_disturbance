# Semi-Supervised Learning for Geological Image Classification
## Official repository for the paper "No labels, no Problem: Exploring Semi-Supervised Learning for Geological Image Classification"

### Abstract
Labelled datasets within Geoscience can often be small whilst deep learning algorithms require large datasets to learn a robust relationship between the data and its label and avoid overfitting.  To overcome the paucity of data, transfer learning has been employed in classification tasks. But an alternative exists: there often is a large corpus of unlabeled data which may enhance the learning process.  To evaluate this potential for subsurface data, we compare a high-performance semi-supervised learning algorithm (SimCLRv2) with supervised transfer learning on a Convolutional Neural Network (CNN) in geological image classification. 

### Introduction

This package uses the SimCLRv2 to classify images in a semi-supervised manner.  
The code offers the following functionality:
* Allows transfer learning (both supervised and self-supervised) of network trained on ImageNet
* Allows fine tuning using task specific images during self-supervision
* Allows training network under self-supervision with task-specific images

### Usage

#### Transfer learning (both supervised and self-supervised) of network trained on ImageNet (Experiments 1, 5, 6):

1) Models trained on ImageNet available to download here: 
	https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2
	
2) Using following script, the projection head can be fine-tuned using labelled data:  

     https://github.com/HIM003/simclr_core_disturbance/blob/main/fine_tune/TF2_finetuning.py

     Specify following: number of classes, path of model, path of data (Train, Validation & Test)
	


#### Fine tuning using task specific images during self-supervision (Experiment 2):

1) Models trained on ImageNet (self-supervised) available to download here:  
	https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2

2) Using script below, freeze different ResNet blocks and train others under self-supervision:
	
 	https://github.com/HIM003/simclr_core_disturbance/blob/main/self_supervised/tf2_dist_finetune/run_finetune.py
	
 	Other than hyperparameters, items that need specifying: number of classes, path of data, checkpoint, blocks to freeze and blocks to further train.  See example below:
	
 	```python
	--mode=train_then_eval --train_mode=finetune --train_batch_size=256 --train_epochs=50 --sk_ratio=0 --learning_rate=0.01 --weight_decay=1e-4 --temperature=0.5 --dataset=imagenet2012 --image_size=224 --eval_split=validation --resnet_depth=50 --use_blur=False --color_jitter_strength=0.5 --width_multiplier=1 --model_dir=/some_path/50d_256b_100e_w1_sk0_100_10train_ft-1_lr01_e50 --use_tpu=False --num_of_classes=10 --image_dir=/some_path/images/tmp_10train/training --checkpoint=/some_path/50d_256b_100e_w1_sk0_100/ckpt-7200 --fine_tune_after_block=-1 --labels="yep"
	```
3) Using script below, the projection head can be fine-tuned using labelled data: 
	
	https://github.com/HIM003/simclr_core_disturbance/blob/main/fine_tune/TF2_finetuning.py
 
 	Specify following: number of classes, path of model, path of data (Train, Validation & Test)


#### Training network under self-supervision with task-specific images (Experiments 3, 4, 6):

1) Using script below, train model under self-supervision: 
	
 	https://github.com/HIM003/simclr_core_disturbance/blob/main/self_supervised/tf2_dist_v3/run.py	
	
 	Other than hyperparameters, items that need specifying: number of classes, path of data & model path.  See example below:
	
 	```python	
	--mode=train_then_eval --train_mode=pretrain --train_batch_size=256 --train_epochs=100 --sk_ratio=0 --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 --dataset=imagenet2012 --image_size=224 --eval_split=validation --resnet_depth=50 --use_blur=False --color_jitter_strength=0.5 --width_multiplier=1 --model_dir=/some_path/50d_256b_100e_w1_sk0_80 --use_tpu=False --num_of_classes=10 --image_dir=/some_path/images/for_training80
	```
  
2) Using script below, the projection head can be fine-tuned using labelled data: 

   	https://github.com/HIM003/simclr_core_disturbance/blob/main/fine_tune/TF2_finetuning.py
	
 	Specify following: number of classes, path of model, path of data (Train, Validation & Test)

