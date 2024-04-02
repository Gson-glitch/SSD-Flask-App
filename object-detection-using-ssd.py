#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls /kaggle/input/images | head')


# In[ ]:


get_ipython().system('ls /kaggle/input/pascal-voc-xml | head')


# ## Installing libraries

# In[ ]:


get_ipython().system('pip install -U --pre tensorflow=="2.*"')
get_ipython().system('pip install tf_slim')


# ## Making sure pycocotools is installed

# In[ ]:


get_ipython().system('pip install pycocotools')


# ## Cloning the TensorFlow Object Detection API

# In[ ]:


import os
import pathlib
import subprocess

# Create the ssd-object-detection directory and change into it
os.makedirs("/kaggle/working/ssd-object-detection", exist_ok=True)
os.chdir("/kaggle/working/ssd-object-detection")

# Check if the current working directory is models or if models exists
path = pathlib.Path.cwd()
if "models" in path.parts:
    while "models" in path.parts:
        os.chdir('..')
        path = pathlib.Path.cwd()

if not pathlib.Path('models').exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/tensorflow/models"], check=True)
else:
    print("Git Repository already cloned. Skipping...")


# ## Compile all Protocol Buffer files in the object_detection/protos directory into Python code, allowing their structures to be used within the TensorFlow Object Detection API.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd models/research/\nprotoc object_detection/protos/*.proto --python_out=.\n')


# ## Copy the setup file for the Object Detection API, and then install it using pip.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cp /kaggle/working/ssd-object-detection/models/research/object_detection/packages/tf2/setup.py /kaggle/working/ssd-object-detection/models/research/\ncd /kaggle/working/ssd-object-detection/models/research\npip install .\n')


# ## Imports

# In[ ]:


import os
import numpy as np
import pandas as pd
import shutil
import random
import subprocess
import tarfile
import urllib.request
from io import BytesIO
from glob import glob
from collections import defaultdict
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt
from matplotlib import patches
import tensorflow as tf



# ## Define the directories

# In[ ]:


xml_read_dir = '/kaggle/input/pascal-voc-xml'  # Read-only directory
xml_write_dir = '/kaggle/working/ssd-object-detection/pascal-voc-xml'  # Writable directory
image_dir = '/kaggle/input/images'
TF_MODEL_GARDEN_DIR = "/kaggle/working/ssd-object-detection"


# ## Script to update the paths in the XML files to point to the correct locations of the images

# In[ ]:


# Create the writable XML directory if it doesn't exist
os.makedirs(xml_write_dir, exist_ok=True)

# Iterate over the XML files and update the path
for xml_file in os.listdir(xml_read_dir):
    if xml_file.endswith('.xml'):
        read_xml_path = os.path.join(xml_read_dir, xml_file)
        write_xml_path = os.path.join(xml_write_dir, xml_file)

        # Parse the XML file
        tree = ET.parse(read_xml_path)
        root = tree.getroot()

        # Extract the filename
        image_filename = root.find('filename').text

        # Construct the new path
        new_path = os.path.join(image_dir, image_filename)

        # Update the path in the XML
        for path_element in root.iter('path'):
            path_element.text = new_path

        # Save the changes to the writable directory
        tree.write(write_xml_path)

print("XML paths have been updated.")


# ## Script to draw bounding boxes around defect types on images

# In[ ]:


def draw_bounding_boxes(image_folder, xml_folder, output_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract the filename
            filename = root.find('filename').text
            image_path = os.path.join(image_folder, filename)

            try:
                # Attempt to open the image
                image = Image.open(image_path)
            except FileNotFoundError:
                print(f"Image file not found, deleting XML file: {xml_path}")
                os.remove(xml_path)  # Delete the XML file
                continue  # Skip to the next XML file

            draw = ImageDraw.Draw(image)

            # Iterate through each object in the XML and draw the bounding boxes
            for obj in root.iter('object'):
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # Draw the bounding box
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red", width=3)

            # Save the image with bounding boxes
            output_path = os.path.join(output_folder, filename)
            image.save(output_path)


# In[ ]:


imgs_bbox_output = '/kaggle/working/ssd-object-detection/imgs_bbox_output'
os.makedirs(imgs_bbox_output, exist_ok=True)
draw_bounding_boxes(image_dir, xml_write_dir, imgs_bbox_output)


# In[ ]:


from PIL import Image
import os
from IPython.display import display

# Iterate over each image file in the directory
for image_file in random.sample(os.listdir(imgs_bbox_output), 4):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(imgs_bbox_output, image_file)
        
        # Load and display the image
        with Image.open(image_path) as img:
            display(img)


# ## Creating a workspace

# In[ ]:


get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/training_demo/annotations')
get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/training_demo/exported-models/my-model')
get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/training_demo/images')
get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/training_demo/images/train')
get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/training_demo/images/test')
get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/training_demo/xml/train')
get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/training_demo/xml/test')
get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/training_demo/models')
get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/training_demo/pre-trained-models')
get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/bugfix/')


# In[ ]:


get_ipython().system('ls /kaggle/working/ssd-object-detection/workspace')


# ## Changing to the working directory

# In[ ]:


os.chdir(TF_MODEL_GARDEN_DIR)
get_ipython().system('ls')


# ## Test whether the model builder for TensorFlow 2.x is working correctly

# In[ ]:


# From within TensorFlow/models/research/
get_ipython().system('python models/research/object_detection/builders/model_builder_tf2_test.py')


# ## Copy images from the /kaggle/input to /kaggle/working

# In[ ]:


training_images_dir = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/images")
for file in glob(os.path.join(image_dir, '*')):
    shutil.copy(file, training_images_dir)
print("First 10 files in the training images directory:")
print(os.listdir(training_images_dir)[:10])


# ## Split the dataset into the training set and the testing set

# In[ ]:


get_ipython().system('ls /kaggle/input/python-utils')


# In[ ]:


get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/scripts/preprocessing')
get_ipython().system('cp /kaggle/input/python-utils/* /kaggle/working/ssd-object-detection/scripts/preprocessing')


# In[ ]:


input_dir = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/images")
split_dataset_script = os.path.join(TF_MODEL_GARDEN_DIR, "scripts", "preprocessing", "split_dataset.py")
command = ['python', split_dataset_script, '--input', input_dir, '--train_ratio', '0.8']
subprocess.run(command)


# ## Create train and test folders for the XML files

# In[ ]:


def sort_xml_files(xml_dir, train_img_dir, test_img_dir, output_dir):
    # Create output directories for train and test XML files
    train_xml_dir = os.path.join(output_dir, 'train')
    test_xml_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_xml_dir, exist_ok=True)
    os.makedirs(test_xml_dir, exist_ok=True)

    # Iterate through each XML file
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get the filename of the image referenced in the XML file
            filename = root.find('filename').text

            # Check if this image is in train or test directory and move XML file accordingly
            if os.path.exists(os.path.join(train_img_dir, filename)):
                shutil.move(xml_path, os.path.join(train_xml_dir, xml_file))
            elif os.path.exists(os.path.join(test_img_dir, filename)):
                shutil.move(xml_path, os.path.join(test_xml_dir, xml_file))
            else:
                print(f"Image for {xml_file} not found in train or test directories.")


# In[ ]:


train_img_dir = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/images/train")
test_img_dir = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/images/test")
xml_output_dir = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/xml")
os.makedirs(xml_output_dir, exist_ok=True)
sort_xml_files(xml_write_dir, train_img_dir, test_img_dir, xml_output_dir)


# ## Create a label map in the workspace/training_demo/annotations directory

# In[ ]:


annotations_dir = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/annotations")


# In[ ]:


label_map = """
item {
  id: 1
  name: 'Bicycle'
}
item {
  id: 2
  name: 'cat'
}
item {
  id: 3
  name: 'dog'
}
item {
  id: 4
  name: 'Female'
}
item {
  id: 5
  name: 'Male'
}
"""

# Write the label map to a file
with open(os.path.join(annotations_dir, 'label_map.pbtxt'), 'w') as file:
    file.write(label_map)

print("Label map created and saved.")


# ## Create TensorFlow Records
# 
# Convert *.xml to *.record
# 

# In[ ]:


# Define the paths
xml_dir_train = os.path.join(xml_output_dir, "train")
xml_dir_test = os.path.join(xml_output_dir, "test")
labels_path = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/annotations/label_map.pbtxt")
output_path_train = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/annotations/train.record")
output_path_test = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/annotations/test.record")
image_dir_train = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/images/train")
image_dir_test = os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/images/test")
preprocessing_dir = os.path.join(TF_MODEL_GARDEN_DIR, "scripts/preprocessing")
script_path = os.path.join(preprocessing_dir, 'generate_tfrecord.py')


# ### train.record

# In[ ]:


# Prepare the command for generate TFRecords on the training data
train_command = [
    'python', script_path,
    '--xml_dir', xml_dir_train,
    '--labels_path', labels_path,
    '--output_path', output_path_train,
    '--image_dir', image_dir_train
]

# Execute the command
subprocess.run(train_command, check=True)


# ### test.record

# In[ ]:


# Prepare the command for generate TFRecords on the training data
test_command = [
    'python', script_path,
    '--xml_dir', xml_dir_test,
    '--labels_path', labels_path,
    '--output_path', output_path_test,
    '--image_dir', image_dir_test
]

# Execute the command
subprocess.run(test_command, check=True)


# In[ ]:


os.listdir(os.path.join(TF_MODEL_GARDEN_DIR, "workspace/training_demo/annotations"))


# ## Downloading the pre-trained model

# In[ ]:


# Define the URL and the destination directory
model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
destination_dir = "/kaggle/working/ssd-object-detection/workspace/training_demo/pre-trained-models"

# Create the directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Define the full path for the downloaded file
download_path = os.path.join(destination_dir, "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz")

# Download the model
urllib.request.urlretrieve(model_url, download_path)

# Extract the model
with tarfile.open(download_path, "r:gz") as tar:
    tar.extractall(path=destination_dir)

# Clean up by removing the downloaded tar.gz file
os.remove(download_path)

# List the contents of the directory to verify
os.listdir(destination_dir)


# ## Create a directory for the custom model

# In[ ]:


get_ipython().system('mkdir -p /kaggle/working/ssd-object-detection/workspace/training_demo/models/ssd_resnet50_v1_fpn')


# ## Configure the Training Pipeline

# In[ ]:


get_ipython().system('ls /kaggle/working/ssd-object-detection/workspace/training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8')


# In[ ]:


get_ipython().system('cp /kaggle/working/ssd-object-detection/workspace/training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config /kaggle/working/ssd-object-detection/workspace/training_demo/models/ssd_resnet50_v1_fpn')


# In[ ]:


# %load /kaggle/working/ssd-object-detection/workspace/training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config


# In[ ]:


get_ipython().run_cell_magic('writefile', '/kaggle/working/ssd-object-detection/workspace/training_demo/models/ssd_resnet50_v1_fpn/pipeline.config', 'model {\n  ssd {\n    num_classes: 5\n    image_resizer {\n      fixed_shape_resizer {\n        height: 224\n        width: 224\n      }\n    }\n    feature_extractor {\n      type: "ssd_resnet50_v1_fpn_keras"\n      depth_multiplier: 1.0\n      min_depth: 16\n      conv_hyperparams {\n        regularizer {\n          l2_regularizer {\n            weight: 0.00039999998989515007\n          }\n        }\n        initializer {\n          truncated_normal_initializer {\n            mean: 0.0\n            stddev: 0.029999999329447746\n          }\n        }\n        activation: RELU_6\n        batch_norm {\n          decay: 0.996999979019165\n          scale: true\n          epsilon: 0.0010000000474974513\n        }\n      }\n      override_base_feature_extractor_hyperparams: true\n      fpn {\n        min_level: 3\n        max_level: 7\n      }\n    }\n    box_coder {\n      faster_rcnn_box_coder {\n        y_scale: 10.0\n        x_scale: 10.0\n        height_scale: 5.0\n        width_scale: 5.0\n      }\n    }\n    matcher {\n      argmax_matcher {\n        matched_threshold: 0.5\n        unmatched_threshold: 0.5\n        ignore_thresholds: false\n        negatives_lower_than_unmatched: true\n        force_match_for_each_row: true\n        use_matmul_gather: true\n      }\n    }\n    similarity_calculator {\n      iou_similarity {\n      }\n    }\n    box_predictor {\n      weight_shared_convolutional_box_predictor {\n        conv_hyperparams {\n          regularizer {\n            l2_regularizer {\n              weight: 0.00039999998989515007\n            }\n          }\n          initializer {\n            random_normal_initializer {\n              mean: 0.0\n              stddev: 0.009999999776482582\n            }\n          }\n          activation: RELU_6\n          batch_norm {\n            decay: 0.996999979019165\n            scale: true\n            epsilon: 0.0010000000474974513\n          }\n        }\n        depth: 256\n        num_layers_before_predictor: 4\n        kernel_size: 3\n        class_prediction_bias_init: -4.599999904632568\n      }\n    }\n    anchor_generator {\n      multiscale_anchor_generator {\n        min_level: 3\n        max_level: 7\n        anchor_scale: 4.0\n        aspect_ratios: 1.0\n        aspect_ratios: 2.0\n        aspect_ratios: 0.5\n        scales_per_octave: 2\n      }\n    }\n    post_processing {\n      batch_non_max_suppression {\n        score_threshold: 9.99999993922529e-09\n        iou_threshold: 0.6000000238418579\n        max_detections_per_class: 100\n        max_total_detections: 100\n        use_static_shapes: false\n      }\n      score_converter: SIGMOID\n    }\n    normalize_loss_by_num_matches: true\n    loss {\n      localization_loss {\n        weighted_smooth_l1 {\n        }\n      }\n      classification_loss {\n        weighted_sigmoid_focal {\n          gamma: 2.0\n          alpha: 0.25\n        }\n      }\n      classification_weight: 1.0\n      localization_weight: 1.0\n    }\n    encode_background_as_zeros: true\n    normalize_loc_loss_by_codesize: true\n    inplace_batchnorm_update: true\n    freeze_batchnorm: false\n  }\n}\ntrain_config {\n  batch_size: 4\n  data_augmentation_options {\n    random_horizontal_flip {\n    }\n  }\n  data_augmentation_options {\n    random_crop_image {\n      min_object_covered: 0.0\n      min_aspect_ratio: 0.75\n      max_aspect_ratio: 3.0\n      min_area: 0.75\n      max_area: 1.0\n      overlap_thresh: 0.0\n    }\n  }\n  sync_replicas: true\n  optimizer {\n    momentum_optimizer {\n      learning_rate {\n        cosine_decay_learning_rate {\n          learning_rate_base: 0.03999999910593033\n          total_steps: 25000\n          warmup_learning_rate: 0.013333000242710114\n          warmup_steps: 2000\n        }\n      }\n      momentum_optimizer_value: 0.8999999761581421\n    }\n    use_moving_average: false\n  }\n  fine_tune_checkpoint: "workspace/training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0"\n  num_steps: 15000\n  startup_delay_steps: 0.0\n  replicas_to_aggregate: 8\n  max_number_of_boxes: 100\n  unpad_groundtruth_tensors: false\n  fine_tune_checkpoint_type: "detection"\n  use_bfloat16: false\n  fine_tune_checkpoint_version: V2\n}\ntrain_input_reader {\n  label_map_path: "workspace/training_demo/annotations/label_map.pbtxt"\n  tf_record_input_reader {\n    input_path: "workspace/training_demo/annotations/train.record"\n  }\n}\neval_config {\n  metrics_set: "coco_detection_metrics"\n  use_moving_averages: false\n}\neval_input_reader {\n  label_map_path: "workspace/training_demo/annotations/label_map.pbtxt"\n  shuffle: false\n  num_epochs: 1\n  tf_record_input_reader {\n    input_path: "workspace/training_demo/annotations/test.record"\n  }\n}\n')


# ## Replace the python3.x/site-packages/tf_slim/data/tfexample_decoder.py script with the file at workspace/bug_fix/tfexample_decoder.py

# In[ ]:


get_ipython().system('find / -name tfexample_decoder.py 2>/dev/null')


# In[ ]:


get_ipython().system('cp /kaggle/input/python-utils/tfexample_decoder.py /kaggle/working/ssd-object-detection/workspace/bugfix')
get_ipython().system('cp /kaggle/working/ssd-object-detection/workspace/bugfix/tfexample_decoder.py /opt/conda/lib/python3.10/site-packages/tf_slim/data/tfexample_decoder.py')
get_ipython().system('diff /kaggle/working/ssd-object-detection/workspace/bugfix/tfexample_decoder.py /opt/conda/lib/python3.10/site-packages/tf_slim/data/tfexample_decoder.py')


# ## Training the model

# In[ ]:


get_ipython().system('cp models/research/object_detection/model_main_tf2.py workspace/training_demo/')


# In[ ]:


# !python workspace/training_demo/model_main_tf2.py \
# --model_dir=workspace/training_demo/models/ssd_resnet50_v1_fpn \
# --pipeline_config_path=workspace/training_demo/models/ssd_resnet50_v1_fpn/pipeline.config


# ## Compressing the custom model for downloading

# In[ ]:


# !zip -r trained-ssd-object-detection.zip /kaggle/working/ssd-object-detection/workspace/training_demo/models


# ## Monitor Training Job Progress using TensorBoard!tensorboard --logdir=workspace/training_demo/models/ssd_resnet50_v1_fpn

# In[ ]:


# !tensorboard --logdir=workspace/training_demo/models/ssd_resnet50_v1_fpn


# ## Exporting the Trained Model
# 
# Copy the models/research/object_detection/exporter_main_v2.py script and paste it straight into your training_demo folder.
# 

# In[ ]:


get_ipython().system('cp models/research/object_detection/exporter_main_v2.py workspace/training_demo/')


# In[ ]:


get_ipython().system('python models/research/object_detection/exporter_main_v2.py      --input_type=image_tensor      --pipeline_config_path=workspace/training_demo/models/ssd_resnet50_v1_fpn/pipeline.config      --trained_checkpoint_dir=workspace/training_demo/models/ssd_resnet50_v1_fpn/      --output_directory=workspace/training_demo/exported-models/my_model')


# ### Function to load an image and convert it to the format expected by the model

# In[ ]:


def load_image_into_numpy_array(path, target_size=(224, 224)):
    """
    Load an image from file and resize it to target_size.
    Args:
    - path (str): The file path of the image.
    - target_size (tuple): The target size to resize the image as (width, height).

    Returns:
    - numpy.ndarray: The image as a numpy array.
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    image = image.resize(target_size) 
    return np.array(image).astype(np.uint8)


# ### Load the saved model

# In[ ]:


get_ipython().system('ls workspace/training_demo/exported-models/my_model/saved_model')


# In[ ]:


model_path = 'workspace/training_demo/exported-models/my_model/saved_model'
model = tf.saved_model.load(model_path)


# In[ ]:


model


# ## Path to the folder containing images to test

# In[ ]:


image_folder_path = '/kaggle/working/ssd-object-detection/workspace/training_demo/images/test'
image_paths = [
    os.path.join(image_folder_path, fname)
    for fname in os.listdir(image_folder_path)
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) 
]


# ## Define a list of labels corresponding to the model's label map

# In[ ]:


labels = ['Bicycle', 'cat', 'dog', 'Female', 'Male']


# ## Make inferences

# In[ ]:


detection_threshold = 0.5


# In[ ]:


# Loop through images and process each one
image_paths = random.sample(image_paths, 1)
for image_path in image_paths:
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor([image_np], dtype=tf.uint8)

    # Run inference
    detections = model(input_tensor)

    # Extract detection data
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int64)
    detection_scores = detections['detection_scores'][0].numpy()

    # Visualize the results and save
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_np)

    # Plot detections
    for box, cls, score in zip(detection_boxes, detection_classes, detection_scores):
        if score > detection_threshold:
            y_min, x_min, y_max, x_max = box
            x_min, x_max, y_min, y_max = x_min * image_np.shape[1], x_max * image_np.shape[1], y_min * image_np.shape[0], y_max * image_np.shape[0]
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x_min, y_min - 10, f'{labels[cls-1]}: {score:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()
    plt.close(fig)


# ## Saving the trained model to be exported to local storage

# In[ ]:


get_ipython().system('zip /kaggle/working/ssd-object-detection/workspace/training_demo/exported-models/my_model/saved_model.zip /kaggle/working/ssd-object-detection/workspace/training_demo/exported-models/my_model/saved_model')

