import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

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


