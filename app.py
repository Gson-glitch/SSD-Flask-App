# ## Importing libraries

# In[1]:


import os
import numpy as np
import io
from PIL import Image, ImageDraw
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
import base64
import tensorflow as tf
from utils.model_utils import load_image_into_numpy_array


# In[2]:

# ## Defining Constants

# In[3]:


MODEL_PATH = "models/ssd_model/saved_model"  
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


# In[4]:


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ## Function to validate uploaded data

# In[5]:


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ## Loading the trained model

# In[6]:


model = tf.saved_model.load(MODEL_PATH)


# ## Functions to handle routes

# In[7]:


@app.route('/')
def index():
    return render_template('index.html') 


# In[8]:


# Route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    detection_threshold = 0.5   
    labels = ['Bicycle', 'cat', 'dog', 'Female', 'Male']

    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            original_image = Image.open(file_path)
            original_width, original_height = original_image.size

            # Load and preprocess the image
            image_np = load_image_into_numpy_array(file_path, target_size=(224, 224))

            # Prepare the image for the model
            input_tensor = tf.convert_to_tensor([image_np], dtype=tf.uint8)

            # Run the model using the appropriate signature
            detections = model.signatures['serving_default'](input_tensor)

            # Extract detection data
            detection_boxes = detections['detection_boxes'].numpy()[0]
            detection_classes = detections['detection_classes'].numpy()[0].astype(np.int64)
            detection_scores = detections['detection_scores'].numpy()[0]

            # Draw bounding boxes and labels on the image
            image_with_boxes = Image.fromarray(image_np)
            draw = ImageDraw.Draw(image_with_boxes)
            for box, cls, score in zip(detection_boxes, detection_classes, detection_scores):
                if score > detection_threshold:
                    y_min, x_min, y_max, x_max = box
                    x_min, x_max, y_min, y_max = x_min * image_np.shape[1], x_max * image_np.shape[1], y_min * image_np.shape[0], y_max * image_np.shape[0]
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
                    draw.text((x_min, y_min), f'{labels[cls - 1]}: {score:.2f}', fill="red")

            # Convert PIL image to base64 encoded string
            buffered = io.BytesIO()
            image_with_boxes.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            img_str = img_str.decode('utf-8')

            # Clean up saved file
            os.remove(file_path)
            
            # Get the predicted class
            highest_score_index = np.argmax(detection_scores)  # Index of the highest score
            final_class = labels[detection_classes[highest_score_index] - 1] if detection_scores[highest_score_index] > detection_threshold else "None"

            # Return the image data for AJAX request
            return jsonify({
                'image_data': img_str,
                'final_class': final_class,
                'width': original_width,
                'height': original_height
            })

    return 'No image uploaded or image type not allowed', 400
        


# In[9]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




