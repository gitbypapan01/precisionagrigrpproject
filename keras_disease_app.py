import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from markupsafe import Markup
from utils.disease import disease_dic

# This is a standalone app to run the Keras model as requested, 
# without modifying your original app.py.

app = Flask(__name__)

# Load the new Keras model
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

# The 39 labels defined in your Keras system
keras_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew',
    'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Mapping to reconcile the differences between the Keras labels and your existing disease_dic keys
label_mapping = {
    'Cherry___Powdery_mildew': 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry___healthy': 'Cherry_(including_sour)___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust': 'Corn_(maize)___Common_rust_',
    'Corn___Northern_Leaf_Blight': 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn___healthy': 'Corn_(maize)___healthy',
    'Tomato___Target_Spot': 'Tomato___Target_Spo'
}

# Ensure upload directory exists
UPLOAD_FOLDER = './uploadimages'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.expand_dims(feature, axis=0) # Add batch dimension required by tf models
    return feature

def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model.predict(img)
    prediction_label = keras_labels[prediction.argmax()]
    return prediction_label

@app.route('/', methods=['GET'])
def home():
    # Utilizing your existing template
    return render_template('disease.html', title='- Disease Detection')

@app.route('/disease-predict', methods=['POST'])
def disease_prediction():
    title = '- Disease Detection'
    if request.method == "POST":
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        
        # Save image temporarily 
        temp_name = f"temp_{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, temp_name)
        file.save(filepath)
        
        # Predict using Keras model
        prediction_label = model_predict(filepath)
        
        # Map label if there's a difference, otherwise use the original prediction_label
        mapped_label = label_mapping.get(prediction_label, prediction_label)
        
        if mapped_label == 'Background_without_leaves':
            result_text = Markup("<b>Background detected</b><br/>Please upload an image of a plant leaf.")
        else:
            # Look up the disease information using your existing utils/disease.py dictionary
            disease_info = disease_dic.get(mapped_label, f"<b>Disease:</b> {mapped_label}<br/>No detailed information available.")
            result_text = Markup(str(disease_info))
            
        return render_template(
            'disease-result.html', 
            prediction=result_text, 
            title=title, 
            imagepath=f'/uploadimages/{temp_name}' # Optional, if you wish to render it on the result page
        )

if __name__ == "__main__":
    app.run(debug=True, port=8001) # Running on 8001 so it doesn't conflict with your app.py on 8000
