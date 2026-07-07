from flask import Flask, redirect, render_template, url_for, request, g, make_response, send_from_directory, flash
from markupsafe import Markup
import sqlite3  # Added
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_wtf import FlaskForm
from flask_bcrypt import Bcrypt
import requests
import numpy as np
import pandas as pd
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from utils.fertilizer import fertilizer_dic
from utils.disease import disease_dic

from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, 
    get_jwt_identity, set_access_cookies, unset_jwt_cookies,
    decode_token
)

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------
# (Model loading code is unchanged)
# Loading crop recommendation model
crop_recommendation_model_path = 'models/crop2.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Try to load Keras model, otherwise fall back to PyTorch
keras_disease_model_path = 'models/plant_disease_recog_model_pwp.keras'
keras_disease_model = None

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

label_mapping = {
    'Cherry___Powdery_mildew': 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry___healthy': 'Cherry_(including_sour)___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust': 'Corn_(maize)___Common_rust_',
    'Corn___Northern_Leaf_Blight': 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn___healthy': 'Corn_(maize)___healthy',
    'Tomato___Target_Spot': 'Tomato___Target_Spo'
}

pytorch_disease_classes = ['Apple___Apple_scab',
                           'Apple___Black_rot',
                           'Apple___Cedar_apple_rust',
                           'Apple___healthy',
                           'Blueberry___healthy',
                           'Cherry_(including_sour)_Powdery_mildew',
                           'Cherry_(including_sour)_healthy',
                           'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot',
                           'Corn_(maize)Common_rust',
                           'Corn_(maize)_Northern_Leaf_Blight',
                           'Corn_(maize)_healthy',
                           'Grape___Black_rot',
                           'Grape__Esca(Black_Measles)',
                           'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
                           'Grape___healthy',
                           'Orange__Haunglongbing(Citrus_greening)',
                           'Peach___Bacterial_spot',
                           'Peach___healthy',
                           'Pepper,bell__Bacterial_spot',
                           'Pepper,bell__healthy',
                           'Potato___Early_blight',
                           'Potato___Late_blight',
                           'Potato___healthy',
                           'Raspberry___healthy',
                           'Soybean___healthy',
                           'Squash___Powdery_mildew',
                           'Strawberry___Leaf_scorch',
                           'Strawberry___healthy',
                           'Tomato___Bacterial_spot',
                           'Tomato___Early_blight',
                           'Tomato___Late_blight',
                           'Tomato___Leaf_Mold',
                           'Tomato___Septoria_leaf_spot',
                           'Tomato___Spider_mites Two-spotted_spider_mite',
                           'Tomato___Target_Spot',
                           'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                           'Tomato___Tomato_mosaic_virus',
                           'Tomato___healthy']

active_backend = None
pytorch_disease_model = None
keras_model_num_classes = None

# Try loading Keras backend
try:
    import tensorflow as tf
    import os
    if os.path.exists(keras_disease_model_path):
        keras_disease_model = tf.keras.models.load_model(keras_disease_model_path)
        active_backend = 'keras'
        # Safely extract output shape length
        if hasattr(keras_disease_model, 'output_shape'):
            keras_model_num_classes = keras_disease_model.output_shape[-1]
        elif hasattr(keras_disease_model, 'output'):
            keras_model_num_classes = keras_disease_model.output.shape[-1]
        else:
            keras_model_num_classes = 38
        print(f"Keras disease model loaded successfully as primary backend. Output classes: {keras_model_num_classes}")
except Exception as e:
    print(f"Keras loading skipped or failed: {e}")

# Load PyTorch backend as fallback if Keras is not available
if active_backend is None:
    try:
        pytorch_disease_model = ResNet9(3, len(pytorch_disease_classes))
        pytorch_disease_model.load_state_dict(torch.load(
            'models/plant_disease_model.pth', map_location=torch.device('cpu')))
        pytorch_disease_model.eval()
        active_backend = 'pytorch'
        print("PyTorch disease model loaded successfully as fallback backend.")
    except Exception as e:
        print(f"Error loading PyTorch fallback model: {e}")

# (weather_fetch and predict_image functions are unchanged)
def weather_fetch(city_name):
    # Normalize city name
    city_clean = city_name.strip().lower() if city_name else ""
    
    # 1. First, try to make the actual API request
    try:
        api_key = config.weather_api_key
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url, timeout=3)
        x = response.json()
        if response.status_code == 200 and "main" in x:
            y = x["main"]
            temperature = round((y["temp"] - 273.15), 2)
            humidity = y["humidity"]
            return [temperature, humidity]
    except Exception as e:
        print(f"Weather API request error: {e}")
        
    # 2. Fall back to realistic historical weather data for popular agricultural/urban cities
    city_fallbacks = {
        "mumbai": [27.0, 75.0],
        "delhi": [29.0, 50.0],
        "bangalore": [24.0, 60.0],
        "bengaluru": [24.0, 60.0],
        "kolkata": [28.0, 70.0],
        "chennai": [30.0, 72.0],
        "pune": [25.0, 55.0],
        "hyderabad": [27.0, 55.0],
        "jaipur": [31.0, 40.0],
        "lucknow": [28.0, 55.0],
        "patna": [27.0, 60.0],
        "ahmedabad": [30.0, 50.0],
        "bhopal": [26.0, 50.0],
        "chandigarh": [25.0, 50.0],
        "coimbatore": [26.0, 65.0],
        "indore": [26.0, 52.0],
        "nagpur": [29.0, 48.0],
        "surat": [28.0, 70.0],
        "visakhapatnam": [28.0, 75.0],
        "patiala": [26.0, 52.0],
        "ludhiana": [26.0, 52.0],
        "london": [15.0, 70.0],
        "new york": [18.0, 65.0]
    }
    
    if city_clean in city_fallbacks:
        print(f"Weather API failed. Using historical fallback weather for '{city_name}': {city_fallbacks[city_clean]}")
        return city_fallbacks[city_clean]
        
    # 3. Default fallback if city is not in the dictionary
    print(f"Weather API failed and city '{city_name}' not in fallback database. Using default averages (Temp: 25.0, Humidity: 60.0).")
    return [25.0, 60.0]

def predict_image(img):
    global active_backend, keras_disease_model, pytorch_disease_model, keras_model_num_classes
    
    if active_backend == 'keras' and keras_disease_model is not None:
        import tensorflow as tf
        image = Image.open(io.BytesIO(img))
        image = image.convert('RGB')
        
        # Converted Keras model (38 classes) requires 256x256 inputs. Original Keras model (39 classes) requires 160x160.
        target_size = (256, 256) if keras_model_num_classes == 38 else (160, 160)
        image = image.resize(target_size)
        
        feature = tf.keras.utils.img_to_array(image)
        feature = np.expand_dims(feature, axis=0) # Add batch dimension
        prediction = keras_disease_model.predict(feature)
        idx = prediction.argmax()
        
        # Resolve class list based on number of output units
        if keras_model_num_classes == 39:
            prediction_label = keras_labels[idx]
            return prediction_label, 'keras_39'
        else:
            prediction_label = pytorch_disease_classes[idx]
            return prediction_label, 'keras_38'
        
    elif active_backend == 'pytorch' and pytorch_disease_model is not None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)), # Force exactly 256x256 square to prevent matrix shape crashes
            transforms.ToTensor(),
        ])
        image = Image.open(io.BytesIO(img))
        image = image.convert('RGB') # Force RGB to prevent channel mismatch errors
        img_t = transform(image)
        img_u = torch.unsqueeze(img_t, 0)
        yb = pytorch_disease_model(img_u)
        _, preds = torch.max(yb, dim=1)
        prediction = pytorch_disease_classes[int(preds[0].item())]
        return prediction, 'pytorch'
        
    else:
        return "Model not loaded. Please ensure models are present.", 'none'

# --- App Configuration ---
app = Flask(__name__)
bcrypt = Bcrypt(app)
app.config["SECRET_KEY"] = 'thisisaverylongandsecuresecretkey12345'

# --- JWT Configuration (Unchanged) ---
app.config["JWT_SECRET_KEY"] = "thisisaverylongandsecuresecretkey12345"
app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
app.config["JWT_COOKIE_CSRF_PROTECT"] = False
jwt = JWTManager(app)

# --- Template Context Processor to check user login status safely ---
@app.context_processor
def inject_user_status():
    token = request.cookies.get('access_token_cookie')
    with open("debug.log", "a") as f:
        f.write(f"[DEBUG inject_user_status] Token found: {token is not None}\n")
    if token:
        try:
            # decode_token validates the signature and checks expiration
            decoded = decode_token(token)
            with open("debug.log", "a") as f:
                f.write(f"[DEBUG inject_user_status] Decode success for: {decoded.get('sub')}\n")
            return dict(is_logged_in=True, current_user=decoded.get('sub'))
        except Exception as e:
            with open("debug.log", "a") as f:
                f.write(f"[DEBUG inject_user_status] Decode failed: {str(e)}\n")
            pass
    return dict(is_logged_in=False, current_user=None)


# --- JWT Error Callbacks to Redirect to Signup/Login ---
@jwt.unauthorized_loader
def unauthorized_callback(err_str):
    flash("You must signup or login first to access our agricultural services.", "warning")
    return redirect(url_for('signup'))

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    flash("Your login session has expired. Please login again.", "warning")
    return redirect(url_for('login'))

@jwt.invalid_token_loader
def invalid_token_callback(err_str):
    flash("Invalid session token. Please login or signup.", "warning")
    return redirect(url_for('signup'))


# --- NEW: SQLite3 Database Setup ---
DATABASE = 'database.db'

def get_db_connection():
    """Opens a new database connection."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # This lets you access columns by name
    return conn

@app.teardown_appcontext
def close_connection(exception):
    """Closes the database connection at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """
    Initializes the database.
    Run this once manually to create your .db file and tables.
    """
    with app.app_context():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create User table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS User (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        
        # Create UserAdmin table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS UserAdmin (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        
        # Create ContactUs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ContactUs (
                sno INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                text TEXT NOT NULL,
                date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database tables created.")

# --- Removed SQLAlchemy Model Classes ---

# --- Forms (Modified validate_username) ---
class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(),Length(min=3,max=50)],render_kw={"placeholder":"username"})
    password = PasswordField(validators=[InputRequired(),Length(min=3,max=50)],render_kw={"placeholder":"password"})
    submit = SubmitField("Register")

    def validate_username(self, username):
        # Replaced SQLAlchemy query with sqlite3
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM User WHERE username = ?', (username.data,)).fetchone()
        conn.close()
        if user:
            raise ValidationError("That username already exist. please choose different one.")

class AdminRegisterForm(RegisterForm):
    def validate_username(self, username):
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM UserAdmin WHERE username = ?', (username.data,)).fetchone()
        conn.close()
        if user:
            raise ValidationError("That admin username already exist. please choose different one.")

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(),Length(min=3,max=50)],render_kw={"placeholder":"username"})
    password = PasswordField(validators=[InputRequired(),Length(min=3,max=50)],render_kw={"placeholder":"password"})
    submit = SubmitField("Login")

# --- Routes (Unprotected) ---
@app.route("/")
def hello_world():
    return render_template("index.html")
    
@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method=='POST':
        name = request.form['name']
        email = request.form['email']
        text = request.form['text']
        
        # Replaced SQLAlchemy with sqlite3
        conn = get_db_connection()
        conn.execute('INSERT INTO ContactUs (name, email, text) VALUES (?, ?, ?)',
                     (name, email, text))
        conn.commit()
        conn.close()
    
    return render_template("contact.html")

# --- Updated /login Route ---
@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    
    if request.method == 'POST':
        print(f"[DEBUG login] Form submitted. Data: {request.form}")
        print(f"[DEBUG login] Form validation result: {form.validate()}")
        print(f"[DEBUG login] Form errors: {form.errors}")
        with open("debug.log", "a") as f:
            f.write(f"LOGIN: method={request.method}, form={request.form}, errors={form.errors}, valid={form.validate()}\n")
            
    if form.validate_on_submit():
        # Replaced SQLAlchemy query with sqlite3
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM User WHERE username = ?', (form.username.data,)).fetchone()
        conn.close()
        
        if user:
            # Note: Accessing data using dictionary keys (e.g., user['password'])
            if bcrypt.check_password_hash(user['password'], form.password.data):
                access_token = create_access_token(identity=user['username'])
                response = make_response(redirect(url_for('dashboard')))
                set_access_cookies(response, access_token)
                return response
            else:
                print("[DEBUG login] Password mismatch")
                form.password.errors.append("Invalid password.")
        else:
            print("[DEBUG login] User not found")
            form.username.errors.append("Username does not exist.")
                
    return render_template("login.html", form=form)

# --- Dashboard Route (Unchanged logic) ---
@app.route('/dashboard', methods=['GET', 'POST'])
@jwt_required()
def dashboard():
    title = 'dashboard'
    return render_template('dashboard.html', title=title)

# --- Logout Route (Unchanged logic) ---
@app.route('/logout', methods=['GET', 'POST'])
@jwt_required(optional=True)
def logout():
    response = make_response(redirect(url_for('hello_world')))
    unset_jwt_cookies(response)
    flash("Logged out successfully.", "success")
    return response

# --- Updated /signup Route ---
@app.route("/signup", methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if request.method == 'POST':
        print(f"[DEBUG signup] Form submitted. Data: {request.form}")
        print(f"[DEBUG signup] Form validation result: {form.validate()}")
        print(f"[DEBUG signup] Form errors: {form.errors}")
        with open("debug.log", "a") as f:
            f.write(f"SIGNUP: method={request.method}, form={request.form}, errors={form.errors}, valid={form.validate()}\n")

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        
        # Replaced SQLAlchemy with sqlite3
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO User (username, password) VALUES (?, ?)',
                         (form.username.data, hashed_password))
            conn.commit()
            print("[DEBUG signup] User created successfully")
        except sqlite3.IntegrityError as e:
            # This is a fallback, the form validation should catch it
            print(f"[DEBUG signup] SQLite IntegrityError: {e}")
        finally:
            conn.close()
            
        return redirect(url_for('login'))

    return render_template("signup.html", form=form)

# --- Protected Routes  ---
@app.route('/crop-recommend')
@jwt_required()
def crop_recommend():
    title = 'crop-recommend - Crop Recommendation'
    return render_template('crop.html', title=title)

@app.route('/fertilizer')
@jwt_required()
def fertilizer_recommendation():
    title = '- Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
@jwt_required()
def disease_prediction():
    title = '- Disease Detection'
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction_label, backend = predict_image(img)
            
            if backend == 'none':
                prediction = Markup(f"<div class='alert alert-danger'>{prediction_label}</div>")
            elif backend == 'keras':
                mapped_label = label_mapping.get(prediction_label, prediction_label)
                if mapped_label == 'Background_without_leaves':
                    prediction = Markup("<b>Background detected</b><br/>Please upload an image of a plant leaf.")
                else:
                    disease_info = disease_dic.get(mapped_label, f"<b>Disease:</b> {mapped_label}<br/>No detailed information available.")
                    prediction = Markup(str(disease_info))
            else: # PyTorch backend
                prediction = Markup(str(disease_dic.get(prediction_label, f"<b>Disease:</b> {prediction_label}")))
                
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            print(f"Error in prediction: {e}")
            pass
    return render_template('disease.html', title=title)

# ===============================================================================================

# RENDER PREDICTION PAGES (Unchanged, no DB access)
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = '- Crop Recommendation'
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
        
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    my_prediction = crop_recommendation_model.predict(data)
    prediction_val = my_prediction[0]
    
    # Resolve prediction_val to string label if it is numeric (e.g., crop.pkl uses 1-22 indices)
    if isinstance(prediction_val, (int, np.integer, float)):
        crop_classes = [
            'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 
            'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 
            'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 
            'rice', 'watermelon'
        ]
        idx = int(prediction_val) - 1
        if 0 <= idx < len(crop_classes):
            final_prediction = crop_classes[idx]
        else:
            final_prediction = f"Unknown crop (index {prediction_val})"
    else:
        final_prediction = prediction_val
        
    return render_template('crop-result.html', prediction=final_prediction, title=title)

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    # ... (unchanged) ...
    title = '- Fertilizer Suggestion'
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    df = pd.read_csv('Data/fertilizer.csv')
    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]
    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0: key = 'NHigh'
        else: key = "Nlow"
    elif max_value == "P":
        if p < 0: key = 'PHigh'
        else: key = "Plow"
    else:
        if k < 0: key = 'KHigh'
        else: key = "Klow"
    response = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=response, title=title)

# --- Admin Routes (Updated) ---

@app.route("/display")
def querydisplay():
    # Replaced SQLAlchemy query with sqlite3
    conn = get_db_connection()
    alltodo = conn.execute('SELECT * FROM ContactUs').fetchall()
    conn.close()
    return render_template("display.html", alltodo=alltodo)

# --- Updated /AdminLogin Route ---
@app.route("/AdminLogin", methods=['GET', 'POST'])
def AdminLogin():
    form = LoginForm()
    
    if request.method == 'POST':
        print(f"[DEBUG AdminLogin] Form submitted. Data: {request.form}")
        print(f"[DEBUG AdminLogin] Form validation result: {form.validate()}")
        print(f"[DEBUG AdminLogin] Form errors: {form.errors}")
        with open("debug.log", "a") as f:
            f.write(f"ADMIN_LOGIN: method={request.method}, form={request.form}, errors={form.errors}, valid={form.validate()}\n")

    if form.validate_on_submit():
        # Replaced SQLAlchemy query with sqlite3
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM UserAdmin WHERE username = ?', (form.username.data,)).fetchone()
        conn.close()
        
        if user:
            if bcrypt.check_password_hash(user['password'], form.password.data):
                access_token = create_access_token(identity=user['username'])
                response = make_response(redirect(url_for('admindashboard')))
                set_access_cookies(response, access_token)
                return response
            else:
                print("[DEBUG AdminLogin] Password mismatch")
                form.password.errors.append("Invalid password.")
        else:
            print("[DEBUG AdminLogin] Admin user not found")
            form.username.errors.append("Admin username does not exist.")

    return render_template("adminlogin.html", form=form)

# --- Updated /admindashboard Route ---
@app.route("/admindashboard")
@jwt_required()
def admindashboard():
    # sqlite3
    conn = get_db_connection()
    alltodo = conn.execute('SELECT * FROM ContactUs').fetchall()
    alluser = conn.execute('SELECT * FROM User').fetchall()
    conn.close()
    
    return render_template("admindashboard.html", alltodo=alltodo, alluser=alluser)

@app.route("/delete_message/<int:sno>")
@jwt_required()
def delete_message(sno):
    conn = get_db_connection()
    conn.execute('DELETE FROM ContactUs WHERE sno = ?', (sno,))
    conn.commit()
    conn.close()
    return redirect(url_for('admindashboard'))

# --- Updated /reg (Admin Signup) Route ---
@app.route("/reg", methods=['GET', 'POST'])
def reg():
    form = AdminRegisterForm()

    if request.method == 'POST':
        print(f"[DEBUG reg] Form submitted. Data: {request.form}")
        print(f"[DEBUG reg] Form validation result: {form.validate()}")
        print(f"[DEBUG reg] Form errors: {form.errors}")
        with open("debug.log", "a") as f:
            f.write(f"ADMIN_REG: method={request.method}, form={request.form}, errors={form.errors}, valid={form.validate()}\n")

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        
        #  sqlite3
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO UserAdmin (username, password) VALUES (?, ?)',
                         (form.username.data, hashed_password))
            conn.commit()
            print("[DEBUG reg] Admin user created successfully")
        except sqlite3.IntegrityError as e:
            print(f"[DEBUG reg] SQLite IntegrityError: {e}")
        finally:
            conn.close()
            
        return redirect(url_for('AdminLogin'))

    return render_template("reg.html", form=form)


if __name__ == "__main__":
    # IMPORTANT: Call init_db() once if the database doesn't exist
    # You can run this manually from the terminal:
    # python -c "from app import init_db; init_db()"
    app.run(debug=True, port=8000)