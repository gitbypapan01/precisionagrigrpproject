# Precision Agriculture Using Machine Learning 🌱

Precision Agriculture is an intelligent web-based application designed to help farmers and agricultural enthusiasts make data-driven decisions. By leveraging Machine Learning and Deep Learning models, the system offers actionable insights to improve crop yields, maintain soil health, and detect plant diseases early.

## 🚀 Features

The application provides three core ML-powered services:

1. **🌾 Crop Prediction**:
   - Recommends the most suitable crop to cultivate based on soil parameters (Nitrogen, Phosphorus, Potassium), temperature, humidity, pH, and rainfall.
   
2. **🧪 Fertilizer Recommendation**:
   - Suggests the optimal fertilizer to use based on the specific soil conditions and the crop being cultivated, helping to maximize growth and minimize chemical waste.
   
3. **🍂 Crop Disease Detection**:
   - An image-based deep learning classifier (built with Keras/TensorFlow) that allows users to upload a picture of a plant leaf. The system predicts the disease and provides suggestions on how to cure it.

### Other Key Features:
- **User Authentication**: Secure Login, Signup, and Admin dashboard.
- **Modern UI**: A responsive, beautiful interface built with Bootstrap and custom CSS.
- **Dark Mode Support**: Seamlessly toggle between light and dark themes.

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, TensorFlow/Keras, Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Database**: SQLite

## 📁 Project Structure

- `app.py`: Main Flask application handling routing and backend logic.
- `models/`: Contains the trained Machine Learning models (`.pkl` and `.h5` files).
- `templates/`: HTML templates for the frontend web pages.
- `static/`: CSS, JavaScript, and Image assets.
- `utils/`: Helper functions and scripts for data processing and model inference.
- `Data/`: Datasets used for training the models.
- `retrain.py`, `retrain_crop.py`, etc.: Scripts for retraining the models with new data.

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/precision-agriculture-ml.git
   cd precision-agriculture-ml
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the web app:**
   Open your browser and navigate to `http://localhost:5000`

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

### Contributors
- [Saikat-nkb](https://github.com/Saikat-nkb)

## 📜 License

© 2026 Copyright: Precision Agriculture Using Machine Learning. All Rights Reserved.
