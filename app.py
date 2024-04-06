from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for
import pymongo
from pymongo import MongoClient
import bcrypt
app = Flask(__name__)

client =pymongo.MongoClient("mongodb://localhost:27017/")
db = client["crop_recommendation"]
users_collection = db["users"]

data = pd.read_csv("Crop_recommendation.csv")

X = data.drop(columns=['label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

registered_users = {}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        phone = request.form['phone']
        aadhar = request.form['aadhar']
        # Insert user data into MongoDB
        users_collection.insert_one({"username": username, "password": password, "phone": phone, "aadhar": aadhar})
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({"username": username, "password": password})
        if user:
            # User exists, redirect to index or any other page
            return redirect(url_for('index'))
        else:
            # Invalid credentials, render login page again with a message
            return render_template('login.html', message="Invalid username or password")
    return render_template('login.html')




@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = request.form.to_dict()
        new_data = [[float(features['N']), float(features['P']), float(features['K']), float(features['temperature']), float(features['humidity']), float(features['ph']), float(features['rainfall'])]]
        crop_probabilities = model.predict_proba(new_data)[0]
        top_n = 4
        top_n_indices = crop_probabilities.argsort()[-top_n:][::-1]
        top_n_crops = [model.classes_[index] for index in top_n_indices]
        return render_template('result.html', top_n_crops=top_n_crops)


if __name__ == '__main__':
    app.run(debug=True)




@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = request.form.to_dict()
        new_data = [[float(features['N']), float(features['P']), float(features['K']), float(features['temperature']), float(features['humidity']), float(features['ph']), float(features['rainfall'])]]
        crop_probabilities = model.predict_proba(new_data)[0]
        top_n = 5 
        top_n_indices = crop_probabilities.argsort()[-top_n:][::-1]
        top_n_crops = [model.classes_[index] for index in top_n_indices]
        return render_template('result.html', top_n_crops=top_n_crops)


if __name__ == '__main__':
    app.run(debug=True)
