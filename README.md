# songskip
# 🎵 Song Skip Predictor Using Machine Learning

## 📌 Project Overview

The **Song Skip Predictor** is a Machine Learning project that predicts whether a user is likely to skip a song based on their listening behavior, audio characteristics, and personal preferences. Instead of relying only on song features, the model also considers user-related information such as mood, listening time, premium subscription status, and previous skip history, making the prediction more realistic.

The project follows a complete end-to-end Machine Learning workflow, from data preprocessing to model deployment using Streamlit.

---

# 📂 Project Structure

```
Song Skip Predictor/
│
├── song_skip_dataset.csv        # Dataset
├── songskip01.py                # Data Cleaning + EDA + Model Training
├── song_skip_model.pkl          # Trained Logistic Regression Model
├── label_encoders.pkl           # Saved Label Encoders
├── app.py                       # Streamlit Web Application
├── requirements.txt             # Required Libraries
└── README.md                    # Project Documentation
```

---

# ⚙️ Workflow

## 1. Data Collection

* Loaded the dataset using Pandas.
* The dataset contains **4000+ records** with user behavior and song audio features.

### Features Used

* Age
* Gender
* Premium User
* Listening Time
* Mood
* Genre
* Danceability
* Energy
* Valence
* Acousticness
* Speechiness
* Instrumentalness
* Liveness
* Tempo
* Loudness
* Duration
* Popularity
* Previous Skips
* Repeat Count

**Target**

* Skip (0 = No Skip, 1 = Skip)

---

## 2. Data Inspection

Performed an initial analysis to understand the dataset.

* Shape
* Columns
* Data Types
* Missing Values
* Duplicate Records
* Statistical Summary

---

## 3. Data Cleaning

Improved the dataset quality by:

* Removing duplicate records
* Dropping unnecessary columns (`user_id`)
* Checking missing values
* Verifying data types
* Detecting outliers

This ensures the model learns from clean and reliable data.

---

## 4. Exploratory Data Analysis (EDA)

Performed visual analysis using **Matplotlib** and **Seaborn**.

### Univariate Analysis

* Gender Distribution
* Premium Users
* Mood Distribution
* Genre Distribution
* Skip Distribution
* Listening Time Distribution

### Numerical Analysis

Distribution plots for:

* Danceability
* Energy
* Tempo
* Popularity
* Loudness
* Duration
* Previous Skips

### Bivariate Analysis

Compared Skip behavior with:

* Gender
* Mood
* Genre
* Listening Time
* Premium Users

### Advanced Visualizations

* Boxplots
* Scatter Plots
* Correlation Heatmap
* Pair Plot

These analyses helped identify patterns influencing song skipping.

---

## 5. Feature Engineering

Categorical features were converted into numerical values using **Label Encoding**.

Encoded columns:

* Gender
* Premium User
* Listening Time
* Mood
* Genre

All encoders were saved for future predictions.

---

## 6. Model Building

Dataset was divided into training and testing sets using **Train-Test Split (80:20)**.

Machine Learning Algorithm used:

**Logistic Regression**

The model was trained on user behavior and audio features to classify whether a song would be skipped.

---

## 7. Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

Overall Accuracy:

**78%**

This indicates that the model can correctly predict song skip behavior in most cases.

---

## 8. Model Saving

The trained model was stored using **Joblib**.

Saved files:

* `song_skip_model.pkl`
* `label_encoders.pkl`

This avoids retraining the model every time the application starts.

---

## 9. Web Application

A responsive web application was developed using **Streamlit**.

### Features

* User-friendly interface
* Dropdowns for categorical inputs
* Sliders for numerical values
* Real-time prediction
* Prediction confidence score
* Skip / Not Skip result

The application loads the saved model and encoders to make instant predictions.

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

---

# 📊 Machine Learning Pipeline

```
Dataset
      │
      ▼
Data Inspection
      │
      ▼
Data Cleaning
      │
      ▼
EDA
      │
      ▼
Label Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Logistic Regression
      │
      ▼
Model Evaluation
      │
      ▼
Save Model
      │
      ▼
Streamlit Deployment
```

---

# ⭐ Unique Selling Points (USP)

* Predicts skip behavior using **both user preferences and audio features**, making predictions more realistic.
* Follows a **complete end-to-end Machine Learning workflow**, from data preprocessing to deployment.
* Includes comprehensive **Exploratory Data Analysis (EDA)** to understand user listening patterns.
* Saves trained models and encoders for **fast, reusable predictions** without retraining.
* Interactive **Streamlit web application** enables real-time predictions through a clean interface.
* Uses **Logistic Regression**, providing a lightweight, interpretable, and efficient baseline model.
* Modular project structure makes it easy to upgrade to advanced models like Random Forest or XGBoost in the future.
* Demonstrates practical use of data preprocessing, feature encoding, model evaluation, and deployment in one project.

---

# 📌 Conclusion

The **Song Skip Predictor** demonstrates how machine learning can be applied to understand music listening behavior. By combining user demographics, listening habits, and song audio features, the model predicts whether a song is likely to be skipped. The project showcases the complete machine learning lifecycle—from data preparation and visualization to model training, evaluation, and deployment—making it a strong portfolio project for roles in Data Science, Machine Learning, and AI.
