import numpy as np #loading important libraries 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
df = pd.read_csv("song_skip_dataset.csv")#loading dataset 
#  starting data inspection 
# print(df.head(25))
# print(df.shape)
# print(df.columns)
# print(df.rows)
# print(df.isnull().sum())
# print(df.info())
# print(df.describe())
# print(df.duplicated().sum())
# print(df["skip"].value_counts)
# ab ham krenge data cleaning 
# print(df.duplicated().sum())
# print(df.isnull().sum())
# print(df.info())
# sns.boxplot(x=df["age"]) # no ouliers found 
# plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ================================
# BASIC INFORMATION
# ================================

print("Shape :", df.shape)
print("\nColumns\n", df.columns)
print("\nInfo")
print(df.info())

print("\nStatistics")
print(df.describe())

# ================================
# UNIVARIATE ANALYSIS
# ================================

# Skip Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="skip", data=df)
plt.title("Skip Distribution")
plt.show()

# Gender
plt.figure(figsize=(6,4))
sns.countplot(x="gender", data=df)
plt.title("Gender Distribution")
plt.show()

# Premium User
plt.figure(figsize=(6,4))
sns.countplot(x="premium_user", data=df)
plt.title("Premium User Distribution")
plt.show()

# Listening Time
plt.figure(figsize=(7,4))
sns.countplot(x="listening_time", data=df)
plt.title("Listening Time")
plt.xticks(rotation=45)
plt.show()

# Mood
plt.figure(figsize=(8,4))
sns.countplot(y="mood", data=df)
plt.title("Mood Distribution")
plt.show()

# Genre
plt.figure(figsize=(8,5))
sns.countplot(y="genre", data=df)
plt.title("Genre Distribution")
plt.show()

# ================================
# NUMERIC DISTRIBUTION
# ================================

numeric_columns = [
    "age",
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness",
    "tempo",
    "loudness_db",
    "duration_sec",
    "popularity",
    "previous_skips",
    "repeat_count"
]

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()

# ================================
# BIVARIATE ANALYSIS
# ================================

# Gender vs Skip
plt.figure(figsize=(6,4))
sns.countplot(x="gender", hue="skip", data=df)
plt.title("Gender vs Skip")
plt.show()

# Premium vs Skip
plt.figure(figsize=(6,4))
sns.countplot(x="premium_user", hue="skip", data=df)
plt.title("Premium User vs Skip")
plt.show()

# Listening Time vs Skip
plt.figure(figsize=(7,4))
sns.countplot(x="listening_time", hue="skip", data=df)
plt.title("Listening Time vs Skip")
plt.xticks(rotation=45)
plt.show()

# Mood vs Skip
plt.figure(figsize=(10,5))
sns.countplot(x="mood", hue="skip", data=df)
plt.title("Mood vs Skip")
plt.xticks(rotation=45)
plt.show()

# Genre vs Skip
plt.figure(figsize=(10,5))
sns.countplot(x="genre", hue="skip", data=df)
plt.title("Genre vs Skip")
plt.xticks(rotation=45)
plt.show()

# ================================
# BOXPLOTS
# ================================

box_columns = [
    "age",
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness",
    "tempo",
    "loudness_db",
    "duration_sec",
    "popularity",
    "previous_skips",
    "repeat_count"
]

for col in box_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="skip", y=col, data=df)
    plt.title(f"{col} vs Skip")
    plt.show()

# ================================
# SCATTER PLOTS
# ================================

plt.figure(figsize=(7,5))
sns.scatterplot(
    x="popularity",
    y="duration_sec",
    hue="skip",
    data=df
)
plt.title("Popularity vs Duration")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(
    x="tempo",
    y="popularity",
    hue="skip",
    data=df
)
plt.title("Tempo vs Popularity")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(
    x="energy",
    y="danceability",
    hue="skip",
    data=df
)
plt.title("Energy vs Danceability")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(
    x="valence",
    y="energy",
    hue="skip",
    data=df
)
plt.title("Valence vs Energy")
plt.show()

# ================================
# CORRELATION HEATMAP
# ================================

plt.figure(figsize=(14,10))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# ================================
# PAIRPLOT
# ================================

sns.pairplot(
    df[
        [
            "danceability",
            "energy",
            "tempo",
            "popularity",
            "skip"
        ]
    ],
    hue="skip"
)

plt.show()

print("\nEDA Completed Successfully")

# ==========================================
# SONG SKIP PREDICTOR USING LOGISTIC REGRESSION
# ==========================================

# import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("song_skip_dataset.csv")

print("="*60)
print("Dataset Loaded Successfully")
print("="*60)

# ==========================================
# BASIC INFORMATION
# ==========================================

print("\nShape :", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

# ==========================================
# DATA CLEANING
# ==========================================

df.drop_duplicates(inplace=True)

# Drop user_id
df.drop("user_id", axis=1, inplace=True)

print("\nAfter Cleaning Shape :", df.shape)

# ==========================================
# LABEL ENCODING
# ==========================================

# categorical_columns = [
#     "gender",
#     "premium_user",
#     "listening_time",
#     "mood",
#     "genre"
# ]

# encoder = LabelEncoder()

# for col in categorical_columns:
#     df[col] = encoder.fit_transform(df[col])

# print("\nEncoding Completed")
categorical_columns = [
    "gender",
    "premium_user",
    "listening_time",
    "mood",
    "genre"
]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("\nEncoding Completed")

joblib.dump(encoders, "label_encoders.pkl")
print("Label Encoders Saved Successfully")

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("skip", axis=1)

y = df["skip"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ==========================================
# MODEL TRAINING
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\n" + "="*60)

print("MODEL PERFORMANCE")

print("="*60)

print("Accuracy  :", round(accuracy*100,2),"%")

print("Precision :", round(precision*100,2),"%")

print("Recall    :", round(recall*100,2),"%")

print("F1 Score  :", round(f1*100,2),"%")

print("\nConfusion Matrix")

print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "song_skip_model.pkl")

print("\nModel Saved Successfully")

print("="*60)

# ==========================================
# SAMPLE PREDICTION
# ==========================================

sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\nPrediction for First Record :", prediction[0])

# joblib.dump(encoder, "label_encoder.pkl")