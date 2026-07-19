import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Song Skip Predictor",
    page_icon="🎵",
    layout="wide"
)

# ===========================
# LOAD MODEL
# ===========================

model = joblib.load("song_skip_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("🎵 Song Skip Predictor")
st.write("Predict whether a user is likely to skip a song.")

st.divider()

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 10, 80, 25)

    gender = st.selectbox(
        "Gender",
        list(encoders["gender"].classes_)
    )

    premium = st.selectbox(
        "Premium User",
        list(encoders["premium_user"].classes_)
    )

    listening = st.selectbox(
        "Listening Time",
        list(encoders["listening_time"].classes_)
    )

    mood = st.selectbox(
        "Mood",
        list(encoders["mood"].classes_)
    )

    genre = st.selectbox(
        "Genre",
        list(encoders["genre"].classes_)
    )

    danceability = st.slider("Danceability",0.0,1.0,0.50)
    energy = st.slider("Energy",0.0,1.0,0.50)
    valence = st.slider("Valence",0.0,1.0,0.50)
    acousticness = st.slider("Acousticness",0.0,1.0,0.30)

with col2:

    speechiness = st.slider("Speechiness",0.0,1.0,0.20)
    instrumentalness = st.slider("Instrumentalness",0.0,1.0,0.20)
    liveness = st.slider("Liveness",0.0,1.0,0.20)

    tempo = st.slider("Tempo",50,220,120)

    loudness = st.slider("Loudness (dB)",-60,0,-10)

    duration = st.slider("Duration (sec)",60,500,180)

    popularity = st.slider("Popularity",0,100,50)

    previous_skips = st.slider("Previous Skips",0,20,2)

    repeat_count = st.slider("Repeat Count",0,20,1)

# ===========================
# PREDICT
# ===========================

if st.button("Predict"):

    gender = encoders["gender"].transform([gender])[0]
    premium = encoders["premium_user"].transform([premium])[0]
    listening = encoders["listening_time"].transform([listening])[0]
    mood = encoders["mood"].transform([mood])[0]
    genre = encoders["genre"].transform([genre])[0]

    data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "premium_user": premium,
        "listening_time": listening,
        "mood": mood,
        "genre": genre,
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "acousticness": acousticness,
        "speechiness": speechiness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "tempo": tempo,
        "loudness_db": loudness,
        "duration_sec": duration,
        "popularity": popularity,
        "previous_skips": previous_skips,
        "repeat_count": repeat_count
    }])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    st.divider()

    if prediction == 1:
        st.error("⏭️ User is Likely to Skip the Song")
        st.progress(float(probability[1]))
        st.write(f"Confidence : **{probability[1]*100:.2f}%**")
    else:
        st.success("✅ User is Likely to Listen")
        st.progress(float(probability[0]))
        st.write(f"Confidence : **{probability[0]*100:.2f}%**")

    st.subheader("Prediction Probability")

    st.write(f"Not Skip : **{probability[0]*100:.2f}%**")
    st.write(f"Skip : **{probability[1]*100:.2f}%**")