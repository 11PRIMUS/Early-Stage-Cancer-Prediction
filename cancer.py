import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

skin_cancer_model = tf.keras.models.load_model(r'D:\python\skin_best.keras')
breast_cancer_model = tf.keras.models.load_model(r'D:\python\breast_best.keras')
brain_tumor_model = tf.keras.models.load_model(r'D:\python\best_model.keras')

skin_cancer_classes = [
    'Keratosis',
    'Basal Cell Carcinoma',
    'Dermatofibroma',
    'Melanoma',
    'Nevus',
    'Pigmented Benign Keratosis',
    'Seborrheic Keratosis',
    'Squamous Cell Carcinoma',
    'Vascular Lesion'
]

st.title('Cancer Classification (Skin, Breast, Brain Tumor)')
st.write("Classify skin cancer, breast cancer, or brain tumors using deep learning models.")

uploaded_file = st.file_uploader("Choose an image for cancer classification...", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded scan", use_column_width=True)

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Skin Cancer Prediction
        with st.spinner("Classifying for skin cancer..."):
            skin_cancer_prediction = skin_cancer_model.predict(img_array)[0]
            skin_predicted_class = np.argmax(skin_cancer_prediction)
            skin_confidence = np.max(skin_cancer_prediction) * 100
            if skin_confidence > 50:  # Assuming > 50% confidence means cancer is detected
                st.write(f"Skin Cancer detected: **{skin_cancer_classes[skin_predicted_class]}** with confidence: {skin_confidence:.2f}%")
                st.success("Stopping further checks as cancer was detected.")
                st.stop()  # Stop further checks

            st.write(f"No Skin Cancer detected with high confidence.")

        # Breast Cancer Prediction (Binary Classification)
        with st.spinner("Classifying for breast cancer..."):
            breast_cancer_prediction = breast_cancer_model.predict(img_array)[0][0]
            breast_confidence = breast_cancer_prediction * 100
            if breast_cancer_prediction > 0.5:
                st.write(f"Breast Cancer detected with confidence: {breast_confidence:.2f}%")
                st.success("Stopping further checks as cancer was detected.")
                st.stop()  # Stop further checks

            st.write(f"No Breast Cancer detected with confidence: {100 - breast_confidence:.2f}%")

        # Brain Tumor Prediction (Binary Classification)
        with st.spinner("Classifying for brain tumor..."):
            brain_tumor_prediction = brain_tumor_model.predict(img_array)[0][0]
            brain_confidence = brain_tumor_prediction * 100
            if brain_tumor_prediction > 0.5:
                st.write(f"Brain Tumor detected with confidence: {brain_confidence:.2f}%")
            else:
                st.write(f"No Brain Tumor detected with confidence: {100 - brain_confidence:.2f}%")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
