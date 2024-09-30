import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model(r'D:\python\best_model.keras')
st.title('Brain Tumor Classification')

model_accuracy = 0.956 
st.write(f"Model Accuracy: {model_accuracy * 100:.2f}%")

# Upload MRI scan
uploaded_file = st.file_uploader("Choose an MRI scan...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI scan", use_column_width=True)
    
    img = image.resize((224, 224))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    
    prediction = model.predict(img_array)[0][0]
    
    # Display prediction confidence
    st.write(f"Prediction Confidence: {prediction * 100:.2f}%")
    
    # threshold
    if prediction > 0.5:
        st.write("Result: **Tumor detected**")
    else:
        st.write("Result: **No tumor detected**")
