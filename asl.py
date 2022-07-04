import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras
import cv2

st.sidebar.header("What is American Sign Language?")

st.sidebar.info('''American Sign Language (ASL) is a complete, natural language that has the same linguistic properties as spoken languages, 
with grammar that differs from English. ASL is expressed by movements of the hands and face.''')

st.sidebar.subheader("This project will recognize the alphabet shown by hand in American Sign Language.")


st.header("American Sign Language Recognizer.")
st.write("Upload image of a Hand which show American Sign Language alphabet.")

st.subheader("Upload Image Here.")

file = st.file_uploader("Upload image", type=["jpg", "png"])

model = keras.models.load_model('ASL.h5')

output = []
for i in range(65,91):
    output.append(chr(i))
output.append("del")
output.append("nothing")
output.append("space")


def predict(image):
    data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)
    size = (64, 64)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    normalized_image = image_array.astype(np.float32) / 255.0
    data[0] = normalized_image
    y = np.argmax(model.predict([data]))
    return y


if file is not None:
    img = Image.open(file)
    st.image(img)
    y = predict(img)
    st.subheader(f"Alphabet is {output[y]}")


