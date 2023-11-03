# importing library 
import streamlit as st
from PIL import Image
import tensorflow
from tensorflow.keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import numpy as np
from numpy import vstack

# loading the model 
cust = {'InstanceNormalization':InstanceNormalization}
model_horse2zebra = load_model('./g_model_AtoB_023740.h5', cust)
model_zebra2horse = load_model('./g_model_BtoA_023740.h5', cust)

# load images and generae image
def load_image(image):
    image = np.arraye(image)
    image = image[np.newaxis,...]
    return (image)

def generate_image(model, image):
    generated_image = model.predict(image)
    images = vstack(generated_image)
    images = (images+1)/2.0

st.title("Horse Zebra GAN Web App")
st.image(Image.open('./1.png'))

pick = st.selectbox("Please select a GAN model to use: ", ["Horse 2 Zebra", "Zebra 2 Horse"])

if pick == "Horse 2 Zebra":
    st.write("This is a GAN model for Generating Zebra images from Horses")
    st.write("Try out the GAN model with a default image of a horse or simply upload an image")

    if st.button("Try with Default Image"):
        image = load_image(Image.open("./horse.jpg"))
        st.subheader("Horse Image")
        st.image(image)

        st.subheader("Generated Zebra Image")
        st.image(generate_image(model_horse2zebra,image))
    
    st.subheader("Upload an image file of a Horse to convert to Zebra")
    upload_file = st.file_uploader("Upload JPG image file of a horse only", type=["jpg","jpeg"])


    if upload_file:
        image = load_image(Image.open(upload_file))
        st.image(generate_image(model_horse2zebra,image))

else:
    st.write("This is a GAN model for Generating Horese images from Zebra")
    st.write("Try out the GAN model with a default image of a Zebra or simply upload an image")

    if st.button("Try with Default Image"):
        image = load_image(Image.open("./horse.jpg"))
        st.subheader("Horse Image")
        st.image(image)

        st.subheader("Generated Horse Image")
        st.image(generate_image(model_zebra2horse,image))
    
    st.subheader("Upload an image file of a Horse to convert to Hrse")
    upload_file = st.file_uploader("Upload JPG image file of a Zebra only", type=["jpg","jpeg"])


    if upload_file:
        image = load_image(Image.open(upload_file))
        st.image(generate_image(model_zebra2horse,image))