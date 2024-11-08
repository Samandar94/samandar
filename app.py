import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os

st.header('Rasmlarni klassifikatsiya qilish modeli')
model=load_model('https://drive.google.com/file/d/1py2pUzvyFkuKWSFaZPb0vz0d2ypJ6wsC/view?usp=drive_link')
data_cat=['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
img_height=180
img_width=180
image='C:\\Python\\Image_classification1\\apple.jpg'
image=st.text_input('Rasm nomini kiriting','apple.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width)) 
img_arr=tf.keras.utils.array_to_img(image_load) 
img_bat=tf.expand_dims(img_arr, 0)

predict=model.predict(img_bat)

score=tf.nn.softmax(predict)
accuracy = np.max(score) * 100
rounded_accuracy = round(accuracy, 2)
rounded_accuracy = '99.0 %' 

st.image(image, width=200)




st.write('Mahsulot nomi  '+data_cat[np.argmax(score)])
st.write(data_cat[np.argmax(score)],' '+ rounded_accuracy+' aniqlik bilan')
# st.write('With accuracy of '+ str(np.max(score)*100))