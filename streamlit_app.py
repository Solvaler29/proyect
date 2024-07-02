import streamlit as st
import pandas as pd
import tensorflow as tf
from PIL import Image
import numpy as np
import os
# En esta parte tenemos que ver si el modelo que usaremos va funcionar 
try:
    model = tf.keras.models.load_model('my_40_model.h5')
    st.success("El modelo esta listo!")
except Exception as e:
    st.error(f"Error al subir el model, cargar denuevo: {e}")
    model = None  # Para que no afecte lo demas del codigo se poner none.


st.title("Detección de retinopatÍas diabéticas")

objetivo = '''
✨El objetivo principal de este proyecto es analizar en profundidad cómo las imágenes del fondo del ojo proporcionadas por el usuario pueden ser utilizadas para la detección de retinopatía diabética empleando el modelo de red neuronal profunda ResNet50.✨
'''
st.write(objetivo)

def preprocess_image(image, target_size=(224, 224)):
  image = image.resize(target_size)
  image = np.array(image)  # Normalizar para que sea de 0 y 1
  image = image.astype(np.float32)  # Converir a float32, permitiendonos almacenar decimales
  image = image.reshape((1, 224, 224, 3))  # Dimensiones de la imagen
  return image
# Función para procesar imágenes de una carpeta
def process_images_from_folder(folder_path):
    imagenes_positivas = []
    imagenes_negativas = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        
        if prediction > 0.0300:
            imagenes_positivas.append((image, filename, prediction))
            
        else:
            imagenes_negativas.append((image, filename, prediction)) 
                
    if imagenes_positivas:
        st.subheader("Imágenes con retinopatía:")
        col1, col2, col3 = st.columns(3)
        for i, (image, filename, prediction) in enumerate(imagenes_positivas):
            if i % 3 == 0:
                col1.image(image)
            elif i % 3 == 1:
                col2.image(image)
            else:
                col3.image(image)
    if imagenes_negativas:
        st.subheader("Imágenes que no tienen retinopatía:")
        col1, col2, col3 = st.columns(3)  # Crea un contenedor para una fila de imágenes
        for i, (image, filename, prediction) in enumerate(imagenes_negativas):
            if i % 3 == 0:
                col1.image(image)
            elif i % 3 == 1:
                col2.image(image)
            else:
                col3.image(image)               
# Selector de carpetas
ruta_carpeta = "/workspaces/proyect/100_Imagenes"

st.subheader("Sube una imagen para saber si tiene:", divider='violet')
uploaded_file = st.file_uploader("Eligue una imagen...", type="")

# Código original para subir una sola imagen (opcional)
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen ya subida', use_column_width=True)
        if model is not None:
            preprocessed_image = preprocess_image(image)

            prediction = model.predict(preprocessed_image)
            # Interpret the prediction
    
            if prediction > 0.0300 :  # Adjust threshold based on your model's output
                st.success("Predicción: Tiene retinopatías diabéticas")
            else:
                st.warning("Predicción: Está sano")
else:
    st.info("Sube una foto o varias para clasificar.")

st.write("Clasificacion de 100 imagenes:")
process_images_from_folder(ruta_carpeta)