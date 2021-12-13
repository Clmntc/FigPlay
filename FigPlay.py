# Déploiement streamlit
import streamlit as st
# import cv2
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
# Visualisation
# import matplotlib.pyplot as plt
# import plotly.express as px
# import seaborn as sns
# # Sklearn metrics 
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
#     confusion_matrix, classification_report
# # Holdout
# from sklearn.model_selection import train_test_split

# Keras 
# from tensorflow import keras
# 
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation,Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import Adam

##########################################
##### Import du dataset test et du modèle
@st.cache(allow_output_mutation=True)
def load_data():
   X = pd.read_csv("test.csv", decimal=',')
   X /= 255
   X = X.values.reshape(X.shape[0], 28, 28, 1) # Réduction des données et reshape 
   return X

X = load_data()

@st.cache(allow_output_mutation=True)
def load_pred():
    model = load_model('Amodel.h5')
    y_pred = model.predict(X).round() # Récupérer les bonnes prédictions (sous forme One hot encoder)
    y_pred = np.argmax(y_pred,axis=1) # Mettre au bon format pour pouvoir score
    return y_pred

y_pred = load_pred()
##########################################

st.markdown("""
<style>
.title-font {
    font-size:40px !important;
    color:white;
    font-weight: bold;
    background-color: #6200EE;
    text-align: center;
}
.title2-font {
    font-size:40px !important;
    color:white;
    font-weight: bold;
    background-color: #03DAC6;
    text-align: center;
}
.small-font {
    font-size:12px !important;
    color:#018786;
    font-weight: lighter;
    text-align: center;
}
.resultat-font {
    font-size:20px !important;
    color:#6200EE;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

image = Image.open('figplay.png')

colT1,colT2, colT3 = st.columns([1,8,1])
with colT2:
    st.image(image, width=560)

#VERSION 1
##### TEST ALEATOIRE 

def version1():
    st.markdown('<p class="title-font">Tu peux pas test !</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Enfin si quand même...</p>', unsafe_allow_html=True)

    def button():
        global index
        index = np.random.choice(X.shape[0])
        st.image(X[index], width=256)
        st.write('Voilà un ...', y_pred[index], '!')

    if st.button('Fais moi une prédiction'):
        st.write(button())  
        
# VERSION 2
###################
#### TEST ALEATOIRE 

def version2():
    st.markdown('<p class="title-font">Dessine-moi un chiffre</p>', unsafe_allow_html=True)
    colTa1,colTa2 = st.columns([1,1])
    with colTa1:
        SIZE = 256
        if 'key' not in st.session_state:
            st.session_state.key = 0

        canvas_result = st_canvas(
            fill_color='#000000',
            stroke_width=20,
            stroke_color='#FFFFFF',
            background_color='#000000',
            width=SIZE,
            height=SIZE,
            # display_toolbar='reset',
            drawing_mode="freedraw",
            key=f'canvas{st.session_state.key}'
            )

        # if canvas_result.image_data is not None:
        #     img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        img = canvas_result.image_data
        
        model = load_model('Amodel.h5')

        if st.button('Predict'):
            image = Image.fromarray((img[:, :, 0]).astype(np.uint8))
            image = image.resize((28, 28))
            image = image.convert('L')
            image = (tf.keras.utils.img_to_array(image)/255)
            image = image.reshape(1,28,28,1)
            test_x = tf.convert_to_tensor(image)
            val = model.predict(test_x)
            st.write(f'result: {np.argmax(val[0])}')

    ##################
    ##### Statistiques
    with colTa2:
        st.markdown('<p class="title2-font">Est-ce que le résultat est bon ?</p>', unsafe_allow_html=True)

        # Instancier le décompte
        if 'total' not in st.session_state:
            st.session_state.total = 0
        if 'oui' not in st.session_state:
            st.session_state.oui = 0

        # Les boutons
        if st.button('oui'):
            st.session_state.oui += 1
            st.session_state.total += 1
        if st.button('non'):
            st.session_state.total += 1

        total = st.session_state.total
        oui = st.session_state.oui

        try:
            st.write(oui/total*100,'de réussite')
        except:
            pass


####################
##### SLIDER
def main():
    page = st.sidebar.selectbox("Choose a page", ["Version 1", "Version 2"])

    if page == "Version 1":
        st.write(version1())
    elif page == "Version 2":
        st.write(version2())

if __name__ == "__main__":
    main()



# streamlit run C:/Users/Utilisateur/Desktop/Arturo/Reseau_neuronal_convolutif/Streamlit/FigPlay.py --server.maxUploadSize=1028

# https://figplay.herokuapp.com/


# # créer environnement python pour l'app
# python -m venv myenv

# # activer l'environnement python
# # allez dans le dossier myenv avec 
# cd venv
# # puis 
# ./Scripts/activate
