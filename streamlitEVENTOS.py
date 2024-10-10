# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 23:41:53 2024

@author: torre
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib

# Cargar el modelo
model = joblib.load('bestmodelSVM.sav')

def main():
    st.title("Implementación con LR/ SVM")
    
    # Crear columnas para la imagen y los datos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        image = Image.open('logo.png')
        st.image(image, use_column_width=True)
        st.write('### Irvin A. Torres')
        st.write('### Matrícula: 315463')
        st.write('### Machine Learning MIC')
    
    with col2:
        year = st.number_input('Año:', min_value=2005.0, max_value=2015.0)
        county = st.selectbox('Condado:', opciones=['Statewide', 'Alameda', 'Amador', 'Butte', 'Calaveras', 'Colusa', 'Contra Costa', 'Del Norte', 'El Dorado', 'Fresno', 'Glenn', 'Humboldt', 'Imperial', 'Inyo', 'Kern', 'Kings', 'Lake', 'Lassen', 'Los Angeles', 'Madera', 'Marin', 'Mariposa', 'Mendocino', 'Merced', 'Modoc', 'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange', 'Placer', 'Plumas', 'Riverside', 'Sacramento', 'San Benito', 'San Bernardino', 'San Diego', 'San Francisco', 'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara', 'Santa Clara', 'Santa Cruz', 'Shasta', 'Siskiyou', 'Solano', 'Sonoma', 'Stanislaus', 'Sutter', 'Tehama', 'Trinity', 'Tulare', 'Tuolumne', 'Ventura', 'Yolo', 'Yuba'])
    
    with col3:
        psi_description = st.selectbox('Descripción PSI:', opciones=['Retained Surgical Item or Unretrieved Device Fragment', 'Iatrogenic Pneumothorax', 'Central Venous Catheter-Related Blood Stream Infection', 'Postoperative Wound Dehiscence', 'Accidental Puncture or Laceration', 'Transfusion Reaction', 'Perioperative Hemorrhage or Hematoma']) 
    
    if st.button('Predecir'):
        input_data = pd.DataFrame({'Year': [year], 'County': [county], 'PSIDescription': [psi_description]})
        prediction = model.predict(input_data)
        st.write(f'Predicción de ObsRate: {prediction[0]}')
        

if __name__ == '__main__': 
    main()