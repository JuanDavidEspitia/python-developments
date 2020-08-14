# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 08:50:00 2020

@author: jespitiaa

Desarrollos de practica en Python - Juan David Espitia A.

"""

# Primeros pasos - Importamos las librerias
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation

# ---- Comenzamos el analisis exploratorio de los datos  ---- #

# Leemos el archivo .csv
df = pd.read_csv('C:/Users/jespitiaa/Documents/GitHub/python-developments/input/car-data.csv')

# Imprimimos el datafrmae con los datos cargados
df

# Listamos los ultimos 5 registros
df.tail(5)

# Listamos los 5 primeros registros
df.head(5)

# Mostramos la cantidad de filas que tienen el dataframe
len(df)

# Mostramos la cantidad de filas y columnas en una tupla
df.shape

# Metodo para mostrar formulas estadisticas de las distintas columnas
# Podemos ver la media, la moda, max y min, etc
# Metodo muy util 
df.describe()

# Revisamos los tipos de datos de cada columna del Dataframe
df.dtypes

# Eliminamos algunas columnas innecesarias
df = df.drop(['Driven_Wheels', 'Number of Doors', 'Popularity'], axis=1)
df.head(5)

# Renombramos las columnas del Dataframe
df = df.rename(columns=
               {"Make": "Marca",
                "Model": "Modelo",
                "Year": "Año",
                "Engine Fuel Type": "Tipo_Combustible",
                "Engine HP": "Potencia_HP", 
                "Engine Cylinders": "Cilindros", 
                "Transmission Type": "Tipo_Transmision",
                "Market Category": "Categoria_Mercado",
                "Vehicle Size": "Tamaño_Vehiculo",
                "Vehicle Style ": "Estilo_Vehiculo",
                "highway MPG": "MPG_Carretera", 
                "city mpg": "MPG_Ciudad", 
                "MSRP": "Precio" })
df.head(5)
# Revisamos que las columas esten renombradas corectamente
df.dtypes

# Primero validamos si tiene registros duplicados
df.duplicated()

# Eliminamos registros duplicados
len(df)
df = df.drop_duplicates()
len(df)

# Verificamos la cantidad de valores nulos por columna
df.count()        # Cantidad de valores por columna
df.isnull().sum() # Cantidad de nulos por columna

# Si queremos borramos los registros que tienen nulos en alguna columna 
# con el siguiente comando 
# df = df.dropna()









