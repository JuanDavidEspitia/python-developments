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
#%matplotlib qt
#%matplotlib inline 

#Libreria calidad de datos
import pandas_profiling 

#Librerias de modelos
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


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

# Aplicamos algunas transformaciones convirtiendo de Millas/Galon (MPG) a Kilometros/Litro (KPL)
df["MPG_Carretera"]
df["MPG_Carretera"] = df["MPG_Carretera"] * 0.425144
df["MPG_Carretera"]
# Realizamos el mismo procedimiento pero para la columna MPG_Ciudad
df["MPG_Ciudad"]
df["MPG_Ciudad"] = df["MPG_Ciudad"] * 0.425144
df["MPG_Ciudad"]

# Nuevamente renombramos las columnas a Kilometros por galon
df = df.rename(columns={
    "MPG_Carretera": "KPL_Carretera", 
    "MPG_Ciudad": "KPL_Ciudad"})
df.dtypes


# Creamos una nueva columna de consumo de combustible en carretera
# Clasificamos por Bajo, Medio y Alto con base a una clasificacion del KPL en carretera
df["KPL_Carretera"]
# Aplicamos una funcion lambda con un IF anidado para clasificar
df['Consumo_KPL_Carretera'] = df['KPL_Carretera'].apply(
    lambda x: 'Bajo' if x <= 15 else 
             'Medio 'if x > 15 and x <= 31 
                     else 'Alto')


# Agrupamos la cantidad de cada item del campo Categoria_Mercado, Marca
# Obteniendo asi promedio, sumatoria y cantidad
df.groupby(['Categoria_Mercado'])
df.groupby(['Categoria_Mercado']).count()
df.groupby(by=["Marca"]).sum()
df.groupby(by=["Marca"]).mean()
df.groupby(['Categoria_Mercado', 'Año']).count()
df2 = df.groupby(['Marca','Año']).agg(['count', 'mean']) # Agregando dos columnas
df2 = df.groupby(['Marca']).size().reset_index(name='Cantidad')
df2 = df.groupby(['Marca', 'Año']).size().reset_index(name='Cantidad')

# Guardamos la primera version del Dataframe  en formato CSV
df.to_csv (r'C:/Users/jespitiaa/Documents/GitHub/python-developments/output/dataset_car_usa.csv'
           ,index = False, header=True)

# En algunas columnas con Nulos, vamos a reemplazar el valor Null por otro por defecto
df.count()
# ejemplo: Remplazamos el valor null por 0
# df.loc[df['set_of_numbers'].isnull(), 'set_of_numbers'] = 0
# Reemplazamos los registros vacios en la columna Categoria_Mercado por 'Default'
# Forma 1
df.loc[df['Categoria_Mercado'].isnull(), 'Categoria_Mercado'] = 'Default'
# Forma 2
df["Categoria_Mercado"].fillna("Sin Categoria", inplace = True)
# Guardamos nuevamente el Dataframe en formato CSV 
df.to_csv (r'C:/Users/jespitiaa/Documents/GitHub/python-developments/output/df3_car_usa.csv'
           ,index = False, header=True)
df.head(5)

# ----------------- Analisis exploratorio de datos ------------------- #
# Detectando valores atípicos
sns.boxplot(data=df)
# Detectanmos que la Columna Precio tiene muchos datos atipicos
# Analizamos por separado cada columna
sns.boxplot(x=df['Precio'])
sns.boxplot(x=df['Potencia_HP'])
sns.boxplot(x=df['Cilindros'])

# ----------------- Analizamos por separado cada columna --------------- #

# Analizamos en detalle
sns.distplot(df['Precio'])
df['Precio'].describe()

# Analizamos en detalle
sns.distplot(df['Año'])
df['Año'].describe()

# Analizamos en detalle
sns.distplot(df['Cilindros'])
df['Cilindros'].describe()

# Histogramas por el campo Marca
df["Marca"].value_counts().plot(kind='bar', figsize=(20,5))
plt.title("Cantidad de autos por marca")
plt.ylabel('Cantidad de autos')
plt.xlabel('Marca');
plt.show()

# Histogramas por el campo Categoria_Mercado
df["Categoria_Mercado"].value_counts().plot(kind='bar', figsize=(20,5))
plt.title("Cantidad de autos por marca")
plt.ylabel('Cantidad de autos')
plt.xlabel('Categoria');
plt.show()

# Histogramas por el campo Año
df["Año"].value_counts().plot(kind='bar', figsize=(20,5))
plt.title("Cantidad de autos por Año")
plt.ylabel('Cantidad de autos')
plt.xlabel('Año');
plt.show()

# Histogramas por el campo Tipo_Combustible
df["Tipo_Combustible"].value_counts().plot(kind='bar', figsize=(20,5))
plt.title("Cantidad de autos por marca")
plt.ylabel('Cantidad de autos')
plt.xlabel('Tipo De Combustible');
plt.show()

# Analicemos los deciles y algunos percentiles para entender esos valores anómalos. Para esto, usaremos el método quantile para calcular los percentiles.
# Los cuartiles, que dividen a la distribución en cuatro partes (corresponden a los cuantiles 0,25; 0,50 y 0,75);
# Los quintiles, que dividen a la distribución en cinco partes (corresponden a los cuantiles 0,20; 0,40; 0,60 y 0,80);
# Los deciles, que dividen a la distribución en diez partes;
# Los percentiles, que dividen a la distribución en cien partes.

# En el método Quantile: 0, 0.5 y 1 equivalen a valores mínimo, mediana y máximo.
sns.boxplot(x=df['Precio'])
df['Precio'].quantile([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .97, .98, .99, 1])

# Recortemos los outliers... por ejemplo, recortando entre 200 y el cuartil del 90% (67208 como vimos en la celda anterior)
corte_inferior = 200
corte_superior = df["Precio"].quantile(0.9)

df = df[(df["Precio"] < corte_superior) & (df["Precio"] > corte_inferior)]

# -------------------  Tipos de graficas   ------------------- #

sns.distplot(df['Precio'])

# Histogramas
df["Marca"].value_counts().plot(kind='bar', figsize=(20,5))
plt.title("Cantidad de autos por marca")
plt.ylabel('Cantidad de autos')
plt.xlabel('Marca');

# Mapas de calor
correlaciones= df.corr()
sns.heatmap(correlaciones)
correlaciones

# Dispersion
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['Potencia_HP'], df['Precio'])
ax.set_xlabel('Potencia_HP')
ax.set_ylabel('Precio')
plt.show()

# Perfilamiento de datos
df.profile_report(style={'full_width':True})
profile = df.profile_report()
rejected_variables = profile.get_rejected_variables(threshold=1)
profile = df.profile_report(title='Pandas Profiling Report')
profile.to_file(output_file='C:/Users/jespitiaa/Documents/GitHub/python-developments/output/Perfilado_df.html')


