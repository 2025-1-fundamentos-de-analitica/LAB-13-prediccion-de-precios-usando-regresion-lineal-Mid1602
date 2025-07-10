#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
import os
from glob import glob 

def cargar_datos():
    # Cargar datos de prueba y entrenamiento
    prueba = pd.read_csv(
        "./files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )

    entrenamiento = pd.read_csv(
        "./files/input/train_data.csv.zip",
        index_col=False,
        compression="zip",
    )

    return entrenamiento, prueba

def limpiar_datos(df):
    # Limpiar y preprocesar los datos
    df_limpio = df.copy()
    año_actual = 2021
    columnas_a_eliminar = ['Year', 'Car_Name']
    df_limpio["Edad"] = año_actual - df_limpio["Year"]
    df_limpio = df_limpio.drop(columns=columnas_a_eliminar)
    return df_limpio

def dividir_datos(df):
    # Dividir en características (X) y objetivo (Y)
    X = df.drop(columns=["Present_Price"])
    y = df["Present_Price"]
    return X, y

def crear_pipeline(X_train):
    # Crear un pipeline con procesamiento de datos y modelo
    cat_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    num_features = [col for col in X_train.columns if col not in cat_features]

    preprocesador = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_features),
            ('scaler', MinMaxScaler(), num_features),
        ],
    )

    pipeline = Pipeline(
        [
            ("preprocesador", preprocesador),
            ('selector', SelectKBest(f_regression)),
            ('modelo', LinearRegression())
        ]
    )
    return pipeline

def crear_estimador(pipeline):
    # Crear un estimador con búsqueda de hiperparámetros
    grid = {
        'selector__k': range(1, 25),
        'modelo__fit_intercept': [True, False],
        'modelo__positive': [True, False]
    }

    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    return busqueda

def _crear_directorio_salida(directorio):
    # Crear o limpiar directorio de salida
    if os.path.exists(directorio):
        for archivo in glob(f"{directorio}/*"):
            os.remove(archivo)
        os.rmdir(directorio)
    os.makedirs(directorio)

def _guardar_modelo(ruta, estimador):
    # Guardar el modelo entrenado
    _crear_directorio_salida("files/models/")

    with gzip.open(ruta, "wb") as f:
        pickle.dump(estimador, f)

def calcular_metricas(tipo, y_true, y_pred):
    # Calcular las métricas de evaluación
    return {
        "tipo": "metricas",
        "conjunto": tipo,
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mad': float(median_absolute_error(y_true, y_pred)),
    }

def _ejecutar_trabajos():
    # Función principal que ejecuta el flujo de trabajo
    train, test = cargar_datos()
    train = limpiar_datos(train)
    test = limpiar_datos(test)
    X_train, y_train = dividir_datos(train)
    X_test, y_test = dividir_datos(test)
    pipeline = crear_pipeline(X_train)

    estimador = crear_estimador(pipeline)
    estimador.fit(X_train, y_train)

    _guardar_modelo(
        os.path.join("files/models/", "modelo.pkl.gz"),
        estimador,
    )

    y_pred_test = estimador.predict(X_test)
    metricas_test = calcular_metricas("prueba", y_test, y_pred_test)
    y_pred_train = estimador.predict(X_train)
    metricas_train = calcular_metricas("entrenamiento", y_train, y_pred_train)

    os.makedirs("files/output/", exist_ok=True)

    with open("files/output/metricas.json", "w", encoding="utf-8") as archivo:
        archivo.write(json.dumps(metricas_train) + "\n")
        archivo.write(json.dumps(metricas_test) + "\n")

if __name__ == "__main__":
    _ejecutar_trabajos()
