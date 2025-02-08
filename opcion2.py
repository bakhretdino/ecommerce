import pandas as pd
import numpy as np
from datetime import datetime
from glob import glob
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from collections import defaultdict
from multiprocessing import Pool
import kagglehub
from sklearn.preprocessing import MultiLabelBinarizer

# os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
# local_dataset_path = os.path.join(os.getcwd(), 'ecommerce_dataset')
# os.makedirs(local_dataset_path, exist_ok=True)

# Download dataset
dataset_path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")

def procesar_chunk_mapreduce(chunk):
    """
    Función Map para procesar cada chunk de datos
    """
    resultados = {
        'interacciones': defaultdict(int),
        'compras': defaultdict(float),
        'vistas': defaultdict(int),
        'productos_juntos': defaultdict(int)
    }
    
    # Optimizar tipos de datos
    # chunk['user_id'] = chunk['user_id'].astype('uint32')
    # chunk['product_id'] = chunk['product_id'].astype('uint32')
    # chunk['price'] = chunk['price'].astype('float32')

    # Procesar cada sesión en el chunk
    for session_id, session_data in chunk.groupby('user_session'):
        # Contar interacciones
        user_id = session_data['user_id'].iloc[0]
        resultados['interacciones'][user_id] += len(session_data)

        # Compras
        compras = session_data[session_data['event_type'] == 'purchase']
        if not compras.empty:
            resultados['compras'][user_id] += compras['price'].sum()

        # Productos vistos
        productos_vistos = session_data[session_data['event_type'] == 'view']['product_id'].unique()
        resultados['vistas'][user_id] += len(productos_vistos)

        # Productos vistos juntos
        if len(productos_vistos) > 1:
            for i in range(len(productos_vistos)):
                for j in range(i + 1, len(productos_vistos)):
                    pair = tuple(sorted([productos_vistos[i], productos_vistos[j]]))
                    resultados['productos_juntos'][pair] += 1

    return resultados


def reduce_resultados(resultados_list):
    """
    Función Reduce para combinar resultados
    """
    resultados_combinados = {
        'interacciones': defaultdict(int),
        'compras': defaultdict(float),
        'vistas': defaultdict(int),
        'productos_juntos': defaultdict(int)
    }
    
    for resultados in resultados_list:
        for key in resultados:
            for id_, valor in resultados[key].items():
                resultados_combinados[key][id_] += valor

    return resultados_combinados


def aplicar_kmeans(df):
    """
    Aplica K-means para segmentar usuarios
    """
    df = df.fillna(0)

    # Preparar características para clustering
    features = pd.DataFrame({
        'total_gasto': df[df['event_type'] == 'purchase'].groupby('user_id')['price'].sum(),
        'frecuencia_compra': df[df['event_type'] == 'purchase'].groupby('user_id').size(),
        'productos_vistos': df[df['event_type'] == 'view'].groupby('user_id').size()
    }).fillna(0)
    
    # Normalizar datos
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features[['total_gasto', 'frecuencia_compra', 'productos_vistos']])

    # Aplicar K-means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    features['cluster'] = kmeans.fit_predict(features_scaled)

    return features


def aplicar_apriori(df, min_support=0.01):
    """
    Aplica algoritmo Apriori para encontrar patrones de compra frecuentes
    """
    df = df.fillna(0)
    # Crear matriz de transacciones
    transacciones = df[df['event_type'] == 'purchase'].groupby('user_session')['product_id'].agg(list)
    
    # Convertir a formato one-hot
    transacciones_matrix = pd.get_dummies(pd.DataFrame(transacciones.tolist()))
    
    # Convert to binary (boolean) values for Apriori
    transacciones_matrix = transacciones_matrix.astype(bool)

    # Aplicar Apriori
    frequent_itemsets = apriori(transacciones_matrix, min_support=min_support, use_colnames=True)

    # Generar reglas de asociación
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    return rules


def analizar_ecommerce(carpeta_datos):
    """
    Función principal de análisis
    """
    # Obtener lista de files CSV extraídos
    files_csv = glob(os.path.join(carpeta_datos, '*.csv'))
    if not files_csv:
        raise ValueError(f"No se encontraron files CSV en {carpeta_datos}")

    resultados_totales = []

    for file in files_csv:
        print(f"Procesando file: {file}")

        # Leer file en chunks con Pandas
        chunk_size = 100000  # Puedes ajustar este valor
        chunks = pd.read_csv(file, usecols=['event_time', 'event_type', 'product_id', 'user_id', 'user_session', 'price'], chunksize=chunk_size)

        # Aplicar MapReduce
        with Pool() as pool:
            resultados_chunks = pool.map(procesar_chunk_mapreduce, chunks)

        # Reducir resultados
        resultados = reduce_resultados(resultados_chunks)
        resultados_totales.append(resultados)

    # Reducir resultados finales
    resultados_finales = reduce_resultados(resultados_totales)

    # Aplicar K-means y Apriori
    print("Aplicando K-means...")
    df = pd.read_csv(file, usecols=['event_time', 'event_type', 'product_id', 'user_id', 'user_session', 'price'])
    segmentos_usuarios = aplicar_kmeans(df)

    print("Aplicando Apriori...")
    reglas_asociacion = aplicar_apriori(df)

    return {
        'mapreduce': resultados_finales,
        'kmeans': segmentos_usuarios,
        'apriori': reglas_asociacion
    }


def extract_chunks(input_csv, output_csv, num_chunks, chunk_size=100000):
    """
    Extrae N chunks de un archivo CSV grande y los guarda en otro CSV.
    """
    chunk_list = []
    chunk_counter = 0

    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        chunk_list.append(chunk)
        chunk_counter += 1
        if chunk_counter >= num_chunks:
            break

    if chunk_list:
        extracted_data = pd.concat(chunk_list)
        extracted_data.to_csv(output_csv, index=False)
        print(f"Extracted {num_chunks} chunks ({chunk_size * num_chunks} rows) and saved to {output_csv}")
    else:
        print("No data was extracted. Check the input file or chunk size.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis de ecommerce")
    parser.add_argument('--test', '-t', action='store_true', help="Ejecutar en modo de prueba con un extracto pequeño de datos")

    args = parser.parse_args()

    # Si se ejecuta en modo test, usar 'extracted_data.csv'
    if args.test:
        carpeta_datos = "."
    else:
        carpeta_datos = dataset_path

    try:
        resultados = analizar_ecommerce(carpeta_datos)

        print("\n=== Resultados MapReduce ===")
        print(f"Total usuarios únicos: {len(resultados['mapreduce']['interacciones'])}")
        print(f"Total pares de productos vistos juntos: {len(resultados['mapreduce']['productos_juntos'])}")

        print("\n=== Resultados K-means ===")
        print("Distribución de clusters:")
        print(resultados['kmeans']['cluster'].value_counts())

        print("\n=== Resultados Apriori ===")
        print("Top 5 reglas de asociación por confianza:")
        print(resultados['apriori'].sort_values('confidence', ascending=False).head())

    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
