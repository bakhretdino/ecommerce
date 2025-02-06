import pandas as pd
import numpy as np
from datetime import datetime
from glob import glob
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from collections import defaultdict
from multiprocessing import Pool

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
    
    # Procesar cada sesión en el chunk
    for session_id, session_data in chunk.groupby('user_session'):
        # Contar interacciones
        user_id = session_data['user_id'].iloc[0]
        resultados['interacciones'][user_id] += len(session_data)
        
        # Procesar compras
        compras = session_data[session_data['event_type'] == 'purchase']
        if not compras.empty:
            resultados['compras'][user_id] += compras['price'].sum()
            
        # Procesar vistas
        productos_vistos = session_data[session_data['event_type'] == 'view']['product_id'].unique()
        resultados['vistas'][user_id] += len(productos_vistos)
        
        # Registrar productos vistos juntos
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
    # Preparar características para clustering
    features = pd.DataFrame({
        'total_gasto': df[df['event_type'] == 'purchase'].groupby('user_id')['price'].sum(),
        'frecuencia_compra': df[df['event_type'] == 'purchase'].groupby('user_id').size(),
        'productos_vistos': df[df['event_type'] == 'view'].groupby('user_id').size()
    }).fillna(0)
    
    # Normalizar datos
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Añadir resultados al DataFrame
    features['cluster'] = clusters
    
    return features

def aplicar_apriori(df, min_support=0.01):
    """
    Aplica algoritmo Apriori para encontrar patrones de compra frecuentes
    """
    # Crear matriz de transacciones
    transacciones = df[df['event_type'] == 'purchase'].groupby('user_session')['product_id'].agg(list)
    
    # Convertir a formato one-hot
    transacciones_matrix = pd.get_dummies(pd.DataFrame(transacciones.tolist()))
    
    # Aplicar Apriori
    frequent_itemsets = apriori(transacciones_matrix, min_support=min_support, use_colnames=True)
    
    # Generar reglas de asociación
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    return rules

def analizar_ecommerce(carpeta_datos):
    """
    Función principal de análisis
    """
    # Obtener lista de archivos CSV comprimidos
    archivos_csv = glob(os.path.join(carpeta_datos, '*.csv.gz'))
    if not archivos_csv:
        raise ValueError(f"No se encontraron archivos CSV.GZ en {carpeta_datos}")
    
    resultados_totales = []
    datos_completos = pd.DataFrame()
    
    for archivo in archivos_csv:
        print(f"Procesando archivo: {archivo}")
        
        # Leer archivo comprimido en chunks
        chunks = pd.read_csv(archivo, compression='gzip', chunksize=100000)
        
        # Aplicar MapReduce
        with Pool() as pool:
            resultados_chunks = pool.map(procesar_chunk_mapreduce, chunks)
        
        # Combinar resultados
        resultados = reduce_resultados(resultados_chunks)
        resultados_totales.append(resultados)
        
        # Guardar datos para análisis posteriores
        for chunk in pd.read_csv(archivo, compression='gzip', chunksize=100000):
            datos_relevantes = chunk[['event_time', 'event_type', 'product_id', 
                                    'user_id', 'user_session', 'price']].copy()
            datos_completos = pd.concat([datos_completos, datos_relevantes])
    
    # Combinar todos los resultados
    resultados_finales = reduce_resultados(resultados_totales)
    
    # Aplicar K-means
    print("Aplicando K-means...")
    segmentos_usuarios = aplicar_kmeans(datos_completos)
    
    # Aplicar Apriori
    print("Aplicando Apriori...")
    reglas_asociacion = aplicar_apriori(datos_completos)
    
    return {
        'mapreduce': resultados_finales,
        'kmeans': segmentos_usuarios,
        'apriori': reglas_asociacion
    }


# Ejemplo de uso
if __name__ == "__main__":
    carpeta_datos = r"C:\Users\alela\Documents\UNED\datos" 
    try:
        resultados = analizar_ecommerce(carpeta_datos)
        
        # Imprimir resultados
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