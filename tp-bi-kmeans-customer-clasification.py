import pandas as pd
import numpy as np
from datetime import datetime
#import psycopg2
import psycopg2.extras
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Se conecta a la base de datos PostgreSQL
conn = psycopg2.connect(
    host="ec2-44-206-204-65.compute-1.amazonaws.com",
    database="d5m8gh8m2el2op",
    user="vkckyxcefmhvsi",
    password="8e4032a71668e15dcd49ac184298ed23455225d0fe684ebcd3b8bbc62e297d6f"
)

# Ejecuta la consulta SQL para obtener las métricas RFM calculadas
query = '''
SELECT customer_id, EXTRACT(DAY FROM now() - MAX(created_date)) as recency,
       COUNT(order_id) as total_purchases, SUM(total_usd) as total_spent
FROM public.orders
WHERE customer_id != 0
GROUP BY customer_id
'''

data_rfm = pd.read_sql_query(query, conn)

# Escala las métricas RFM para que tengan la misma importancia en el algoritmo de K-Means
scaler = StandardScaler()
data_rfm_scaled = scaler.fit_transform(data_rfm[["recency", "total_purchases", "total_spent"]])

# Utiliza el algoritmo de K-Means para agrupar a los clientes en función de las métricas RFM
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=420)
data_rfm["cluster"] = kmeans.fit_predict(data_rfm_scaled)

# Imprime los centroides de cada grupo
for i, centroid in enumerate(kmeans.cluster_centers_):
    print(f"Centroide del grupo {i + 1}: {centroid}")

print("--------------------------------------------")

# Invierte la transformación de los centroides
original_centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Imprime los centroides de cada grupo en el espacio original de las variables
for i, centroid in enumerate(original_centroids):
    print(f"Centroide del grupo {i + 1}: {centroid}")

# Asigna etiquetas descriptivas a cada grupo (por ejemplo, "Group 1", "Group 2", ..., "Group 10")
data_rfm["client_value"] = data_rfm["cluster"].map({
    0: "Group 1",
    1: "Group 2",
    2: "Group 3",
    3: "Group 4",
    4: "Group 5"
})

# Definir el tamaño del lote y la cantidad total de registros
batch_size = 10000
total_records = len(data_rfm)

# Actualizar la columna client_value en la tabla customers en la base de datos en lotes
with conn.cursor() as cursor:
    for batch_start in range(0, total_records, batch_size):
        # Crear una lista vacía para almacenar los valores de actualización
        update_values = []
        
        # Crear la lista de tuplas con los valores de actualización para cada lote
        for index, row in data_rfm.iloc[batch_start: batch_start + batch_size].iterrows():
            update_values.append((row["client_value"], row["customer_id"]))

        # Crear la consulta de actualización para el lote actual
        update_query = '''
        UPDATE public.customers
        SET client_value = data.client_value
        FROM (VALUES %s) AS data (client_value, customer_id)
        WHERE customers.customer_id = data.customer_id
        '''

        # Ejecutar la consulta de actualización con el lote actual
        psycopg2.extras.execute_values(cursor, update_query, update_values)

    # Confirmar los cambios en la base de datos
    conn.commit()

# Cierra la conexión a la base de datos
conn.close()