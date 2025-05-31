import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns

# Ruta relativa desde scrpt/ hasta imagenes/
carpeta_imgs = Path("../imagenes")
carpeta_imgs.mkdir(parents=True, exist_ok=True)

# Contador para nombres únicos
contador_graficos = {}

# Función para guardar los gráficos automáticamente
def guardar_grafico():
    titulo = plt.gca().get_title()
    if not titulo:
        titulo = "grafico"
    nombre_base = "".join(c for c in titulo if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")
    if not nombre_base:
        nombre_base = "grafico"
    if nombre_base not in contador_graficos:
        contador_graficos[nombre_base] = 1
    else:
        contador_graficos[nombre_base] += 1
    numero = contador_graficos[nombre_base]
    nombre_archivo = f"{nombre_base}_{numero:02d}.jpg"
    ruta_guardado = carpeta_imgs / nombre_archivo
    plt.savefig(ruta_guardado, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Imagen guardada: {ruta_guardado}")






# HIPÓTESIS 1 


# Leemos el archivo CSV y lo guardamos en una variable llamada "df"
movies_5000_df = pd.read_csv("../dataF/tmdb_5000_movies.csv")

# Mostramos las primeras 5 filas para ver cómo es el contenido
movies_5000_df.head()

movies_5000_df.info()

# Leemos el archivo CSV y lo guardamos en una variable llamada "df"
movies_details_df = pd.read_csv("../dataF/AllMoviesDetailsCleaned.csv", sep=";")

# Mostramos las primeras 5 filas para ver cómo es el contenido
movies_details_df.head()

movies_details_df.info()


# Filtramos las filas donde el presupuesto (budget) y la recaudación (revenue) sean mayores que 0
dataframe_sin_ceros = movies_5000_df[(movies_5000_df['budget'] > 0) & (movies_5000_df['revenue'] > 0)]

# Mostramos cuántas películas quedan después del filtrado
print("Número de películas después de filtrar:", len(dataframe_sin_ceros))


# Creamos el nuevo DataFrame solo con las columnas que nos interesan
presupuesto_recaudacion_df = dataframe_sin_ceros[['budget', 'revenue']]

# Renombramos las columnas al español
presupuesto_recaudacion_df = presupuesto_recaudacion_df.rename(columns={
    'budget': 'presupuesto',
    'revenue': 'recaudacion'
})

# Mostramos las primeras filas para comprobar los cambios
print(presupuesto_recaudacion_df.head())

# Mostramos el número de registros en el nuevo DataFrame
print("Número de películas con presupuesto y recaudación válidos:", len(presupuesto_recaudacion_df))


# Creamos el gráfico de dispersión
plt.scatter(presupuesto_recaudacion_df['presupuesto'], presupuesto_recaudacion_df['recaudacion'])

# Añadimos título y etiquetas a los ejes
plt.title("Relación entre presupuesto y recaudación")
plt.xlabel("Presupuesto (en dólares)")
plt.ylabel("Recaudación (en dólares)")

# Guardamos el gráfico

guardar_grafico()



# Creamos el gráfico de dispersión sin modificar el DataFrame original
plt.scatter(
    presupuesto_recaudacion_df['presupuesto'] / 1_000_000,  # Convertimos a millones solo para el gráfico
    presupuesto_recaudacion_df['recaudacion'] / 1_000_000   # Convertimos a millones solo para el gráfico
)

# Añadimos título y etiquetas con unidades claras
plt.title("Relación entre presupuesto y recaudación (en millones)")
plt.xlabel("Presupuesto (millones de dólares)")
plt.ylabel("Recaudación (millones de dólares)")

# Guardamos el gráfico

guardar_grafico()



# Mostramos los valores mínimo y máximo del presupuesto y la recaudación
presupuesto_min = presupuesto_recaudacion_df['presupuesto'].min()
presupuesto_max = presupuesto_recaudacion_df['presupuesto'].max()
recaudacion_min = presupuesto_recaudacion_df['recaudacion'].min()
recaudacion_max = presupuesto_recaudacion_df['recaudacion'].max()

print(f"Presupuesto mínimo: {presupuesto_min:,} $")
print(f"Presupuesto máximo: {presupuesto_max:,} $")
print(f"Recaudación mínima: {recaudacion_min:,} $")
print(f"Recaudación máxima: {recaudacion_max:,} $")


# Película con el presupuesto mínimo
pelicula_presupuesto_min = dataframe_sin_ceros[dataframe_sin_ceros['budget'] == dataframe_sin_ceros['budget'].min()]
print("🎬 Película con el presupuesto mínimo:")
print(pelicula_presupuesto_min[['title', 'budget', 'revenue']])

# Película con el presupuesto máximo
pelicula_presupuesto_max = dataframe_sin_ceros[dataframe_sin_ceros['budget'] == dataframe_sin_ceros['budget'].max()]
print("\n🎬 Película con el presupuesto máximo:")
print(pelicula_presupuesto_max[['title', 'budget', 'revenue']])

# Película con la recaudación mínima
pelicula_recaudacion_min = dataframe_sin_ceros[dataframe_sin_ceros['revenue'] == dataframe_sin_ceros['revenue'].min()]
print("\n💸 Película con la recaudación mínima:")
print(pelicula_recaudacion_min[['title', 'budget', 'revenue']])

# Película con la recaudación máxima
pelicula_recaudacion_max = dataframe_sin_ceros[dataframe_sin_ceros['revenue'] == dataframe_sin_ceros['revenue'].max()]
print("\n💸 Película con la recaudación máxima:")
print(pelicula_recaudacion_max[['title', 'budget', 'revenue']])


# Películas con presupuesto menor de 1.000
presupuestos_bajos = dataframe_sin_ceros[dataframe_sin_ceros['budget'] < 1_000]
print("Películas con presupuesto < 1.000:", len(presupuestos_bajos))

# Películas con recaudación menor de 1.000
recaudaciones_bajas = dataframe_sin_ceros[dataframe_sin_ceros['revenue'] < 1_000]
print("Películas con recaudación < 1.000:", len(recaudaciones_bajas))

# Películas que cumplen cualquiera de las dos condiciones
total_sospechosas = dataframe_sin_ceros[
    (dataframe_sin_ceros['budget'] < 1_000) | (dataframe_sin_ceros['revenue'] < 1_000)
]
print("Total películas con valores sospechosos:", len(total_sospechosas))


# Eliminamos las películas con presupuesto o recaudación sospechosamente bajos
dataframe_limpio = dataframe_sin_ceros[
    (dataframe_sin_ceros['budget'] >= 1_000) &
    (dataframe_sin_ceros['revenue'] >= 1_000)
]

# Mostramos cuántas películas quedan tras la limpieza
print("Número de películas tras eliminar valores irreales:", len(dataframe_limpio))



# Dibujamos el gráfico usando los datos ya limpios y en millones de dólares
plt.scatter(
    dataframe_limpio['budget'] / 1_000_000,
    dataframe_limpio['revenue'] / 1_000_000
)

# Añadimos título y etiquetas con unidades
plt.title("Relación entre presupuesto y recaudación (datos limpios)")
plt.xlabel("Presupuesto (millones de dólares)")
plt.ylabel("Recaudación (millones de dólares)")

# Guardamos el gráfico

guardar_grafico()




# Creamos un nuevo DataFrame temporal con valores en millones (sin modificar el original)
df_millones = dataframe_limpio.copy()
df_millones['presupuesto_millones'] = df_millones['budget'] / 1_000_000
df_millones['recaudacion_millones'] = df_millones['revenue'] / 1_000_000

# Creamos el gráfico con línea de regresión
plt.figure()
sns.regplot(
    x='presupuesto_millones',
    y='recaudacion_millones',
    data=df_millones,
    scatter_kws={'alpha': 0.4},  # puntos semitransparentes
    line_kws={'color': 'red'}    # línea roja
)

# Etiquetas y título
plt.title("Presupuesto vs Recaudación con regresión lineal")
plt.xlabel("Presupuesto (millones de dólares)")
plt.ylabel("Recaudación (millones de dólares)")
plt.tight_layout()

# Guardamos el gráfico

guardar_grafico()




# Copiamos el DataFrame limpio para no modificar el original
df_tramos = dataframe_limpio.copy()

# Creamos los tramos de presupuesto (en millones)
tramos = [0, 50_000_000, 100_000_000, 150_000_000, 200_000_000, 250_000_000, 300_000_000, 400_000_000]
nombres_tramos = ['0–50M', '50–100M', '100–150M', '150–200M', '200–250M', '250–300M', '300–400M']

# Creamos una nueva columna llamada 'grupo_presupuesto' con esos tramos
df_tramos['grupo_presupuesto'] = pd.cut(df_tramos['budget'], bins=tramos, labels=nombres_tramos)

# Agrupamos por 'grupo_presupuesto' y calculamos la media de recaudación
media_recaudacion_por_tramo = df_tramos.groupby('grupo_presupuesto')['revenue'].mean() / 1_000_000  # en millones

# Mostramos los resultados numéricos
print("Media de recaudación por grupo de presupuesto (en millones):")
print(media_recaudacion_por_tramo)

# Dibujamos el gráfico
plt.figure(figsize=(8, 5))
media_recaudacion_por_tramo.plot(kind='bar', color='skyblue')

# Añadimos etiquetas y título
plt.title("Media de recaudación por rango de presupuesto")
plt.xlabel("Rango de presupuesto")
plt.ylabel("Media de recaudación (millones de dólares)")
plt.xticks(rotation=45)
plt.tight_layout()

# Guardamos el gráfico

guardar_grafico()




# Contamos cuántas películas hay en cada grupo de presupuesto
conteo_por_grupo = df_tramos['grupo_presupuesto'].value_counts().sort_index()
print("Número de películas por grupo de presupuesto:")
print(conteo_por_grupo)


correlacion = dataframe_limpio['budget'].corr(dataframe_limpio['revenue'])
print(f"Coeficiente de correlación: {correlacion:.2f}")


# Renombramos las columnas en español para coherencia general
dataframe_limpio = dataframe_limpio.rename(columns={
    'budget': 'presupuesto',
    'revenue': 'recaudacion'
})


# Calculamos la matriz de correlación
matriz_corr = dataframe_limpio[['presupuesto', 'recaudacion']].corr()

# Dibujamos el mapa de calor
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title("Matriz de correlación (presupuesto vs. recaudación)")

guardar_grafico()




# HIPÓTESIS 2  

# Leemos el archivo usando '\t' como separador (tabulador)
premios_df = pd.read_csv('../dataF/full_data.csv', sep='\t')

premios_df


# Mostramos información básica sobre el contenido del archivo
premios_df.info()

# Cargamos el segundo dataset, el de las valoraciones en IMDb
tmdb_df= pd.read_csv('../dataF/tmdb_5000_movies.csv')
tmdb_df

# Mostramos la información básica de este dataframe
tmdb_df.info()

# Creamos un nuevo DataFrame solo con las columnas que nos interesan del de premios
premios_reducido = premios_df[['Film', 'Year', 'Winner']].copy()

# Y hacemos lo mismo con el DataFrame de TMDB
tmdb_reducido = tmdb_df[['original_title', 'release_date', 'vote_average']].copy()

# Mostramos las primeras filas de cada uno para confirmar que se han creado correctamente
print("Premios (reducido):")
print(premios_reducido.head())

print("\nTMDB (reducido):")
print(tmdb_reducido.head())


# 1. Creamos una nueva columna con los títulos en minúsculas para comparar mejor
premios_reducido['titulo_normalizado'] = premios_reducido['Film'].str.lower()
tmdb_reducido['titulo_normalizado'] = tmdb_reducido['original_title'].str.lower()

# 2. Extraemos el AÑO como número (por si lo necesitamos para emparejar años más adelante)

# En premios_reducido, cogemos los 4 primeros caracteres de la columna 'Year' y los convertimos en número
premios_reducido['anio_premio'] = premios_reducido['Year'].str[:4].astype('Int64')  # Soporta valores nulos

# En tmdb_reducido, cogemos los 4 primeros caracteres de la fecha de estreno
tmdb_reducido['anio_estreno'] = tmdb_reducido['release_date'].str[:4].astype('Int64')

# 3. Mostramos las primeras filas de cada uno para confirmar los cambios
#print("Premios reducido (con columnas nuevas):")
premios_reducido.head()

print("\nTMDB reducido (con columnas nuevas):")
tmdb_reducido.head()


print("Premios reducido (con columnas nuevas):")
premios_reducido.head()

# Hacemos el merge solo por el título normalizado
peliculas_comunes = pd.merge(
    premios_reducido,
    tmdb_reducido,
    how='inner',  # solo coincidencias
    on='titulo_normalizado',
    suffixes=('_premios', '_tmdb')
)

# Mostramos cuántas coincidencias de películas tenemos
print(f"\nNúmero total de películas coincidentes (solo por título): {len(peliculas_comunes)}")
print()
# Mostramos las primeras filas del nuevo dataframe unido
print("Películas con datos tanto de premios como de valoraciones:")
peliculas_comunes.head()


# Creamos una copia del DataFrame original para trabajar solo con lo necesario en esta hipótesis
peliculas_hipotesis_1 = peliculas_comunes.copy()

# 1. Eliminamos las películas sin valor en 'Winner'
peliculas_hipotesis_1 = peliculas_hipotesis_1.dropna(subset=['Winner'])

# 2. Convertimos 'Winner' a booleano
peliculas_hipotesis_1['Winner'] = peliculas_hipotesis_1['Winner'].astype(bool)

# 3. Eliminamos duplicados por título (nos quedamos solo con la primera aparición)
peliculas_hipotesis_1 = peliculas_hipotesis_1.drop_duplicates(subset='titulo_normalizado')

# Mostramos un resumen
print(peliculas_hipotesis_1[['Film', 'Winner', 'vote_average']].head())

# Mostramos cuántas películas finales tenemos
print(f"\nPelículas preparadas para la hipótesis 1 (sin duplicados ni nulos en 'Winner'): {len(peliculas_hipotesis_1)}")


# Separamos las películas en dos grupos según si ganaron o no
ganadoras = peliculas_hipotesis_1[peliculas_hipotesis_1['Winner'] == True]
no_ganadoras = peliculas_hipotesis_1[peliculas_hipotesis_1['Winner'] == False]

# Mostramos cuántas hay en cada grupo
print(f"Número de películas GANADORAS: {len(ganadoras)}")
print(f"Número de películas NO ganadoras: {len(no_ganadoras)}")

# Estadísticas descriptivas para cada grupo
print("\n--- Estadísticas para GANADORAS ---")
print(ganadoras['vote_average'].describe())

print("\n--- Estadísticas para NO GANADORAS ---")
print(no_ganadoras['vote_average'].describe())


# Cargamos el nuevo archivo que sí contiene ganadoras y no ganadoras
oscars_df = pd.read_csv('../dataF/the_oscar_award.csv')

# Mostramos información general para ver columnas y tipos
oscars_df.info()

# Mostramos las primeras filas para ver ejemplos reales
oscars_df.head()


# Filtramos solo las filas que tienen nombre de película
oscars_df = oscars_df.dropna(subset=['film'])

# Mostramos cuántas filas quedan después del filtrado
print(f"Número de filas con película válida: {len(oscars_df)}")

# Mostramos un par de ejemplos
print(oscars_df[['film', 'winner']].head())


# Creamos una nueva columna con el título en minúsculas
oscars_df['titulo_normalizado'] = oscars_df['film'].str.lower()

# Mostramos solo 2 columnas de títulos para confirmar que se ha hecho bien
oscars_df[['film', 'titulo_normalizado']].head()


# Creamos el DataFrame reducido con solo las columnas necesarias
oscars_reducido = oscars_df[['film', 'winner', 'titulo_normalizado']].copy()

# Mostramos las primeras filas para comprobar que está bien
oscars_reducido.head()


# Unimos los dos DataFrames "oscars_reducido con tmdb_reducido" por el título normalizado
osc_tmdb_red_unidos_df = pd.merge(
    oscars_reducido,
    tmdb_reducido,
    how='inner',
    on='titulo_normalizado'
)

# Mostramos cuántas películas se han unido
print(f"Número de películas unidas: {len(osc_tmdb_red_unidos_df)}")

# Mostramos algunas filas para comprobar el resultado
osc_tmdb_red_unidos_df[['film', 'winner', 'vote_average']].head()


# Creamos una copia limpia (Sin valores nulos ni duplicados por título) para trabajar sin
#  alterar el original
osc_tmdb_red_unidos_limpio_df = osc_tmdb_red_unidos_df.copy()

# Eliminamos filas sin nota del público (por si hay alguna)
osc_tmdb_red_unidos_limpio_df = osc_tmdb_red_unidos_limpio_df.dropna(subset=['vote_average'])

# Eliminamos duplicados por título normalizado
osc_tmdb_red_unidos_limpio_df = osc_tmdb_red_unidos_limpio_df.drop_duplicates(subset='titulo_normalizado')

# Mostramos cuántas películas quedan para el análisis final
print(f"Número de películas listas para comparar: {len(osc_tmdb_red_unidos_limpio_df)}")

# Mostramos un ejemplo
print(osc_tmdb_red_unidos_limpio_df[['film', 'winner', 'vote_average']].head())


# Comparo valoraciones de ganadoras vs no ganadoras
# Separamos los dos grupos
ganadoras = osc_tmdb_red_unidos_limpio_df[osc_tmdb_red_unidos_limpio_df['winner'] == True]
no_ganadoras = osc_tmdb_red_unidos_limpio_df[osc_tmdb_red_unidos_limpio_df['winner'] == False]

# Mostramos cuántas hay en cada grupo
print(f"Número de películas GANADORAS: {len(ganadoras)}")
print(f"Número de películas NO GANADORAS: {len(no_ganadoras)}")

# Estadísticas descriptivas para cada grupo
print("\n--- Estadísticas de valoración (GANADORAS) ---")
print(ganadoras['vote_average'].describe())

print("\n--- Estadísticas de valoración (NO GANADORAS) ---")
print(no_ganadoras['vote_average'].describe())


# Creamos listas de valores para cada grupo
valores_ganadoras = osc_tmdb_red_unidos_limpio_df[osc_tmdb_red_unidos_limpio_df['winner'] == True]['vote_average']
valores_no_ganadoras = osc_tmdb_red_unidos_limpio_df[osc_tmdb_red_unidos_limpio_df['winner'] == False]['vote_average']

# Creamos el boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([valores_ganadoras, valores_no_ganadoras], labels=['Ganadoras', 'No ganadoras'])

# Añadimos etiquetas y título
plt.title('Comparación de valoraciones IMDb\nGanadoras vs No ganadoras de premios')
plt.ylabel('Valoración media (vote_average)')
plt.grid(True)

# Guardamos el gráfico

guardar_grafico()



from scipy.stats import ttest_ind

# ganadoras = osc_tmdb_red_unidos_limpio_df[osc_tmdb_red_unidos_limpio_df['winner'] == True]
# no_ganadoras = osc_tmdb_red_unidos_limpio_df[osc_tmdb_red_unidos_limpio_df['winner'] == False]

# Listas de notas para cada grupo
notas_ganadoras = ganadoras['vote_average']
notas_no_ganadoras = no_ganadoras['vote_average']

# Test t de diferencia de medias
resultado_ttest = ttest_ind(notas_ganadoras, notas_no_ganadoras)

# Mostramos el resultado
print(f"T-statistic: {resultado_ttest.statistic}")
print(f"P-valor: {resultado_ttest.pvalue}")



# HIPÓTESIS 3

# Cargamos el data de TMDb
tmdb_df= pd.read_csv('../dataF/tmdb_5000_movies.csv')
tmdb_df

# Mostramos la información básica de este dataframe
tmdb_df.info()

# Creamos un nuevo DataFrame solo con las columnas necesarias
tiempo_valoración_df = tmdb_df[['runtime', 'vote_average']]

# Mostramos información básica del nuevo DataFrame
tiempo_valoración_df.info()


tiempo_valoración_df

# 1. Eliminamos las 2 filas con runtime nulo
tiempo_valoración_df = tiempo_valoración_df.dropna(subset=['runtime'])

# 2. Eliminamos filas donde runtime sea igual a 0
tiempo_valoración_df = tiempo_valoración_df[tiempo_valoración_df['runtime'] != 0]

# 3. Como sabemos que no hay NaN, eliminamos sólo las filas donde vote_average sea igual a 0
tiempo_valoración_df = tiempo_valoración_df[tiempo_valoración_df['vote_average'] != 0]

# Mostramos información básica del nuevo DataFrame
tiempo_valoración_df.info()


# Renombramos las columnas a español
tiempo_valoración_df = tiempo_valoración_df.rename(columns={
    'runtime': 'duración',
    'vote_average': 'valoración_media'
})

tiempo_valoración_df.head()


# Nueva función para clasificar en 4 grupos
def clasificar_duración(minutos):
    if minutos < 60:
        return 'Muy corta'
    elif 60 <= minutos < 90:
        return 'Corta'
    elif 90 <= minutos <= 120:
        return 'Media'
    else:
        return 'Larga'

# Aplicamos la función al DataFrame
tiempo_valoración_df['grupo_duración'] = tiempo_valoración_df['duración'].apply(clasificar_duración)

# Ver algunas filas para revisar que se asignó correctamente
print(tiempo_valoración_df[['duración', 'grupo_duración']].head(10))

# Contar cuántas películas hay por grupo
tiempo_valoración_df['grupo_duración'].value_counts()


# Creamos un nuevo DataFrame excluyendo el grupo 'Muy corta'
tiempo_valoración_filtrado = tiempo_valoración_df[tiempo_valoración_df['grupo_duración'] != 'Muy corta']

# Contamos cuántas películas hay en cada grupo después de eliminar 'Muy corta'
tiempo_valoración_filtrado['grupo_duración'].value_counts()


# Calculamos la valoración media de cada grupo
medias_por_grupo = tiempo_valoración_filtrado.groupby('grupo_duración')['valoración_media'].mean()

# Mostramos los resultados
print(medias_por_grupo)


# Ordenamos la serie de medias de menor a mayor
medias_ordenadas = medias_por_grupo.sort_values()
# Configuramos los datos para el gráfico
grupos = medias_ordenadas.index
valoraciones = medias_por_grupo.values

# Creamos la gráfica de barras
plt.figure(figsize=(8, 5))  # Tamaño del gráfico
plt.bar(grupos, valoraciones)

# Añadimos títulos y etiquetas
plt.title('Valoración media por grupo de duración')
plt.xlabel('Duración por Grupo')
plt.ylabel('Valoración media')

# Guardamos el gráfico

guardar_grafico()


# Colores en gradación azul: más claro para "Corta", más oscuro para "Larga"
colores_azules = sns.color_palette("Blues", n_colors=3)

# Ordenamos las medias
medias_ordenadas = medias_por_grupo.sort_values()
grupos = medias_ordenadas.index
valoraciones = medias_ordenadas.values

# Creamos el gráfico
plt.figure(figsize=(8, 5))
barras = plt.bar(grupos, valoraciones, color=colores_azules)

# Títulos y etiquetas
plt.title('Valoración media por duración')
plt.xlabel('Duración', labelpad=15)
plt.ylabel('Valoración media')

# Diccionario de etiquetas de duración
duraciones = {
    'Corta': '60-89 min',
    'Media': '90-120 min',
    'Larga': '>120 min'
}

# Etiquetas de duración encima de cada barra
for i, barra in enumerate(barras):
    altura = barra.get_height()
    grupo = grupos[i]
    texto = duraciones[grupo]
    plt.text(barra.get_x() + barra.get_width() / 2, altura + 0.1, texto,
             ha='center', fontsize=10)

# Guardamos el gráfico

guardar_grafico()


# Usamos 3 tonos progresivos de azul (del más claro al más oscuro)
colores_azules = sns.color_palette("Blues", n_colors=3)

# Creamos el boxplot con esa gradación
plt.figure(figsize=(8, 5))
sns.boxplot(x='grupo_duración', y='valoración_media',
            data=tiempo_valoración_filtrado,
            order=['Corta', 'Media', 'Larga'],
            palette=colores_azules)

# Títulos y etiquetas
plt.title('Distribución de valoraciones por duración')
plt.xlabel('Duración', labelpad=15)
plt.ylabel('Valoración media')

# Guardamos el gráfico

guardar_grafico()


plt.figure(figsize=(8, 5))
sns.regplot(x='duración', y='valoración_media', data=tiempo_valoración_df, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
plt.title('Tendencia entre duración y valoración')
plt.xlabel('Duración (minutos)', labelpad=15)
plt.ylabel('Valoración media')

# Guardamos el gráfico
guardar_grafico()



# HIPÓTESIS 4

# Cargamos el data de TMDb
tmdb_df= pd.read_csv('../dataF/tmdb_5000_movies.csv')
tmdb_df.head(2)

# Mostramos la información básica de este dataframe
tmdb_df.info()

# Importamos la librería json para poder convertir textos JSON en estructuras de Python
import json

# Función para convertir el texto de la columna 'genres' en una lista de nombres de géneros
def extraer_nombres_generos(texto_generos):
    try:
        # Paso 1: Convertimos el texto JSON a una lista de diccionarios
        lista_diccionarios = json.loads(texto_generos)

        # Paso 2: Creamos una lista vacía para guardar los nombres
        nombres_generos = []

        # Paso 3: Recorremos cada diccionario dentro de la lista
        for genero in lista_diccionarios:
            # Extraemos el valor del campo 'name' y lo añadimos a la lista
            nombres_generos.append(genero['name'])

        # Paso 4: Devolvemos la lista de nombres
        return nombres_generos

    except json.JSONDecodeError:
        # Si hay un error al convertir el texto, devolvemos una lista vacía
        return []

# Aplicamos la función a toda la columna 'genres' y creamos una nueva columna 'lista_generos'
tmdb_df['lista_generos'] = tmdb_df['genres'].apply(extraer_nombres_generos)

# Mostramos las primeras filas para comprobar que se ha creado correctamente la nueva columna
# para que te muestre 200 filas:
pd.set_option('display.max_rows', 200)

tmdb_df[['genres', 'lista_generos']].head()


# Creamos un conjunto vacío para guardar los géneros únicos (sin repeticiones)
generos_unicos = set()

# Recorremos cada fila de la columna 'lista_generos'
for lista in tmdb_df['lista_generos']:
    # Por si hay valores nulos o mal formateados
    if isinstance(lista, list):
        for genero in lista:
            generos_unicos.add(genero)

# Convertimos el conjunto en una lista ordenada alfabéticamente
generos_unicos = sorted(list(generos_unicos))

# Mostramos el número total de géneros únicos encontrados
print(f"Total de géneros únicos: {len(generos_unicos)}")

# Mostramos la lista completa de géneros únicos
for genero in generos_unicos:
    print(genero)


# Comprobamos cuántas filas tienen 'lista_generos' vacía o nula
faltan_generos = tmdb_df['lista_generos'].isnull().sum()
listas_vacias = tmdb_df['lista_generos'].apply(lambda x: isinstance(x, list) and len(x) == 0).sum()

print(f"Filas con valor nulo en 'lista_generos': {faltan_generos}")
print(f"Filas con lista vacía en 'lista_generos': {listas_vacias}")



# Eliminamos las filas donde 'lista_generos' es una lista vacía
tmdb_df = tmdb_df[tmdb_df['lista_generos'].apply(lambda x: len(x) > 0)]

# Comprobamos que ya no hay listas vacías
print(tmdb_df['lista_generos'].apply(lambda x: len(x) == 0).sum())

# Mostramos el número total de filas (películas) en el DataFrame
print(f"Número total de películas: {len(tmdb_df)}")


# Lista definitiva de géneros considerados SERIOS
generos_serios = [
    'Action', 'Crime', 'Documentary', 'Drama', 'Foreign', 'History',
    'Horror', 'Mystery', 'TV Movie', 'Thriller', 'War', 'Western', 'Religion'
]

# Función que clasifica una película como 'seria' o 'ligera' según el % de géneros serios
def clasificar_pelicula(lista_generos):
    contador_serios = 0

    for genero in lista_generos:
        if genero in generos_serios:
            contador_serios += 1

    porcentaje_serios = contador_serios / len(lista_generos)

    if porcentaje_serios > 0.5:
        return 'seria'
    else:
        return 'ligera'

# Aplicamos la función al DataFrame
tmdb_df['tipo_pelicula'] = tmdb_df['lista_generos'].apply(clasificar_pelicula)

# Mostramos algunas filas para comprobar que se ha aplicado correctamente
tmdb_df[['lista_generos', 'tipo_pelicula']].head(15)


# Agrupamos el DataFrame por 'tipo_pelicula' y calculamos la media de 'vote_average'
medias_por_tipo = tmdb_df.groupby('tipo_pelicula')['vote_average'].mean()

# Mostramos el resultado
print("Media de valoración (vote_average) por tipo de película:")
print(medias_por_tipo)



# Ajustamos el tamaño de la figura
plt.figure(figsize=(8, 5))

# Creamos el boxplot con seaborn
sns.boxplot(
    data=tmdb_df,
    x='tipo_pelicula',        # Clasificación: 'seria' o 'ligera'
    y='vote_average',         # Valoración media
    palette='pastel'          # Colores suaves (opcional)
)

# Añadimos título y etiquetas
plt.title('Distribución de valoraciones por tipo de película (boxplot)')
plt.xlabel('Tipo de película')
plt.ylabel('Valoración media (vote_average)')

# Guardamos el gráfico

guardar_grafico()


# Separamos las valoraciones según el tipo de película para calcular la varianza
# para hacer el t-test de Student 
valoraciones_serias = tmdb_df[tmdb_df['tipo_pelicula'] == 'seria']['vote_average']
valoraciones_ligeras = tmdb_df[tmdb_df['tipo_pelicula'] == 'ligera']['vote_average']

# comprobamos si tienen varianzas distintas (para calcular el t-test de Student)
print("Varianza películas serias:", valoraciones_serias.var())
print("Varianza películas ligeras:", valoraciones_ligeras.var())



from scipy.stats import ttest_ind

# Separamos las valoraciones por tipo de película
valoraciones_serias = tmdb_df[tmdb_df['tipo_pelicula'] == 'seria']['vote_average']
valoraciones_ligeras = tmdb_df[tmdb_df['tipo_pelicula'] == 'ligera']['vote_average']

# Realizamos el test t para muestras independientes (varianzas diferentes)
t_stat, p_valor = ttest_ind(valoraciones_serias, valoraciones_ligeras, equal_var=False)

# Mostramos los resultados
print(f'Estadístico t: {t_stat:.4f}')
print(f'Valor p: {p_valor:.4f}')


# HIPÓTESIS 5

# Leemos el archivo con datos de casting
casting_df = pd.read_csv("../dataF/AllMoviesCastingRaw.csv", sep=";")

# Leemos el archivo de películas y lo llamamos 'tmdb_df'
tmdb_df = pd.read_csv("../dataF/tmdb_5000_movies.csv")

# Mostramos la información del DataFrame de casting
print("📄 Información de 'casting_df':")
casting_df.info()

print("\n" + "-"*80 + "\n")

# Mostramos la información del DataFrame de películas
print("🎬 Información de 'tmdb_df':")
tmdb_df.info()


 #Vamos a comprobar si los id del tmdb_df también aparecen en casting_df y si están 
 # duplicados allí

# Contamos cuántas veces aparece cada 'id' en casting_df
conteo_ids_casting = casting_df['id'].value_counts()

# Mostramos cuántos ids tienen más de una fila en casting_df
print("Número de ids duplicados en casting_df:", (conteo_ids_casting > 1).sum())

# Y cuántos hay en total en tmdb_df
print("Número total de películas en tmdb_df:", tmdb_df['id'].nunique())


# Extraemos solo la columna 'id' del DataFrame de casting
# Así creamos una versión más pequeña que solo tiene los identificadores
casting_temp = casting_df[['id']]

# Extraemos del DataFrame tmdb_df las columnas 'id' y 'title'
# 'title' contiene el nombre de la película, que usaremos para comprobar si los ids coinciden con los títulos
tmdb_temp = tmdb_df[['id', 'title']]

# Hacemos una unión (merge) de los dos DataFrames por la columna 'id'
# 'inner' significa que solo se incluirán los ids que existan en ambos DataFrames
comparacion = pd.merge(casting_temp, tmdb_temp, on="id", how="inner")

# Mostramos las primeras 10 filas del resultado
# Esto nos permitirá ver qué títulos hay para los ids que existen en ambos archivos
print(comparacion.head(10))


# Seleccionamos columnas de interés del DataFrame de casting
# Para ver qué actores están en cada id
casting_verificacion = casting_df[['id', 'actor1_name', 'actor2_name', 'actor3_name']]

# Hacemos un merge con tmdb_df para obtener el título de la película
verificacion_completa = pd.merge(casting_verificacion, tmdb_df[['id', 'title']], on='id', how='inner')

# Mostramos algunas filas
verificacion_completa.head(50)


# Paso 1: Extraemos las columnas que necesitamos de cada DataFrame
# Del DataFrame de casting nos quedamos con 'id' y 'actor_number'
casting_reducido = casting_df[['id', 'actor_number']]

# Del DataFrame tmdb_df nos quedamos con 'id', 'title' y 'vote_average'
tmdb_reducido = tmdb_df[['id', 'title', 'vote_average']]

# Paso 2: Unimos ambos DataFrames por 'id'
# Solo conservará las películas que existan en los dos archivos
dfs_combinados = pd.merge(casting_reducido, tmdb_reducido, on='id', how='inner')

# Paso 3: Mostramos la información del nuevo DataFrame
print(" Información del nuevo DataFrame combinado:")
dfs_combinados.info()

# Paso 4: Mostramos las primeras filas para ver cómo ha quedado
print("\n Primeras filas del nuevo DataFrame:")
dfs_combinados.head()


# Creamos una lista con los nombres de las columnas que queremos conservar
columnas_actores = ['id', 'actor1_name', 'actor2_name', 'actor3_name', 'actor4_name', 'actor5_name']

# Creamos una copia del DataFrame casting_df con solo esas columnas
actores_df = casting_df[columnas_actores].copy()

# Mostramos las primeras 25 filas del nuevo DataFrame para ver cómo están los datos
print("🎬 Primeras 25 filas del DataFrame 'actores_df':")
actores_df.head(25)


# Vamos a reemplazar el texto 'none' por valores nulos (NaN) en las columnas de actores
# Esto nos permitirá luego contar cuántos actores reales tiene cada película

# Aplicamos esto solo a las columnas actor1_name a actor5_name (excluimos 'id')
columnas_nombres = ['actor1_name', 'actor2_name', 'actor3_name', 'actor4_name', 'actor5_name']

# Usamos un bucle para recorrer cada columna y hacer el reemplazo
for columna in columnas_nombres:
    actores_df[columna] = actores_df[columna].replace('none', pd.NA)

# Mostramos de nuevo las 25 primeras filas para comprobar que ya no aparece 'none'
print("Revisión después de reemplazar 'none' por NaN:")
actores_df.head(25)


# Paso 1: Seleccionamos solo las columnas que contienen los nombres de actores principales
columnas_nombres = ['actor1_name', 'actor2_name', 'actor3_name', 'actor4_name', 'actor5_name']

# Paso 2: Creamos una nueva columna que contará cuántos de esos campos NO están vacíos (no son NaN)
# .notna() devuelve True donde hay valor, y False donde hay vacío
# .sum(axis=1) suma los True en cada fila (True cuenta como 1)
actores_df['n_act'] = actores_df[columnas_nombres].notna().sum(axis=1)

# Paso 3: Mostramos algunas filas para comprobar que la nueva columna se ha añadido correctamente
print("📊 Recuento de actores principales por película:")
actores_df.head(25)



# Paso 3: Unimos ambos DataFrames por la columna 'id'
dfs_unidos = pd.merge(actores_df, tmdb_reducido, on='id', how='inner')

# Paso 4: Mostramos información básica del DataFrame final
print("ℹ️ Información del DataFrame 'dfs_unidos':")
dfs_unidos.info()

# Paso 5: Mostramos las primeras filas para comprobar que se ha unido bien
print("\n📄 Primeras 25 filas del DataFrame combinado:")
dfs_unidos.head(25)


# Definimos el nuevo orden de columnas
nuevo_orden_columnas = [
    'id', 'title',                      # Primero el identificador y el título
    'actor1_name', 'actor2_name',       # Luego los actores
    'actor3_name', 'actor4_name', 'actor5_name',
    'n_act',            # Después el número de actores principales
    'vote_average'                      # Y por último la valoración
]

# Reordenamos el DataFrame usando la lista anterior
dfs_unidos = dfs_unidos[nuevo_orden_columnas]

# Mostramos las primeras filas para comprobar que se ha hecho correctamente
print("✅ Columnas reordenadas:")
dfs_unidos.head(5)


# Cambiamos el nombre de la columna 'vote_average' por 'vot_av'
dfs_unidos = dfs_unidos.rename(columns={'vote_average': 'vot_av'})

# Mostramos las primeras filas para confirmar el cambio
print("✅ Cambio de nombre aplicado. Vista previa:")
dfs_unidos.head(5)



dfs_unidos.info()


# 🎭 Paso 1: Filtramos las películas que tienen 0 actores principales
peliculas_sin_actores = dfs_unidos[dfs_unidos['n_act'] == 0]

# Mostramos cuántas son
print(f"🎭 Número de películas sin actores principales: {len(peliculas_sin_actores)}")

# Mostramos las primeras filas para inspección
peliculas_sin_actores.head(10)


# Paso 1: Filtramos las películas con valoración igual a 0
peliculas_con_valoracion_cero = dfs_unidos[dfs_unidos['vot_av'] == 0]

# Paso 2: Contamos cuántas hay
cantidad_valoracion_cero = len(peliculas_con_valoracion_cero)
print(f"🎯 Películas con valoración 0: {cantidad_valoracion_cero}")

# Paso 3 (opcional): Vemos algunas de ellas para entender el tipo de películas que son
print(peliculas_con_valoracion_cero[['title', 'vot_av']].head(10))


# Creamos un nuevo DataFrame filtrado: sin películas con 0 actores ni votación 0
dfs_limpio = dfs_unidos[(dfs_unidos['n_act'] > 0) & (dfs_unidos['vot_av'] > 0)]

# Mostramos cuántas películas quedan tras limpiar
print(f"✅ Películas disponibles para análisis después de limpiar: {len(dfs_limpio)}")

# Mostramos una vista previa para confirmar que todo está en orden
dfs_limpio.head(10)


# 1️⃣ Total de películas antes de la limpieza
total_original = len(dfs_unidos)
print(f"🎬 Total original de películas: {total_original}")

# 2️⃣ Cuántas películas tienen 0 actores
sin_actores = (dfs_unidos['n_act'] == 0).sum()
print(f"🎭 Películas con 0 actores: {sin_actores}")

# 3️⃣ Cuántas películas tienen votación igual a 0
vot_cero = (dfs_unidos['vot_av'] == 0).sum()
print(f"⭐ Películas con vot_av = 0: {vot_cero}")

# 4️⃣ Cuántas cumplen ambas condiciones a la vez (para evitar contarlas dos veces)
ambas_condiciones = ((dfs_unidos['n_act'] == 0) & (dfs_unidos['vot_av'] == 0)).sum()
print(f"🔁 Películas con 0 actores Y votación 0: {ambas_condiciones}")

# 5️⃣ Películas eliminadas realmente (sin contar duplicados)
eliminadas = sin_actores + vot_cero - ambas_condiciones
print(f"🧹 Total de películas eliminadas: {eliminadas}")

# 6️⃣ Total esperado tras la limpieza
esperado_final = total_original - eliminadas
print(f"📊 Total esperado tras limpieza: {esperado_final}")

# 7️⃣ Total real en dfs_limpio
real_final = len(dfs_limpio)
print(f"✅ Total real en dfs_limpio: {real_final}")

# 8️⃣ Comprobación final
if real_final == esperado_final:
    print("✔️ Limpieza verificada: todo cuadra.")
else:
    print("⚠️ Algo no cuadra: revisar condiciones.")


# 1️⃣ Contamos cuántos títulos únicos hay
titulos_unicos = dfs_limpio['title'].nunique()
print(f"🔢 Número de títulos únicos: {titulos_unicos}")

# 2️⃣ Contamos el total de filas (películas)
total_peliculas = len(dfs_limpio)
print(f"🎬 Total de películas en dfs_limpio: {total_peliculas}")

# 3️⃣ Calculamos cuántos títulos están repetidos
titulos_repetidos = total_peliculas - titulos_unicos
print(f"♻️ Títulos repetidos: {titulos_repetidos}")

# 4️⃣ (Opcional) Mostrar los títulos duplicados y cuántas veces se repiten
print("\n📋 Títulos repetidos (top 10):")
repetidos = dfs_limpio['title'].value_counts()
repetidos = repetidos[repetidos > 1].head(10)
print(repetidos)


# 1️⃣ Obtenemos los títulos que aparecen más de una vez
titulos_duplicados = dfs_limpio['title'].value_counts()
titulos_duplicados = titulos_duplicados[titulos_duplicados > 1].index.tolist()

# 2️⃣ Filtramos el DataFrame para mostrar solo las películas con esos títulos repetidos
registros_duplicados = dfs_limpio[dfs_limpio['title'].isin(titulos_duplicados)]

# 3️⃣ Ordenamos por título para verlos juntos
registros_duplicados = registros_duplicados.sort_values(by='title')

# 4️⃣ Mostramos el resultado

pd.set_option('display.max_rows', None)  # Mostrar todas las filas si hay pocas
registros_duplicados.reset_index(drop=True, inplace=True)
registros_duplicados


# Función que clasifica cada película según el número de actores principales
def clasificar_grupo(n):
    """
    Clasifica una película en uno de tres grupos según cuántos actores principales tiene.
    """
    if n <= 2:
        return '1-2'     # Grupo con 1 o 2 actores
    elif n <= 4:
        return '3-4'     # Grupo con 3 o 4 actores
    else:
        return '5+'      # Grupo con 5 o más actores

# Aplicamos la función a la columna 'n_act' y creamos la nueva columna 'g_act'
dfs_limpio['g_act'] = dfs_limpio['n_act'].apply(clasificar_grupo)

# Mostramos una vista previa para verificar
print("✅ Grupos creados y asignados correctamente:")
dfs_limpio.head(25)



# Agrupamos las películas por el grupo de actores 'g_act'
# y calculamos la media de valoración 'vot_av' en cada grupo
media_por_grupo = dfs_limpio.groupby('g_act')['vot_av'].mean().reset_index()

# Ordenamos los grupos en el orden deseado: '1-2', '3-4', '5+'
orden_grupos = ['1-2', '3-4', '5+']
media_por_grupo['g_act'] = pd.Categorical(media_por_grupo['g_act'], categories=orden_grupos, ordered=True)
media_por_grupo = media_por_grupo.sort_values('g_act')

# Mostramos el resultado
print("📊 Valoración media por grupo de actores:")
media_por_grupo


# Definimos los datos
grupos = media_por_grupo['g_act']
medias = media_por_grupo['vot_av']

# Paleta de azules claros a oscuros
colores = ['#ADD8E6', '#4682B4', '#0B3D91']

# Crear gráfico de barras con colores personalizados
plt.figure(figsize=(8, 5))
plt.bar(grupos, medias, color=colores, width=0.6)

# Añadir etiquetas y título con separación mejorada
plt.title("Valoración media según número de actores principales", pad=15)
plt.xlabel("Nº de actores principales", labelpad=15)
plt.ylabel("Valoración IMDB", labelpad=10)

# Mostrar valores sobre cada barra
for i, valor in enumerate(medias):
    plt.text(i, valor + 0.03, f"{valor:.2f}", ha='center', fontsize=10)

plt.tight_layout()

# Guardamos el gráfico

guardar_grafico()




# Ajustamos el tamaño de la figura
plt.figure(figsize=(8, 5))

# Creamos el boxplot con seaborn
sns.boxplot(
    data=dfs_limpio,
    x='g_act',        # Clasificación: 'seria' o 'ligera'
    y='vot_av',         # Valoración media
    palette='pastel'          # Colores suaves (opcional)
)

# Añadimos título y etiquetas
plt.title('Distribución de valoraciones por tipo de película (boxplot)')
plt.xlabel('Tipo de película')
plt.ylabel('Valoración media (vote_average)')

# Guardamos el gráfico

guardar_grafico()