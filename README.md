# Análisis Exploratorio de Datos del Cine

> **Versión de Python empleada:** 3.11.9
> **Estado del proyecto:** **finalizado**

Este repositorio contiene un **análisis exploratorio de datos (EDA)** centrado en el mundo del *cine* y su relación con diversas variables cuantitativas y cualitativas (presupuesto, recaudación, premios Óscar, número de actores protagonistas, géneros, etc.).
El estudio se articula a través de **cinco hipótesis** contrastadas con datos reales procedentes de varias bases públicas.

---

## Tabla de contenido

1. [Estructura del proyecto](#estructura-del-proyecto)
2. [Fuentes de datos](#fuentes-de-datos)
3. [Hipótesis y enfoque analítico](#hipótesis-y-enfoque-analítico)
4. [Dependencias](#dependencias)
5. [Ejecución paso a paso](#ejecución-paso-a-paso)
6. [Resultados clave](#resultados-clave)
7. [Licencia](#licencia)
8. [Contacto](#contacto)

---

## Estructura del proyecto

```
EDA_1/
├── data/         # CSV originales (fuentes de datos)
├── imagenes/     # Gráficos generados automáticamente
├── memoria/      # Memoria completa del EDA (Memoria.ipynb)
├── notebooks/    # 5 notebooks independientes (una por hipótesis)
└── script/       # edacine.py – ejecuta todo el flujo desde consola
```

*El script `script/edacine.py` crea la carpeta `imagenes/` si no existe y guarda allí todas las figuras generadas.*

---

## Fuentes de datos

| Fichero CSV                   | Descripción resumida                                |
| ----------------------------- | --------------------------------------------------- |
| `AllMoviesCastingRaw.csv`     | Reparto original de películas (crudo)               |
| `AllMoviesDetailsCleaned.csv` | Detalles de películas (limpio)                      |
| `full_data.csv`               | Dataset combinado preliminar                        |
| `the_oscar_award.csv`         | Historial de premios Óscar (× categorías)           |
| `tmdb_5000_credits.csv`       | Créditos de TMDb 5000 (reparto/técnico)             |
| `tmdb_5000_movies.csv`        | Metadatos de TMDb 5000 (presupuesto, géneros, etc.) |

---

## Hipótesis y enfoque analítico

| Nº    | Planteamiento                                                                                                                                    | Métricas / técnicas principales                                                      |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| **1** | \*«Cuanto mayor es el **presupuesto**, mayor es la \**recaudación»*                                                                              | Limpieza de *outliers*, dispersogramas, regresión lineal, coeficiente de correlación |
| **2** | *«Las películas que **ganan premios importantes** (Óscar, etc.) obtienen **mejores valoraciones** del público en IMDb que las que no los ganan»* | Cruce con histórico de Óscar, boxplots comparativos, test de Welch                   |
| **3** | \*«La **duración ideal** para alcanzar una alta valoración oscila entre \**90 y 120 minutos»*                                                    | Histogramas de distribución, cálculo de medias móviles, contraste de medias          |
| **4** | *«Las películas con **géneros serios** (drama, thriller, histórico…) reciben **valoraciones superiores** a las de géneros ligeros»*              | Clasificación de géneros, boxplot, t‑test de Student (Welch)                         |
| **5** | *«El **número de actores protagonistas** influye en la valoración: el grupo de **3‑4 actores** logra las medias más altas»*                      | Unión de créditos + películas, agrupación por tramos (1‑2 / 3‑4 / 5+), ANOVA/t‑test  |

Cada hipótesis se documenta paso a paso en su notebook correspondiente y de forma narrativa en `memoria/Memoria.ipynb`, con código, explicaciones y conclusiones parciales.

---

## Dependencias

Las librerías clave aparecen importadas en la memoria, y se recomiendan las siguientes versiones mínimas:

```text
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
scipy>=1.11
```

> **Sugerencia**: genera un entorno virtual y ejecuta:
>
> ```bash
> pip install pandas numpy matplotlib seaborn scipy
> ```

Si deseas un listado congelado, ejecuta `pip freeze > requirements.txt` y súbelo al repo.

---

## Ejecución paso a paso

```bash
# 1. Clona el repositorio
git clone https://github.com/tu-usuario/EDA_1.git
cd EDA_1

# 2. (Opcional) crea un entorno virtual
python -m venv venv
source venv/bin/activate      # Linux / macOS
# .\venv\Scripts\activate    # Windows

# 3. Instala dependencias
pip install -r requirements.txt   # o las librerías listadas arriba

# 4. Lanza el análisis completo
python script/eda.py
```

El script leerá los CSV de `data/`, generará todas las figuras dentro de `imagenes/` y mostrará un resumen de resultados en consola.

---

## Resultados clave

* **Hipótesis 1:** Se observa una correlación positiva (\~0,70) entre presupuesto y recaudación tras eliminar valores extremos.
* **Hipótesis 2:** Las películas galardonadas con Óscar presentan una media de valoración 0,6 puntos superior (p‑valor < 0,05).
* **Hipótesis 3:** El rango de 90‑120 min concentra el 65 % de los títulos con valoración ≥ 7 puntos.
* **Hipótesis 4:** Los géneros denominados “serios” muestran una mediana desplazada a la derecha y diferencia estadísticamente significativa.
* **Hipótesis 5:** El grupo intermedio (3‑4 actores principales) obtiene la mejor valoración media; las diferencias son moderadas pero consistentes.

Consulta la carpeta `imagenes/` y la memoria para gráficos y detalles completos.

---

## Licencia

Actualmente **sin licencia específica**. Si vas a reutilizar el código o los datos, por favor cita la fuente y menciona al autor.

---

## Contacto

Para comentarios o mejoras, abre un *issue* en GitHub o escribe a `eth....com`.

