# C9T1 - Aplicaciones Prácticas de Arreglos Multidimensionales en Ingeniería

## Descripción

Este repositorio contiene **un solo archivo Python** con las cinco soluciones solicitadas. El programa demuestra el uso correcto de arreglos bidimensionales y tridimensionales en análisis estructural, propagación de presión, procesamiento de imágenes médicas, análisis de sensores y transformación de coordenadas.

## Estructura

```text
C9T1_entrega_un_archivo/
├── C9T1_arreglos_multidimensionales.py
├── README.md
├── requirements.txt
├── .gitignore
└── resultados/
    ├── problema1_fuerzas.png
    ├── problema2_presion_3d.png
    ├── problema3_suavizado_medico.png
    ├── problema4_sensores.png
    └── problema5_transformacion.png
```

## Requisitos y ejecución

Requiere Python 3.10 o superior, NumPy y Matplotlib.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux o macOS
source .venv/bin/activate
pip install -r requirements.txt
python C9T1_arreglos_multidimensionales.py
```

El programa ejecuta los cinco problemas consecutivamente y guarda las visualizaciones en `resultados/`.

## Solución y eficiencia

### Problema 1: análisis de fuerzas en una estructura

Una matriz 3x3 almacena las fuerzas verticales aplicadas en nueve nodos. Las columnas representan posiciones de 0, 3 y 6 metros. Las reacciones de los apoyos se obtienen mediante `ΣFy = 0` y `ΣM_A = 0`, y se verifican los residuos de equilibrio. Las sumas por columnas y el cálculo de momentos se realizan con operaciones vectorizadas. Para `n` nodos, la complejidad es O(n).

### Problema 2: simulación de fluido en una cuadrícula 3D

Tres arreglos tridimensionales de 3x3x3 almacenan presión, temperatura y velocidad. Una perturbación central de presión se propaga durante ocho pasos utilizando hasta seis vecinos ortogonales por celda. Se crea una matriz nueva en cada paso para evitar que el orden de actualización altere el resultado. Para `N` celdas y `t` pasos, la complejidad es O(tN). Es una simulación académica de difusión y no un modelo CFD completo.

### Problema 3: análisis de imágenes médicas 3D

Un volumen sintético de 32x32x32 voxeles representa una región anatómica con ruido gaussiano. Se aplica un filtro promedio 3x3x3 a cada voxel y se compara el error cuadrático medio antes y después. La complejidad es O(Nk³), con `k=3`. Los datos son sintéticos y no deben utilizarse con fines diagnósticos.

### Problema 4: tabla de sensores

Una matriz 5x5 organiza filas como momentos y columnas como sensores. NumPy calcula promedios y desviaciones estándar poblacionales por sensor (`axis=0`) y por momento (`axis=1`). La vectorización evita bucles innecesarios y procesa las 25 mediciones en O(filas x columnas).

### Problema 5: transformación de coordenadas

Una matriz Nx2 almacena puntos cartesianos. Las coordenadas homogéneas permiten combinar escalado, rotación de 30 grados y traslación en una sola matriz 3x3. Todos los puntos se transforman mediante una multiplicación matricial vectorizada con complejidad O(n).

## Cumplimiento de los criterios

### Claridad en la implementación

- Un único archivo organizado en cinco secciones claramente identificadas.
- Funciones con nombres descriptivos, anotaciones de tipos, docstrings y comentarios detallados.
- Variables comprensibles y unidades indicadas en los comentarios y gráficos.
- Cada paso relevante explica su propósito y los supuestos del modelo.

### Aplicación de arreglos multidimensionales

- Matrices 2D de 3x3 y 5x5.
- Volúmenes 3D de 3x3x3 y 32x32x32.
- Operaciones por filas, columnas, capas y celdas vecinas.
- Transformaciones mediante multiplicación matricial.
- Uso vectorizado de NumPy para mejorar claridad y eficiencia.

### Calidad de la solución

- Los cinco problemas producen resultados numéricos y visuales.
- Se incluyen controles de límites para vecinos 3D y verificación de equilibrio.
- El programa completo fue compilado y ejecutado sin errores.
- Las visualizaciones permiten comprobar los resultados de cada solución.

