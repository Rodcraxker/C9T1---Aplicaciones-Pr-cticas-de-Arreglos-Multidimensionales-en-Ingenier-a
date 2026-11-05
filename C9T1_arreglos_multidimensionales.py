"""
C9T1 - Aplicaciones Prácticas de Arreglos Multidimensionales en Ingeniería
Autor: Rod Munoz

Este único archivo contiene las soluciones de los cinco problemas solicitados.
Cada sección utiliza funciones, nombres descriptivos, comentarios y arreglos
multidimensionales de NumPy. Al ejecutarlo, muestra resultados numéricos y
crea cinco visualizaciones dentro de la carpeta resultados.
"""

# =============================================================================
# PROBLEMA1 FUERZAS
# =============================================================================
"""Problema 1: análisis simplificado de fuerzas en una estructura 3x3.

Cada celda contiene la carga vertical Fy aplicada en un nodo. Se calculan las
reacciones verticales de dos apoyos mediante equilibrio estático global:
ΣFy = 0 y ΣM_A = 0. Supuesto: estructura plana, cargas verticales y dos apoyos
simples ubicados en los extremos de una luz de 6 m.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SALIDA = Path(__file__).parent / "resultados"
SALIDA.mkdir(exist_ok=True)

# Filas: niveles de la estructura. Columnas: posiciones x = 0, 3 y 6 m.
# Los valores negativos representan fuerzas verticales hacia abajo, en kN.
fuerzas_y = np.array([
    [-10.0, -15.0, -12.0],
    [ -8.0, -11.0,  -9.0],
    [ -6.0,  -7.0,  -5.0],
], dtype=float)
posiciones_x = np.array([0.0, 3.0, 6.0], dtype=float)
luz = 6.0


def calcular_reacciones(cargas: np.ndarray, x: np.ndarray, longitud: float) -> tuple[float, float]:
    """Calcula reacciones RA y RB usando equilibrio de fuerzas y momentos."""
    cargas_por_columna = cargas.sum(axis=0)
    carga_total_descendente = -cargas_por_columna.sum()
    momento_cargas_en_a = -(cargas_por_columna * x).sum()
    reaccion_b = momento_cargas_en_a / longitud
    reaccion_a = carga_total_descendente - reaccion_b
    return reaccion_a, reaccion_b


def verificar_equilibrio(cargas: np.ndarray, x: np.ndarray, ra: float, rb: float) -> tuple[float, float]:
    """Devuelve residuos numéricos de ΣFy y ΣM_A; deben ser cercanos a cero."""
    residuo_fuerzas = ra + rb + cargas.sum()
    residuo_momentos = rb * luz + (cargas.sum(axis=0) * x).sum()
    return residuo_fuerzas, residuo_momentos


def visualizar(cargas: np.ndarray) -> None:
    """Genera un mapa de calor de las cargas nodales."""
    fig, ax = plt.subplots(figsize=(6, 4))
    imagen = ax.imshow(cargas, cmap="coolwarm", vmin=cargas.min(), vmax=0)
    for i in range(cargas.shape[0]):
        for j in range(cargas.shape[1]):
            ax.text(j, i, f"{cargas[i, j]:.0f} kN", ha="center", va="center")
    ax.set_title("Fuerzas verticales aplicadas en nodos")
    ax.set_xlabel("Columna estructural")
    ax.set_ylabel("Nivel")
    fig.colorbar(imagen, ax=ax, label="Fuerza Fy (kN)")
    fig.tight_layout()
    fig.savefig(SALIDA / "problema1_fuerzas.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ra, rb = calcular_reacciones(fuerzas_y, posiciones_x, luz)
    rf, rm = verificar_equilibrio(fuerzas_y, posiciones_x, ra, rb)
    print("Matriz de fuerzas Fy (kN):\n", fuerzas_y)
    print(f"Reacción en A: {ra:.2f} kN")
    print(f"Reacción en B: {rb:.2f} kN")
    print(f"Residuo ΣFy: {rf:.2e} kN")
    print(f"Residuo ΣM_A: {rm:.2e} kN·m")
    visualizar(fuerzas_y)

# =============================================================================
# PROBLEMA2 FLUIDO 3D
# =============================================================================
"""Problema 2: propagación simplificada de presión en una cuadrícula 3x3x3.

El estado se almacena en arreglos tridimensionales para presión, temperatura y
velocidad. La presión se actualiza con un esquema explícito de difusión usando
los seis vecinos ortogonales. Es un modelo académico, no un solver CFD.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SALIDA = Path(__file__).parent / "resultados"
SALIDA.mkdir(exist_ok=True)
FORMA = (3, 3, 3)
PASOS = 8
COEF_PROPAGACION = 0.12  # Debe permanecer pequeño para estabilidad numérica.

presion = np.full(FORMA, 101.3, dtype=float)       # kPa
presion[1, 1, 1] = 130.0                          # perturbación central
temperatura = np.full(FORMA, 25.0, dtype=float)   # °C
velocidad = np.zeros(FORMA + (3,), dtype=float)   # vector [vx, vy, vz] m/s


def vecinos_ortogonales(celda: tuple[int, int, int], forma: tuple[int, int, int]):
    """Genera únicamente vecinos válidos, evitando accesos fuera del arreglo."""
    z, y, x = celda
    for dz, dy, dx in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
        nz, ny, nx = z + dz, y + dy, x + dx
        if 0 <= nz < forma[0] and 0 <= ny < forma[1] and 0 <= nx < forma[2]:
            yield nz, ny, nx


def actualizar_presion(actual: np.ndarray, coeficiente: float) -> np.ndarray:
    """Calcula simultáneamente el siguiente estado para evitar sesgo por orden."""
    siguiente = actual.copy()
    for indice in np.ndindex(actual.shape):
        valores_vecinos = [actual[v] for v in vecinos_ortogonales(indice, actual.shape)]
        promedio_vecinos = np.mean(valores_vecinos)
        siguiente[indice] = actual[indice] + coeficiente * (promedio_vecinos - actual[indice])
    return siguiente


def simular(inicial: np.ndarray, pasos: int) -> list[np.ndarray]:
    """Conserva el historial necesario para comparar inicio y final."""
    historial = [inicial.copy()]
    actual = inicial.copy()
    for _ in range(pasos):
        actual = actualizar_presion(actual, COEF_PROPAGACION)
        historial.append(actual.copy())
    return historial


def visualizar(inicial: np.ndarray, final: np.ndarray) -> None:
    """Compara la capa central antes y después de la propagación."""
    fig, ejes = plt.subplots(1, 2, figsize=(9, 4))
    minimo, maximo = inicial.min(), inicial.max()
    for ax, datos, titulo in zip(ejes, (inicial[1], final[1]), ("Inicial", f"Después de {PASOS} pasos")):
        im = ax.imshow(datos, cmap="plasma", vmin=minimo, vmax=maximo)
        ax.set_title(titulo)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.colorbar(im, ax=ejes, label="Presión (kPa)", shrink=0.8)
    fig.suptitle("Propagación de presión - capa central z=1")
    fig.savefig(SALIDA / "problema2_presion_3d.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    historial = simular(presion, PASOS)
    final = historial[-1]
    print("Presión inicial, capa z=1:\n", presion[1])
    print("Presión final, capa z=1:\n", np.round(final[1], 3))
    print(f"Temperatura media: {temperatura.mean():.2f} °C")
    print(f"Velocidad media: {np.linalg.norm(velocidad, axis=3).mean():.2f} m/s")
    visualizar(presion, final)

# =============================================================================
# PROBLEMA3 IMAGEN MEDICA 3D
# =============================================================================
"""Problema 3: suavizado de un volumen médico 3D mediante filtro promedio.

Se usa un volumen sintético reproducible porque la tarea no proporciona datos
clínicos. El filtro 3x3x3 reduce ruido promediando el vecindario de cada voxel.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SALIDA = Path(__file__).parent / "resultados"
SALIDA.mkdir(exist_ok=True)
SEMILLA = 42


def crear_volumen_sintetico(tamano: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Crea un objeto esférico limpio y una versión con ruido gaussiano."""
    z, y, x = np.indices((tamano, tamano, tamano))
    centro = (tamano - 1) / 2
    radio = tamano * 0.28
    volumen_limpio = np.where((x-centro)**2 + (y-centro)**2 + (z-centro)**2 <= radio**2, 180.0, 30.0)
    rng = np.random.default_rng(SEMILLA)
    ruido = rng.normal(0, 25, volumen_limpio.shape)
    volumen_ruidoso = np.clip(volumen_limpio + ruido, 0, 255)
    return volumen_limpio, volumen_ruidoso


def filtro_promedio_3d(volumen: np.ndarray, radio: int = 1) -> np.ndarray:
    """Suaviza cada voxel con promedio local 3D, incluyendo bordes replicados."""
    acolchado = np.pad(volumen, radio, mode="edge")
    suavizado = np.empty_like(volumen, dtype=float)
    for indice in np.ndindex(volumen.shape):
        z, y, x = indice
        vecindario = acolchado[z:z+2*radio+1, y:y+2*radio+1, x:x+2*radio+1]
        suavizado[indice] = vecindario.mean()
    return suavizado


def error_cuadratico_medio(referencia: np.ndarray, estimacion: np.ndarray) -> float:
    return float(np.mean((referencia - estimacion) ** 2))


def visualizar(limpio: np.ndarray, ruidoso: np.ndarray, suavizado: np.ndarray) -> None:
    capa = limpio.shape[0] // 2
    fig, ejes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, imagen, titulo in zip(ejes, (limpio[capa], ruidoso[capa], suavizado[capa]),
                                  ("Referencia", "Con ruido", "Suavizada")):
        ax.imshow(imagen, cmap="gray", vmin=0, vmax=255)
        ax.set_title(titulo)
        ax.axis("off")
    fig.suptitle("Filtro promedio 3D - capa central")
    fig.tight_layout()
    fig.savefig(SALIDA / "problema3_suavizado_medico.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    limpio, ruidoso = crear_volumen_sintetico()
    suavizado = filtro_promedio_3d(ruidoso)
    print(f"ECM antes del filtrado: {error_cuadratico_medio(limpio, ruidoso):.2f}")
    print(f"ECM después del filtrado: {error_cuadratico_medio(limpio, suavizado):.2f}")
    visualizar(limpio, ruidoso, suavizado)

# =============================================================================
# PROBLEMA4 SENSORES
# =============================================================================
"""Problema 4: análisis de una tabla 5x5 de temperatura.

Convención solicitada: filas = momentos en el tiempo; columnas = sensores.
Se calculan media y desviación estándar poblacional por fila y por columna.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SALIDA = Path(__file__).parent / "resultados"
SALIDA.mkdir(exist_ok=True)

temperaturas = np.array([
    [22.1, 21.8, 22.5, 23.0, 22.7],
    [22.4, 22.0, 22.8, 23.2, 22.9],
    [22.8, 22.3, 23.1, 23.5, 23.2],
    [23.0, 22.6, 23.4, 23.7, 23.5],
    [23.3, 22.9, 23.7, 24.0, 23.8],
], dtype=float)


def analizar(datos: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula métricas vectorizadas; axis=0 por sensor y axis=1 por momento."""
    return {
        "promedio_sensor": datos.mean(axis=0),
        "desviacion_sensor": datos.std(axis=0, ddof=0),
        "promedio_momento": datos.mean(axis=1),
        "desviacion_momento": datos.std(axis=1, ddof=0),
    }


def visualizar(datos: np.ndarray, metricas: dict[str, np.ndarray]) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    im = ejes[0].imshow(datos, cmap="coolwarm", aspect="auto")
    ejes[0].set_title("Temperatura por momento y sensor")
    ejes[0].set_xlabel("Sensor")
    ejes[0].set_ylabel("Momento")
    ejes[0].set_xticks(range(5), [f"S{i}" for i in range(1, 6)])
    ejes[0].set_yticks(range(5), [f"T{i}" for i in range(1, 6)])
    fig.colorbar(im, ax=ejes[0], label="°C")

    x = np.arange(5)
    ejes[1].errorbar(x, metricas["promedio_sensor"], yerr=metricas["desviacion_sensor"],
                     fmt="o-", capsize=5, label="Media ± desviación")
    ejes[1].set_xticks(x, [f"S{i}" for i in range(1, 6)])
    ejes[1].set_title("Promedio y desviación por sensor")
    ejes[1].set_xlabel("Sensor")
    ejes[1].set_ylabel("Temperatura (°C)")
    ejes[1].grid(alpha=0.3)
    ejes[1].legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "problema4_sensores.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    metricas = analizar(temperaturas)
    print("Tabla 5x5 (filas=momentos, columnas=sensores):\n", temperaturas)
    for nombre, valores in metricas.items():
        print(nombre, np.round(valores, 3))
    visualizar(temperaturas, metricas)

# =============================================================================
# PROBLEMA5 TRANSFORMACION
# =============================================================================
"""Problema 5: transformación de puntos cartesianos con matrices homogéneas.

Se aplica una rotación de 30°, escalado y traslación. Las coordenadas homogéneas
permiten combinar las tres operaciones en una sola multiplicación matricial.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SALIDA = Path(__file__).parent / "resultados"
SALIDA.mkdir(exist_ok=True)

puntos = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [2.0, 1.0],
    [0.0, 1.0],
    [0.0, 0.0],  # cierre del polígono para visualizarlo
], dtype=float)


def matriz_transformacion(angulo_grados: float, escala: tuple[float, float],
                           traslacion: tuple[float, float]) -> np.ndarray:
    """Construye T @ R @ S: escala, luego rota y finalmente traslada."""
    angulo = np.deg2rad(angulo_grados)
    c, s = np.cos(angulo), np.sin(angulo)
    sx, sy = escala
    tx, ty = traslacion
    escalado = np.array([[sx,0,0],[0,sy,0],[0,0,1]], dtype=float)
    rotacion = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
    traslado = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=float)
    return traslado @ rotacion @ escalado


def transformar(coordenadas: np.ndarray, matriz: np.ndarray) -> np.ndarray:
    """Convierte Nx2 a homogéneas Nx3 y aplica la matriz en forma vectorizada."""
    homogeneas = np.column_stack((coordenadas, np.ones(len(coordenadas))))
    resultado = (matriz @ homogeneas.T).T
    return resultado[:, :2]


def visualizar(originales: np.ndarray, transformados: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(originales[:,0], originales[:,1], "o-", label="Original")
    ax.plot(transformados[:,0], transformados[:,1], "s-", label="Transformado")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.set_title("Transformación lineal de coordenadas")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "problema5_transformacion.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    matriz = matriz_transformacion(30, (1.5, 0.8), (3.0, 2.0))
    nuevos_puntos = transformar(puntos, matriz)
    print("Matriz de transformación:\n", np.round(matriz, 3))
    print("Puntos originales:\n", puntos)
    print("Puntos transformados:\n", np.round(nuevos_puntos, 3))
    visualizar(puntos, nuevos_puntos)
