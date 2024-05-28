from numba import njit, prange # Importar njit y prange para paralelizar el código
import numpy as np # Importar numpy para generar vectores aleatorios
import time

@njit(parallel=True)
def sum_vector_parallel(vector):
    total_sum = 0.0
    for i in prange(len(vector)):
        total_sum += vector[i]
    return total_sum

# el decorador @njit sin el argumento parallel=True no paraleliza el código
@njit
def sum_vector_normal(vector):
    total_sum = 0.0
    for i in range(len(vector)):
        total_sum += vector[i]
    return total_sum

# Crear un vector aleatorio grande
vector = np.random.rand(10000000)

# Medir el tiempo de ejecución de la versión paralelizada
start_time = time.time()
result_parallel = sum_vector_parallel(vector)
end_time = time.time()
print("Resultado paralelizado:", result_parallel)
print("Tiempo de ejecución paralelizado:", end_time - start_time, "segundos")

# Medir el tiempo de ejecución de la versión normal
start_time = time.time()
result_normal = sum_vector_normal(vector)
end_time = time.time()
print("Resultado normal:", result_normal)
print("Tiempo de ejecución normal:", end_time - start_time, "segundos")
