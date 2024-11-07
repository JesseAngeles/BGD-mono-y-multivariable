import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from BGD import monovariable

# Leer el archivo CSV
df = pd.read_csv('resources/casas.csv')

# Dividir dataset
X = df.drop("Precio (MDP)", axis=1)
y = df["Precio (MDP)"]

# 70% entrenamiento, 30% prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0, shuffle=True)

#Reiniciar indices
x_test = x_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Calcular los pesos y guardar el historial de pesos y predicciones
iterations = input() or 5
init_weight = input() or 0
alpha = input() or 0.00000025

weight, weight_history, prediction_history = monovariable(x_train=x_train, y_train=y_train, 
                                                 iterations=int(iterations), 
                                                 weight=float(init_weight), 
                                                 alpha=float(alpha))

# Imprimir los resultados
print("\nY_test\n", y_test.head().values)

# Calcular y mostrar las predicciones para el conjunto de prueba con cada peso
print("\nPredictions:")
for i, weight in enumerate(weight_history):
    prediction = []
    for _, x in x_test.iterrows():
        # Predicción para cada muestra del conjunto de prueba
        prediction.append(float(x.values[0] * weight))
    print(f'Iteration {i}: {prediction}')
    
# Calcular el error de estimación para cada muestra
print("\nEstimation error:")
for i, weight in enumerate(weight_history):
    error_sum = 0
    for j, x in x_test.iterrows(): 
        error_sum += abs(float(x.values[0] * weight) - y_test.iloc[j]) 
    print(f'Iteration {i}: {error_sum}')


# Graficar las predicciones durante las iteraciones (Gráfica 1)
plt.figure(figsize=(12, 6))

# Predicciones por iteración
plt.subplot(1, 2, 1)
for i, prediction in enumerate(prediction_history):
    plt.plot(x_train, prediction, label=f'Iteración {i}')
plt.scatter(x_train, y_train, color='black', label='Datos reales')
plt.xlabel("Características")
plt.ylabel("Precio (MDP)")
plt.title("Predicciones en cada iteración")
plt.legend()

# Error de estimación por iteración (Gráfica 2)
errors = [sum(abs(predictions - y_train)) / len(y_train) for predictions in prediction_history]
plt.subplot(1, 2, 2)
plt.plot(range(len(errors)), errors, label="Error absoluto")
plt.xlabel("Iteraciones")
plt.ylabel("Error de estimación")
plt.title("Error de estimación durante las iteraciones")
plt.legend()

plt.tight_layout()
plt.show()
