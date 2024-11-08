import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from BGD import multivariable

#Leer archivo CSV
option = input("1) monovariable\n2) multivariable\n") or 2
if int(option) == 1:
    df = pd.read_csv('resources/casas.csv')
    const_alpha = float(0.00000007)
else:
    df = pd.read_csv('resources/Dataset_multivariable.csv')
    const_alpha = 0.000006
    

#Dividir dataset
X = df.drop(df.columns[-1], axis=1)
y = df[df.columns[-1]]

# 70% entrenamiento, 30% prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0, shuffle=True)

# Reiniciar indices
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# Calcular los pesos
iterations = int(input("Ingrese el número de iteraciones: ") or 5)
init_weight = [] 
for i in enumerate(x_train.columns):
    val = float(input(f'weight {i[0]}: ') or 0)
    init_weight.append(val) 


alpha = float(input("Ingrese la tasa de aprendizaje (alpha): ") or const_alpha)

weight, weight_history, prediction_history = multivariable(x_train, y_train, iterations, init_weight, alpha)

# Imprimir los resultados
print("\nY_test\n", y_test.head().values)

# Calcular y mostrar las predicciones para el conjunto de prueba con cada peso
print("\nPredictions:")
for i, weight in enumerate(weight_history):
    prediction = []
    for _, x in x_test.iterrows():
        # Predicción para cada muestra del conjunto de prueba
        pred = sum(x.iloc[d] * weight[d] for d in range(len(x)))  # Cambiar x[d] por x.iloc[d]
        prediction.append(pred)
    print(f'Iteration {i}: {[float(p) for p in prediction]}')

# Calcular el error de estimación para cada muestra
print("\nEstimation error:")
iteration_errors = []  # Lista para almacenar los errores por iteración

for i, weight in enumerate(weight_history):
    error_sum = 0
    for j, x in x_test.iterrows(): 
        # Cálculo del error para cada muestra
        pred = sum(x.iloc[d] * weight[d] for d in range(len(x)))  # Predicción para cada muestra usando todas las características
        error_sum += abs(pred - y_test.iloc[j])
    iteration_errors.append(error_sum)
    print(f'Iteration {i}: {error_sum}')

if len(weight) == 1:
    plt.subplot(1, 2, 1)
    for i, prediction in enumerate(prediction_history):
        plt.plot(x_train, prediction, label=f'Iteración {i}')
    plt.scatter(x_test, y_test, color='black', label='Datos reales')
    plt.xlabel("Características")
    plt.ylabel("Precio (MDP)")
    plt.title("Predicciones en cada iteración")
    plt.legend()
    
    plt.subplot(1, 2, 2)

# Error de estimación por iteración (Gráfica 2)
plt.plot(range(len(iteration_errors)), iteration_errors, marker='o', label="Error")
plt.xlabel("Iteraciones")
plt.ylabel("Error de estimación")
plt.title("Error de estimación")
plt.legend()

plt.tight_layout()
plt.show()