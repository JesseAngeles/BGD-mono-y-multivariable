def monovariable(x_train, y_train, iterations, weight, alpha):  
    # Almacenamos el historial de pesos y predicciones
    weight_history = []
    prediction_history = []

    # Número de iteraciones
    for a in range(iterations):  
        grad_sum = 0  

        # Iterar sobre el conjunto de entrenamiento
        for i in range(len(x_train)):
            x = x_train.iloc[i].values[0]
            y = y_train.iloc[i]

            # Cálculo del gradiente para cada muestra
            grad_sum += (weight * x - y) * x

        # Actualización del peso (sin sesgo)
        weight -= alpha * grad_sum

        # Guardar el peso y las predicciones para cada iteración
        weight_history.append(weight)
        
        # Realizar las predicciones en el conjunto de entrenamiento
        predictions = [x_train.iloc[i].values[0] * weight for i in range(len(x_train))]
        prediction_history.append(predictions)
        
        print(f'Iteration {a}: [weight = {weight}]')

    return weight, weight_history, prediction_history

def multivariable(x_train, y_train, iterations, weight, alpha):
    # Historial de pesos e historial de predicciones
    weight_history = []
    prediction_history = []
    
    # Número de iteraciones
    for a in range(iterations):
        grad_sum = [0] * len(x_train.columns)  # Inicializar grad_sum para cada característica

        # Iterar sobre las filas de x_train usando iterrows
        for i, x in x_train.iterrows():
            y = y_train.iloc[i]
            prediction = sum(x.iloc[d] * weight[d] for d in range(len(x)))  # Cambiar x[d] por x.iloc[d]
            error = prediction - y  # Error para la muestra
            
            # Gradiente para cada dimensión (característica)
            for d in range(len(x)):
                grad_sum[d] += error * x.iloc[d]  # Gradiente para la característica d

        # Actualización de los pesos utilizando el gradiente calculado
        for d in range(len(weight)):
            weight[d] -= alpha * grad_sum[d]  # Actualizar el peso para la característica d

        # Guardar el historial de pesos y predicciones
        weight_history.append(weight.copy())
        prediction_history.append([sum(x.iloc[d] * weight[d] for d in range(len(x))) for _, x in x_train.iterrows()])

        # Mostrar los pesos en cada iteración
        print(f"Iteration {a}: {[float(w) for w in weight]}")

    return weight, weight_history, prediction_history