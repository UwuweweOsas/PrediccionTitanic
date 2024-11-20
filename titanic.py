import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configurar warnings
import warnings
warnings.filterwarnings('ignore')

# Cargar los datos
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Unificar ambos conjuntos de datos para preprocesarlos juntos
combined_data = pd.concat([train_data, test_data], sort=False).copy()

# Corrección del manejo de valores faltantes
combined_data = combined_data.assign(
    Age=combined_data['Age'].fillna(combined_data['Age'].median()),
    Fare=combined_data['Fare'].fillna(combined_data['Fare'].median()),
    Embarked=combined_data['Embarked'].fillna(combined_data['Embarked'].mode()[0])
)

# Codificación de variables
combined_data['Sex'] = combined_data['Sex'].map({'male': 1, 'female': 0})
combined_data = pd.get_dummies(combined_data, columns=['Embarked'], drop_first=True)

# Crear características adicionales
combined_data['IsMother'] = ((combined_data['Sex'] == 0) & 
                            (combined_data['Parch'] > 0) & 
                            (combined_data['Age'] > 18)).astype(int)

combined_data['Title'] = combined_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
combined_data['IsMrs'] = (combined_data['Title'] == 'Mrs').astype(int)

# Separar los datos preprocesados
train_data = combined_data[combined_data['Survived'].notna()].copy()
test_data = combined_data[combined_data['Survived'].isna()].copy()

passenger_dict = train_data.set_index('PassengerId').to_dict('index')

# Seleccionar características
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
           'Embarked_Q', 'Embarked_S', 'IsMother', 'IsMrs']

X_train = train_data[features].values
y_train = train_data['Survived'].values
X_test = test_data[features].values

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Separar datos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Ajustar callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Ajustar arquitectura del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_shape=(X_train.shape[1],), 
                          activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),  # Aumentar dropout para evitar sobreajuste
    tf.keras.layers.Dense(16, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar modelo
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


def predict_survival_by_id(passenger_id):
    try:
        # Verificar si el ID existe
        if passenger_id not in passenger_dict:
            return f"No se encontró ningún pasajero con el ID {passenger_id}"
 
        passenger = passenger_dict[passenger_id]
        features_list = [passenger[feat] for feat in features]

        scaled_features = scaler.transform([features_list])

        prediction = model.predict(scaled_features, verbose=0)
        probability = prediction[0][0]

        real_outcome = "sobrevivió" if passenger['Survived'] == 1 else "no sobrevivió"
        predicted_outcome = "sobrevivió" if probability > 0.5 else "no sobrevivió"

        mensaje = f"\nPasajero ID: {passenger_id}"
        mensaje += f"\nNombre: {passenger['Name']}"
        mensaje += f"\nClase: {passenger['Pclass']}"
        mensaje += f"\nEdad: {passenger['Age']:.0f}"
        mensaje += f"\nSexo: {'Mujer' if passenger['Sex'] == 0 else 'Hombre'}"
        mensaje += f"\nTarifa pagada: ${passenger['Fare']:.2f}"
        mensaje += f"\n\nPredicción: {predicted_outcome.upper()} (Probabilidad: {probability:.2f})"
        mensaje += f"\nResultado real: {real_outcome.upper()}"
        
        return mensaje
    
    except Exception as e:
        return f"Error al procesar la predicción: {str(e)}"

def main():
    print("\n=== Predictor de Supervivencia del Titanic ===")
    print("Ingrese el ID del pasajero (1-891) o 'q' para salir")
    
    while True:
        entrada = input("\nID del pasajero: ")
        
        if entrada.lower() == 'q':
            break
            
        try:
            passenger_id = int(entrada)
            if 1 <= passenger_id <= 891:
                resultado = predict_survival_by_id(passenger_id)
                print(resultado)
            else:
                print("Por favor ingrese un ID válido entre 1 y 891")
        except ValueError:
            print("Por favor ingrese un número válido o 'q' para salir")

if __name__ == "__main__":
    loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
    print(f'\nPrecisión del modelo en conjunto de validación: {accuracy:.4f}')

    main()