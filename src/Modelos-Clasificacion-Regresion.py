

# Sistema de Predicción de Churn - Telco Customer
# Entrenamiento de modelos ANN: Clasificación y Regresión

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam as KerasAdam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# 1. Cargar datos

df = pd.read_csv(r"C:\Users\Jesus\Downloads\ANN Regresión-20260529\customer_clean.csv")

X = df.drop('Churn', axis=1)
y = df['Churn']

print("Shape X:", X.shape)
print("Distribución Churn:", y.value_counts())


# 2. SMOTE

smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
print("Distribución después de SMOTE:", np.bincount(y_res.astype(int)))

# 3. Normalización

scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)


# 4. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_res_scaled, y_res, test_size=0.2, random_state=42
)
print("Distribución y_train:", np.bincount(y_train.astype(int)))
print("Distribución y_test:", np.bincount(y_test.astype(int)))


# 5. Modelo 1 - Clasificación Binaria

print("\nEntrenando Modelo 1 - Clasificación Binaria...")

model_class = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(), Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_class.compile(
    optimizer=KerasAdam(learning_rate=0.0003),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop_class = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
reduce_lr_class  = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, min_lr=1e-6)

history_class = model_class.fit(
    X_train, y_train,
    epochs=400, batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop_class, reduce_lr_class],
    verbose=1
)

loss_class, acc_class = model_class.evaluate(X_test, y_test, verbose=0)
print(f"Modelo Clasificación - Accuracy: {acc_class:.2f}")


# 6. Modelo 2 - Regresión

print("\nEntrenando Modelo 2 - Regresión...")

model_reg = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(512, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(), Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_reg.compile(
    optimizer=KerasAdam(learning_rate=0.0005),
    loss='huber',
    metrics=['mae']
)

early_stop_reg = EarlyStopping(monitor='val_mae', patience=20, restore_best_weights=True)
reduce_lr_reg  = ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=7, min_lr=1e-6)

history_reg = model_reg.fit(
    X_train, y_train,
    epochs=300, batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop_reg, reduce_lr_reg],
    verbose=1
)

loss_reg, mae_reg = model_reg.evaluate(X_test, y_test, verbose=0)
print(f"Modelo Regresión - MAE: {mae_reg:.2f}")


# 7. Comparación de Modelos

print("\n" + "="*50)
print("     COMPARACIÓN DE MODELOS")
print("="*50)
print(f"Modelo 1 - Clasificación Binaria:")
print(f"  Accuracy: {acc_class:.2f} (mínimo requerido: 0.84) ")
print(f"Modelo 2 - Regresión:")
print(f"  MAE: {mae_reg:.2f} (mínimo requerido: 0.22) ")

# Gráfica comparativa
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(['Clasificación'], [acc_class], color='#4A90D9')
axes[0].axhline(y=0.84, color='red', linestyle='--', label='Mínimo requerido')
axes[0].set_title('Modelo 1 - Accuracy')
axes[0].set_ylim(0, 1)
axes[0].legend()

axes[1].bar(['Regresión'], [mae_reg], color='#2ecc71')
axes[1].axhline(y=0.22, color='red', linestyle='--', label='Máximo permitido')
axes[1].set_title('Modelo 2 - MAE')
axes[1].set_ylim(0, 0.5)
axes[1].legend()

plt.suptitle('Comparación de Modelos ANN - Telco Churn')
plt.tight_layout()
plt.show()


# 8. Exportar modelos y scaler

model_class.save("model_churn_class.keras")
model_reg.save("model_churn_reg.keras")
pickle.dump(scaler, open(r"C:\Users\Jesus\Downloads\ANN Regresión-20260529\scaler.pkl", "wb"))
print("\nModelos y scaler guardados ")