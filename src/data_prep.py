# Limpieza final

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

customer = pd.read_csv("C:\BD_ANN\Ejercicios ANN-20260601\customer_numerico.csv", sep=',')
customer.head(5)

# Eliminamos la columna customerID ya que es un identificador sin valor predictivo
customer.drop(columns=['customerID'], inplace=True)

# Se reemplaza los "No phone service" y "No Internet service" porque semanticamente significan lo mismo que "No"
customer = customer.replace({
    'No internet service': 'No',
    'No phone service': 'No'
})

customer['OnlineSecurity'].unique()

# Identificar variables categoricas
cat_cols = customer.select_dtypes(include=['object']).columns.tolist()
print("Variables categoricas:", cat_cols)

# Aquí se convierten las variables categoricas en numericas. Se hace solo en esas tres columnas porque tienen más de dos variables
customer = pd.get_dummies(customer,
                    columns=['Contract', 'PaymentMethod', 'InternetService'])

customer.head(5)

# Se usa for para no repetir todo el código e ir columna por columna
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'MultipleLines', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'Churn', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
               'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
               'PaymentMethod_Mailed check', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No']

for col in binary_cols:
    customer[col] = le.fit_transform(customer[col])

customer.head(5)

# Verificar correlacion de todas las variables con Churn
customer.corr(numeric_only=True)['Churn'].sort_values()

# Grafico de correlacion con Churn
customer.corr(numeric_only=True)['Churn'][:-1].sort_values().plot(kind='barh', figsize=(10, 10))
plt.title('Correlacion de Variables con Churn')
plt.xlabel('Correlacion')
plt.tight_layout()
plt.show()

# Se guarda el csv limpio
customer.to_csv('customer_clean.csv', index=False)

# 2. Train Test Split

X = customer.drop('Churn', axis=1).values
y = customer['Churn'].values

X.shape

customer.head()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101, stratify=y)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('Y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('Y_test.csv', index=False)

# Escalado

# StandardScaler: transforma los datos para que tengan media 0 y desviacion estandar 1
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

print("Media X_train:", X_train.mean().round(4))
print("Desv. Estandar X_train:", X_train.std().round(4))

# Manejo de datos desbalanceados

# Calcular los pesos por clase de forma automatica
classes = np.array([0, 1])
class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {0: class_weights_array[0], 1: class_weights_array[1]}
print("Class weights:", class_weight_dict)