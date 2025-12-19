import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

n_samples = 100

temperatura = np.random.uniform(15, 35, n_samples)
carga_producao = np.random.uniform(30, 100, n_samples)

# Relação linear com ruído
consumo_energia = (
    2.5 * temperatura +
    1.8 * carga_producao +
    np.random.normal(0, 10, n_samples)
)

df = pd.DataFrame({
    'temperatura': temperatura,
    'carga_producao': carga_producao,
    'consumo_energia': consumo_energia
})

print(df.head())

X = df[['temperatura', 'carga_producao']]
y = df['consumo_energia']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Consumo real (kWh)")
plt.ylabel("Consumo previsto (kWh)")
plt.title("Consumo real vs previsto")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.show()
