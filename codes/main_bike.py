import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score)
from src.visualizations import (relations_with_target,
                                partial_dependence_plot)
from src.eda_module import (drop_spaces_data, replace_empty_nans,
                            convert_df_float)


# bicicletas por hora rentadas
df = pd.read_csv("data/bike/hour.csv")
df = df.drop(columns=['dteday', 'instant'])
print("Las columnas restantes son:", list(df.columns))

# sacar vaccios
df = drop_spaces_data(df)
# hacer identificables los vacios
df = replace_empty_nans(df)
# dropna
df.dropna(inplace=True)
# convertir a flotante
df = convert_df_float(df)


# visualizar la dependecia con el target de cada una de las variables
relations_with_target(df, target="cnt")


# sacar la data
data = df.drop(['cnt'], axis=1)
y = df['cnt']
scaler = MinMaxScaler(0, 1)
x = scaler.fit_transform(data)

# dividir la data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

# hacer un regresor como arbol de decisión
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train, y_train)
predictions = regressor.predict(x_test)

mse = mean_squared_error(y_test, predictions)
r = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print("Mean Squared Error:", mse)
print("R score:", r)
print("Mean Absolute Error:", mae)

# partial dependency plot de las predicciones del modelo entrenado
partial_dependence_plot(df, predictions, target="cnt")
