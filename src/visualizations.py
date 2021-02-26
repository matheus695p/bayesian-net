import pandas as pd
import matplotlib.pyplot as plt


def relations_with_target(df, target="alpha"):
    """
    Visualizar las correlaciones de las características con la variable
    Cantidad
    Parameters
    ----------
    df : DataFrame
        dataframe con columnas
    Returns
    -------
    None.
    """
    for columna in df.columns:
        x1 = df[target]
        x2 = df[columna]
        fig, ax = plt.subplots(1, figsize=(22, 12))

        # qi = 0.1
        # qf = 0.9
        # min_x = x1.quantile(qi)
        # max_x = x1.quantile(qf)
        # min_y = x2.quantile(qi)
        # max_y = x2.quantile(qf)

        plt.scatter(x1, x2, color='orangered')
        titulo = f'Relación de con la variable objetivo de: {columna}'
        plt.title(titulo, fontsize=30)
        plt.xlabel(target, fontsize=30)
        plt.ylabel(columna, fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.set_facecolor('black')
        plt.legend([columna], fontsize=22,
                   loc="upper left")
        # plt.ylim(min_y, max_y)
        # plt.xlim(min_x, max_x)
        plt.show()


def partial_dependence_plot(df, predictions, target="alpha"):
    """
    Visualizar un gráfico de dependencia de las predicciones vs las variables

    Parameters
    ----------
    df : dataframe
        dataframe sin tocar.
    predictions : array
        array con las predicciones del modelo.
    target : string
        nombre de la columna a predecir.
    Returns
    -------
    PDP del modelo recién entrenado
    """

    predictions = pd.DataFrame(predictions, columns=["predicciones"])
    df.drop(columns=[target], inplace=True)
    for columna in df.columns:
        x1 = predictions["predicciones"]
        x2 = df[columna]
        fig, ax = plt.subplots(1, figsize=(22, 12))

        # qi = 0.1
        # qf = 0.9
        # min_x = x1.quantile(qi)
        # max_x = x1.quantile(qf)
        # min_y = x2.quantile(qi)
        # max_y = x2.quantile(qf)

        plt.scatter(x1, x2, color='orangered')
        titulo = f'Análisis de colinealidades variable: {columna}'
        plt.title(titulo, fontsize=30)
        plt.xlabel(target, fontsize=30)
        plt.ylabel(columna, fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.set_facecolor('black')
        plt.legend([columna], fontsize=22,
                   loc="upper left")
        # plt.ylim(min_y, max_y)
        # plt.xlim(min_x, max_x)
        plt.show()
