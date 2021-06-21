import numpy as np
import pandas as pd


def discretize(df, target_name='num_total_ciclos', n_bins=8):
    """
    Discretización en bins de una variable continua para el entrenamiento de
    la red bayesiana

    Parameters
    ----------
    df : pandas.dataframe
        DESCRIPTION.
    n_bins : TYPE, optional
        DESCRIPTION. The default is 8.

    Returns
    -------
    continuous : TYPE
        DESCRIPTION.

    """
    target = df[[target_name]]
    cols = list(df.columns)
    cols.remove(target_name)
    continuous = df[cols]

    # target = (
    #     target
    #     .to_frame()
    #     .assign(recovery_bin=lambda x: pd.qcut(x[target_name],
    #                                            n_bins, duplicates='drop'))
    #     .groupby(target_name+"_bin").transform(lambda x: x.mean())
    #     .apply(np.log1p)
    # )

    target = (
        target
        .apply(lambda x: pd.qcut(x, n_bins, duplicates='drop'))
        .apply(lambda x: [e.mid for e in x])
    )

    continuous = (
        continuous
        .apply(lambda x: pd.qcut(x, n_bins, duplicates='drop'))
        .apply(lambda x: [e.mid for e in x])
    )
    return pd.concat([continuous, target], axis=1)


def discretize_minmax(df, target_name='num_total_ciclos', n_bins=4):
    target = df[[target_name]]
    cols = list(df.columns)
    cols.remove(target_name)
    continuous = df[cols]
    rangos = []
    for col in continuous.columns:
        minimum = continuous[col].min()
        maximum = continuous[col].max()
        continuous[col] = (continuous[col] - minimum) / (maximum - minimum)
        continuous[col] = (continuous[col] * n_bins).apply(int) + 1
        rangos.append([col, minimum, maximum])

    for col in target.columns:
        minimum = target[col].min()
        maximum = target[col].max()
        target[col] = (target[col] - minimum) / (maximum - minimum)
        target[col] = (target[col] * n_bins).apply(int) + 1
        rangos.append([col, minimum, maximum])

    rangos = pd.DataFrame(rangos, columns=["columna", "minimo", "maximo"])
    rangos["diferencia"] = rangos["maximo"] - rangos["minimo"]
    rangos["delta"] = rangos["diferencia"] / n_bins
    for i in range(0, n_bins+1):
        print(i)
        rangos[f"q_{str(i+1)}"] = rangos["delta"] * i + rangos["minimo"]
    for col in rangos.columns:
        try:
            rangos[col] = rangos[col].apply(lambda x: round(x, 2))
        except Exception as e:
            print(e)
            pass
    return pd.concat([continuous, target], axis=1), rangos
