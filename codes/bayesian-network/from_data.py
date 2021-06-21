# import numpy as np
import pandas as pd
from IPython.display import Image
from datetime import datetime
from causalnex.structure.notears import from_pandas
from causalnex.discretiser import Discretiser
from causalnex.network import BayesianNetwork
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from sklearn.model_selection import train_test_split

fecha = datetime.now()
path = "data/mine/ads_combined.pkl"
df = pd.read_pickle(path)
df.dropna(inplace=True)
# filtro por tiempo efectivo
df = df[(df["tiempo_efectivo"] > 0) & (df["factor_operacional"] > 0) &
        (df["produccion_por_hora"] > 50) & (df["n_eventos_ida"] > 100) &
        (df["n_eventos_vuelta"] > 100)]
df.reset_index(drop=True, inplace=True)

# obtener solo las columnas numericas
numerical_columns = []
for col in df.columns:
    type_ = str(df[[col]].dtypes[0])
    if (type_ == "int64") | (type_ == "float64"):
        numerical_columns.append(col)
df = df[numerical_columns]

# seleccionar solo las columnas de interes para hacer el análisis
cols = ['tiempo_operativo', 'tiempo_sin_uso', 'tiempo_efectivo',
        'uso_operativo', 'disponibilidad_mecanica', 'disponibilidad_fisica',
        'factor_operacional', 'utilizacion_efectiva', 'utilizacion',
        'rendimiento_operativo', 'rendimiento_efectivo',
        'rendimiento_m3_h_km', 'promedio_velocidad_promedio',
        'promedio_tiempo_ciclo', 'promedio_tiempo_casino',
        'promedio_tiempo_taller', 'promedio_distancia_recorrida',
        'promedio_combustible_gastado', 'promedio_rendimiento_combustible',
        'promedio_t_carga', 'promedio_t_detencion_carga',
        'promedio_t_descarga', 'promedio_t_detencion_descarga',
        'promedio_t_traslado_ida', 'promedio_t_detencion_traslado_ida',
        'promedio_t_traslado_vuelta', 'promedio_t_detencion_traslado_vuelta',
        'promedio_m3_transportados', 'num_puntos_descarga',
        'num_total_ciclos', 'num_vueltas_casino', 'num_vueltas_taller',
        'num_equipos_gps', 'num_equipos_canbus', 'num_equipos_prod',
        'num_equipos_disponibles', 'num_equipos_no_utilizados',
        'nivel_umbral_repostaje', 'nivel_de_repostaje',
        'tiempo_encendido_hor', 'tiempo_apagado_hor',
        'num_repostajes_realizados', 'promedio_t_traslado',
        'perdida_inicio_turno_promedio', 'perdida_fin_turno_promedio',
        'perdida_inicio_turno', 'perdida_final_turno', 'dia_de_turno',
        'n_bulldozer', 'n_rodillos', 'n_excavadoras',
        'min_distancia_excavadora', 'tiempo_efectivo_carga', 'tiempo_cola',
        'tiempo_carga', 'dist_colas', 'densidad_camiones_por_excavadora',
        'mov_camion', 'n_eventos_ida', 'n_eventos_vuelta',
        'n_eventos_stopped_ida', 'n_eventos_stopped_vuelta',
        'n_total_eventos_cercania', 'n_eventos_ida_moving',
        'n_eventos_vuelta_moving', 'n_eventos_stopped_ida_moving',
        'n_eventos_stopped_vuelta_moving', 'n_total_eventos_cercania_moving',
        'tiempo_repostaje', 'cantidad_reposteos']
df = df[cols]

# discretizacion de los valores numericos
discretised_data = df.copy()
# discretizacion de los valores en el dataframe
order_columns = []
for col in df.columns:
    print("Discretizando la columna", col)
    # normalizacion del las columnas
    new_col = "bining_" + col
    order_columns.append(col)
    order_columns.append(new_col)
    discretised_data[new_col] = Discretiser(
        method="fixed",
        numeric_split_points=[1, 20]).transform(discretised_data[col].values)
    # borrar las columnas originales
    discretised_data.drop(columns=[col], inplace=True)

# quiero que aprenda todo lo que quiera desde los datos
# Unconstrained learning todo se aprende desde los datos
sm = from_pandas(discretised_data)
viz = plot_structure(
    sm,
    graph_attributes={"scale": "5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
Image(viz.draw(format='png'))
viz.draw("results/graphs/test.png")
viz.draw("results/graphs/from_data11.png")
print("Esto tomo:", (datetime.now() - fecha).total_seconds() / 60, "minutos")

# obtener el más grande subgrafo
filter_sm = sm.remove_edges_below_threshold(0.5)
viz = plot_structure(
    filter_sm,
    graph_attributes={"scale": "5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
Image(viz.draw(format='png'))
viz.draw("results/graphs/from_data22.png")

# filtrar
sub_sm = sm.get_largest_subgraph()
viz = plot_structure(
    sub_sm,
    graph_attributes={"scale": "5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
Image(viz.draw(format='png'))
viz.draw("results/graphs/from_data33.png")

# poner un threshold a la conexion entre los nodos

# red
bn = BayesianNetwork(sub_sm)
# división del conjunto de datos
train, test = train_test_split(discretised_data, train_size=0.9,
                               test_size=0.1, random_state=21)
# fit del modelo
bn = bn.fit_node_states(discretised_data)
bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")

s1 = bn.cpds["bining_promedio_t_descarga"]
s2 = bn.cpds["bining_promedio_t_traslado_ida"]
