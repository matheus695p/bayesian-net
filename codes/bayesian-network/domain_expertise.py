import warnings
import pandas as pd
from IPython.display import Image
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.network import BayesianNetwork
from sklearn.model_selection import train_test_split
from causalnex.evaluation import (classification_report, roc_auc)
from causalnex.inference import InferenceEngine
from src.preprocessing.discretizer import (discretize_minmax)
warnings.filterwarnings("ignore")


# cargar los datos
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
# cambio de nombre de algunas columnas
df.rename(columns={"tiempo_cola": "tiempo_cola_carga"}, inplace=True)
selected_cols = ["num_equipos_prod", "tiempo_cola_carga",
                 "promedio_t_detencion_descarga",
                 "promedio_t_detencion_traslado_ida",
                 "promedio_t_detencion_traslado_vuelta",
                 "n_eventos_stopped_ida",
                 "n_eventos_stopped_vuelta",
                 "n_bulldozer", "n_excavadoras", "n_rodillos",
                 "num_total_ciclos"]
df = df[selected_cols]

# verificar que no hayan nans
print(df.isna().sum())
alpha = df.isna().sum()
# discretizacion de los valores numericos
discretised_data = df.copy()
# discretizacion de los valores en el dataframe
order_columns = []
n_labels = 100
for col in cols:
    print("Discretizando la columna", col)
    # normalizacion del las columnas
    new_col = "bining_" + col
    order_columns.append(col)
    order_columns.append(new_col)


# discretised_data = discretize(df, target_name='num_total_ciclos', n_bins=5)

discretised_data, rangos = discretize_minmax(df,
                                             target_name='num_total_ciclos',
                                             n_bins=4)

edges_graph = [
    ("num_equipos_prod", "tiempo_cola_carga"),
    ("num_equipos_prod", "promedio_t_detencion_descarga"),
    ("num_equipos_prod", "promedio_t_detencion_traslado_ida"),
    ("num_equipos_prod",
        "promedio_t_detencion_traslado_vuelta"),
    ("num_equipos_prod", "n_eventos_stopped_ida"),
    ("num_equipos_prod", "n_eventos_stopped_vuelta"),

    ("n_bulldozer", "tiempo_cola_carga"),
    ("n_bulldozer", "promedio_t_detencion_descarga"),
    ("n_bulldozer", "promedio_t_detencion_traslado_ida"),
    ("n_bulldozer", "promedio_t_detencion_traslado_vuelta"),
    ("n_bulldozer", "n_eventos_stopped_ida"),
    ("n_bulldozer", "n_eventos_stopped_vuelta"),

    ("n_rodillos", "tiempo_cola_carga"),
    ("n_rodillos", "promedio_t_detencion_descarga"),
    ("n_rodillos", "promedio_t_detencion_traslado_ida"),
    ("n_rodillos", "promedio_t_detencion_traslado_vuelta"),
    ("n_rodillos", "n_eventos_stopped_ida"),
    ("n_rodillos", "n_eventos_stopped_vuelta"),

    ("n_excavadoras", "tiempo_cola_carga"),
    ("n_excavadoras", "promedio_t_detencion_descarga"),
    ("n_excavadoras", "promedio_t_detencion_traslado_ida"),
    ("n_excavadoras", "promedio_t_detencion_traslado_vuelta"),
    ("n_excavadoras", "n_eventos_stopped_ida"),
    ("n_excavadoras", "n_eventos_stopped_vuelta"),

    ("n_eventos_stopped_ida", "promedio_t_detencion_traslado_ida"),
    ("n_eventos_stopped_vuelta", "promedio_t_detencion_traslado_vuelta"),

    ("n_rodillos", "tiempo_cola_carga"),
    ("n_rodillos", "promedio_t_detencion_descarga"),
    ("n_rodillos", "promedio_t_detencion_traslado_ida"),
    ("n_rodillos", "promedio_t_detencion_traslado_vuelta"),
    ("n_rodillos", "n_eventos_stopped_ida"),
    ("n_rodillos", "n_eventos_stopped_vuelta"),

    ("tiempo_cola_carga", "num_total_ciclos"),
    ("promedio_t_detencion_descarga", "num_total_ciclos"),
    ("promedio_t_detencion_traslado_ida", "num_total_ciclos"),
    ("promedio_t_detencion_traslado_vuelta",
        "num_total_ciclos"),
    ("n_eventos_stopped_ida", "num_total_ciclos"),
    ("n_eventos_stopped_vuelta", "num_total_ciclos")]


sm = StructureModel()
sm.add_edges_from(edges_graph)

viz = plot_structure(
    sm,
    graph_attributes={"scale": "3"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
Image(viz.draw(format='png'))
viz.draw("results/graphs/funcionamiento_logico.png")


# construir la red bayesiana
bn = BayesianNetwork(sm)
# división del conjunto de datos
train, test = train_test_split(discretised_data, train_size=0.9, test_size=0.1,
                               random_state=21)
# fit del modelo
bn = bn.fit_node_states(discretised_data)
bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")

s1 = bn.cpds["num_total_ciclos"]
s1.reset_index(drop=False, inplace=True)
# score del modelo
classification_report(bn, test, "num_total_ciclos")
# score del modelo
roc, auc = roc_auc(bn, test, "num_total_ciclos")
print("Acurracy del modelo: ",  round(auc * 100, 3), "[%]")

# counterfactuals
ie = InferenceEngine(bn)

# ¿Como afecta el numero de ciclos si hay menos numero de camiones?
# P(numero de ciclos| num_equipos_prod=1)
ie.do_intervention(
    "num_equipos_prod", {1: 1, 2: 0, 3: 0, 4: 0, 5: 0})
num_equipos_prod_1 = pd.Series(ie.query()["num_total_ciclos"])

# ¿Como afecta el numero de ciclos si hay muchos numero de camiones?
# P(numero de ciclos| num_equipos_prod=5)
ie.do_intervention(
    "num_equipos_prod", {1: 0, 2: 0, 3: 0, 4: 0, 5: 1})
num_equipos_prod_5 = pd.Series(ie.query()["num_total_ciclos"])
# ¿Como afecta el numero de ciclos si hay menos numero de bulldozers?
# P(numero de ciclos| n_bulldozer=1)
ie.do_intervention(
    "n_bulldozer", {1: 1, 2: 0, 3: 0, 4: 0, 5: 0})
n_bulldozer_1 = pd.Series(ie.query()["num_total_ciclos"])

# ¿Como afecta el numero de ciclos si hay muchos numero de camiones?
# P(numero de ciclos| n_bulldozer=5)
ie.do_intervention(
    "n_bulldozer", {1: 0, 2: 0, 3: 0, 4: 0, 5: 1})
n_bulldozer_5 = pd.Series(ie.query()["num_total_ciclos"])

# ¿Como afecta el numero de ciclos si hay menos numero de excavadoras?
# P(numero de ciclos| n_excavadoras=1)
ie.do_intervention(
    "n_excavadoras", {1: 1, 2: 0, 3: 0, 4: 0, 5: 0})
n_excavadoras_1 = pd.Series(ie.query()["num_total_ciclos"])

# ¿Como afecta el numero de ciclos si hay muchos numero de excavadoras?
# P(numero de ciclos| n_excavadoras=5)
ie.do_intervention(
    "n_excavadoras", {1: 0, 2: 0, 3: 0, 4: 0, 5: 1})
n_excavadoras_5 = pd.Series(ie.query()["num_total_ciclos"])

# ¿Como afecta el numero de ciclos si hay menos numero de rodillos?
# P(numero de ciclos| n_rodillos=1)
ie.do_intervention(
    "n_rodillos", {1: 1, 2: 0, 3: 0, 4: 0, 5: 0})
n_rodillos_1 = pd.Series(ie.query()["num_total_ciclos"])

# ¿Como afecta el numero de ciclos si hay muchos numero de rodillos?
# P(numero de ciclos| n_rodillos=5)
ie.do_intervention(
    "n_rodillos", {1: 0, 2: 0, 3: 0, 4: 0, 5: 1})
n_rodillos_5 = pd.Series(ie.query()["num_total_ciclos"])

# ¿Como afecta el numero de ciclos si los eventos de cercania son elevados?
# P(numero de ciclos| n_eventos_stopped_ida=5)
ie.do_intervention(
    "n_eventos_stopped_ida", {1: 0, 2: 0, 3: 0, 4: 0, 5: 1})
n_eventos_stopped_ida_5 = pd.Series(ie.query()["num_total_ciclos"])

# ¿Como afecta el numero de ciclos si los eventos de cercania son bajos?
# P(numero de ciclos| n_eventos_stopped_ida=1)
ie.do_intervention(
    "n_eventos_stopped_ida", {1: 1, 2: 0, 3: 0, 4: 0, 5: 0})
n_eventos_stopped_ida_1 = pd.Series(ie.query()["num_total_ciclos"])


couterfactuals_ciclos = pd.concat([n_eventos_stopped_ida_1,
                                   n_eventos_stopped_ida_5,
                                   num_equipos_prod_1,
                                   num_equipos_prod_5,
                                   n_bulldozer_1,
                                   n_bulldozer_5,
                                   n_excavadoras_1,
                                   n_excavadoras_5,
                                   n_rodillos_1,
                                   n_rodillos_5], axis=1)
couterfactuals_ciclos.columns = ["n_eventos_stopped_ida_1",
                                 "n_eventos_stopped_ida_5",
                                 "num_equipos_prod_1",
                                 "num_equipos_prod_5",
                                 "n_bulldozer_1",
                                 "n_bulldozer_5",
                                 "n_excavadoras_1",
                                 "n_excavadoras_5",
                                 "n_rodillos_1",
                                 "n_rodillos_5"]

couterfactuals_ciclos.to_excel("data/counter.xlsx")
