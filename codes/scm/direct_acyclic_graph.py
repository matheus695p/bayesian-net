import pandas as pd
from IPython.display import Image
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

path = "data/mine/ads_combined.pkl"
df = pd.read_pickle(path)
df.dropna(inplace=True)

# esto viene de una matriz de correlaciones
path_corr = "data/mine/correlations_presentation.xlsx"
corr = pd.read_excel(path_corr, engine="openpyxl", sheet_name="Hipotesis")

# puntos del grafo
edges = []
for i in range(len(corr)):
    print(i)
    array = (corr["col1"].iloc[i], corr["col2"].iloc[i])
    edges.append(array)

# crear modelo
sm = StructureModel()
sm.add_edges_from(edges)

print(sm.edges)
viz = plot_structure(
    sm,
    graph_attributes={"scale": "3"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
Image(viz.draw(format='png'))
viz.draw("results/graphs/correlations.png")
