import pandas as pd
from src.utils.visualizations import counterfactuals_barplot

df = pd.read_excel("data/counterfactuals.xlsx",
                   engine="openpyxl", sheet_name="Hoja1")


counterfactuals_barplot(df, col="n_eventos_stopped", letter_size=40)
counterfactuals_barplot(df, col="num_equipos_prod", letter_size=40)
counterfactuals_barplot(df, col="n_bulldozer", letter_size=40)
counterfactuals_barplot(df, col="n_excavadoras", letter_size=40)
counterfactuals_barplot(df, col="n_rodillos", letter_size=40)
