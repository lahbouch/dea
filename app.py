# dea_dashboard.py
# Tableau de bord Streamlit DEA pour les ratios ESG et financiers
# Exécuter : streamlit run dea_dashboard.py

import os
import re
import unicodedata
from io import BytesIO
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, lpSum, LpStatus
import time  # Pour estimer le temps

st.set_page_config(page_title="Tableau de bord DEA ESG-Finance", layout="wide")

# ------------------ Utilitaires ------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = s.replace(":", "").replace("-", "").replace("/", "")
    return s

def to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x)
    s = s.replace("\u00A0", "").replace("−", "-").replace("%", "").replace(",", ".").strip()
    if s in ("", "-", "nan", "NaN", "none", "None"): return np.nan
    try: return float(s)
    except: return np.nan

def find_col_norm(df, keywords):
    nmap = {_norm(c): c for c in df.columns}
    for kw in keywords:
        for norm_name, orig in nmap.items():
            if kw in norm_name:
                return orig
    return None

# ------------------ Chargeur de données ------------------
@st.cache_data
def load_esg_data(file_path):
    df = pd.read_excel(file_path, dtype=object)
    df.columns = [str(c) for c in df.columns]
    orig_cols = list(df.columns)

    col_ticker = find_col_norm(df, ["symbol", "ticker", "entreprises"]) or df.columns[0]
    col_year = find_col_norm(df, ["year", "annee"])
    col_month = find_col_norm(df, ["month", "mois"])
    col_e = find_col_norm(df, ["score e", "environmental"])
    col_s = find_col_norm(df, ["score s", "social"])
    col_g = find_col_norm(df, ["score g", "governance"])
    col_total = find_col_norm(df, ["score total", "total"])

    df[col_ticker] = df[col_ticker].ffill()
    df["_Year"] = df[col_year].apply(lambda x: to_float(x))
    df = df[~df["_Year"].isna()].copy()

    for c in [col_e, col_s, col_g, col_total]:
        if c: df[c] = df[c].apply(to_float)

    # Imputer les valeurs manquantes avec la moyenne annuelle par ticker
    for c in [col_e, col_s, col_g, col_total]:
        if c:
            df[c] = df.groupby([col_ticker, "_Year"])[c].transform(lambda x: x.fillna(x.mean()))

    # Agréger en moyennes annuelles
    df = df.groupby([col_ticker, "_Year"])[[col_e, col_s, col_g, col_total]].mean().reset_index()
    cols = {"Ticker": col_ticker, "Year": "_Year"}
    if col_e: cols["E"] = col_e
    if col_s: cols["S"] = col_s
    if col_g: cols["G"] = col_g
    if col_total: cols["Total"] = col_total

    tidy = df[list(cols.values())].copy()
    tidy.columns = list(cols.keys())
    for c in ["E", "S", "G", "Total"]:
        if c not in tidy.columns: tidy[c] = np.nan

    tidy = tidy.dropna(subset=["Ticker", "Year"]).sort_values(["Ticker", "Year"]).reset_index(drop=True)
    return tidy, {"Ticker": col_ticker, "Year": col_year, "E": col_e, "S": col_s, "G": col_g, "Total": col_total}

@st.cache_data
def load_financial_data(file_path):
    df = pd.read_excel(file_path, dtype=object)
    df.columns = [str(c) for c in df.columns]
    orig_cols = list(df.columns)

    col_ticker = find_col_norm(df, ["ticker", "symbol"]) or df.columns[0]
    col_year = find_col_norm(df, ["year", "annee"])
    col_roe = find_col_norm(df, ["roe", "return on equity"])
    col_roa = find_col_norm(df, ["roa", "return on assets"])
    col_pm = find_col_norm(df, ["profit margin", "margin"])
    col_pe = find_col_norm(df, ["pe", "price/earnings"])
    col_pb = find_col_norm(df, ["pb", "price/book"])

    df[col_ticker] = df[col_ticker].ffill()
    df["_Year"] = df[col_year].apply(to_float)
    df = df[~df["_Year"].isna()].copy()

    for c in [col_roe, col_roa, col_pm, col_pe, col_pb]:
        if c: df[c] = df[c].apply(to_float)

    cols = {"Ticker": col_ticker, "Year": "_Year"}
    if col_roe: cols["ROE"] = col_roe
    if col_roa: cols["ROA"] = col_roa
    if col_pm: cols["Profit Margin"] = col_pm
    if col_pe: cols["P/E"] = col_pe
    if col_pb: cols["P/B"] = col_pb

    tidy = df[list(cols.values())].copy()
    tidy.columns = list(cols.keys())
    for c in ["ROE", "ROA", "Profit Margin", "P/E", "P/B"]:
        if c not in tidy.columns: tidy[c] = np.nan

    tidy = tidy.dropna(subset=["Ticker", "Year"]).sort_values(["Ticker", "Year"]).reset_index(drop=True)
    return tidy, {"Ticker": col_ticker, "Year": col_year, "ROE": col_roe, "ROA": col_roa, "Profit Margin": col_pm, "P/E": col_pe, "P/B": col_pb}

# ------------------ Analyse DEA ------------------
def run_dea(df, model_type="CRS", orientation="output", yearly=True):
    inputs = ["E", "S", "G", "Total"]
    outputs = ["ROE", "ROA", "Profit Margin", "P/E", "P/B"]
    df = df.dropna(subset=inputs + outputs)

    # Inverser les scores ESG (plus élevé est meilleur, donc inverser pour la minimisation des intrants)
    for col in inputs:
        df[col] = 100 - df[col]

    # Décaler les sorties pour garantir des valeurs positives
    for col in outputs:
        df[col] = df[col] - df[col].min() + 1

    results = []
    if yearly:
        years = df["Year"].unique()
        if len(years) <= 1:
            st.warning("Une seule année détectée. La barre de progression n'est pas applicable. Exécution de la DEA sur toutes les données.")
            scores = dea_ccr(df, inputs, outputs, model_type, orientation)
            df_all = df[["Ticker", "Year", "Index", "Total", "ROE"]].copy()
            df_all["Efficiency"] = scores
            df_all["Model"] = model_type
            df_all["Yearly"] = True
            results.append(df_all)
        else:
            for i, year in enumerate(years):
                st.write(f"Début de la DEA pour l'année {year}")  # Message de débogage
                df_year = df[df["Year"] == year]
                status_text = st.empty()
                progress_bar = st.progress(0)
                start_time = time.time()
                avg_time_per_dmu = 0.1  # Estimation initiale en secondes par DMU
                scores = dea_ccr(df_year, inputs, outputs, model_type, orientation)
                df_year = df_year[["Ticker", "Year", "Index", "Total", "ROE"]].copy()
                df_year["Efficiency"] = scores
                df_year["Model"] = model_type
                df_year["Yearly"] = True
                results.append(df_year)
                progress_bar.progress((i + 1) / len(years))
                time_per_year = time.time() - start_time
                remaining_years = len(years) - (i + 1)
                estimated_time = remaining_years * time_per_year / 60  # en minutes
                status_text.text(f"Traitement de l'année {year}: {((i + 1) / len(years) * 100):.1f}% complet. Temps estimé restant: {estimated_time:.2f} minutes")
                time.sleep(0.1)  # Petit délai pour assurer la mise à jour de l'UI
                progress_bar.empty()
                status_text.empty()
    else:
        scores = dea_ccr(df, inputs, outputs, model_type, orientation)
        df_all = df[["Ticker", "Year", "Index", "Total", "ROE"]].copy()
        df_all["Efficiency"] = scores
        df_all["Model"] = model_type
        df_all["Yearly"] = False
        results.append(df_all)

    return pd.concat(results, ignore_index=True)

def dea_ccr(df, inputs, outputs, model_type, orientation):
    n = len(df)
    efficiencies = []
    for i in range(n):
        prob = LpProblem("Enveloppe DEA", LpMaximize)
        phi = LpVariable("phi", lowBound=0)
        lambda_j = [LpVariable(f"lambda_{j}", lowBound=0) for j in range(n)]

        prob += phi

        # Contraintes des intrants
        for k in range(len(inputs)):
            prob += lpSum([lambda_j[j] * df[inputs[k]].iloc[j] for j in range(n)]) <= df[inputs[k]].iloc[i]

        # Contraintes des sorties
        for k in range(len(outputs)):
            prob += lpSum([lambda_j[j] * df[outputs[k]].iloc[j] for j in range(n)]) >= phi * df[outputs[k]].iloc[i]

        if model_type == "VRS":
            prob += lpSum(lambda_j) == 1

        prob.solve()
        if LpStatus[prob.status] == "Optimal":
            eff = phi.varValue
            efficiencies.append(1 / eff if eff > 0 else 0.0)
        else:
            efficiencies.append(0.0)  # Infeasible/unbounded

    return efficiencies

# ------------------ Principal ------------------
st.title("Tableau de bord DEA ESG-Finance")
st.caption("Charge les données ESG et financières à partir des dossiers esg/ et ratios/, exécute la DEA et visualise l'efficacité.")

# Chargement des données
try:
    esg_files = {"S&P500": "esg/SP500.xlsx", "CAC40": "esg/CAC40.xlsx", "DAX40": "esg/DAX40.xlsx"}
    fin_files = {"S&P500": "ratios/SP500_Ratios.xlsx", "CAC40": "ratios/CAC40_Ratios.xlsx", "DAX40": "ratios/DAX40_Ratios.xlsx"}
    esg_dfs = {}
    fin_dfs = {}
    for idx in esg_files:
        if os.path.exists(esg_files[idx]):
            esg_dfs[idx], _ = load_esg_data(esg_files[idx])
            esg_dfs[idx]["Index"] = idx
        if os.path.exists(fin_files[idx]):
            fin_dfs[idx], _ = load_financial_data(fin_files[idx])
            fin_dfs[idx]["Index"] = idx
    if not esg_dfs or not fin_dfs:
        st.error("Fichiers manquants dans les dossiers esg/ ou ratios/.")
        st.stop()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# Fusion des données
df = pd.concat([esg_dfs[idx].merge(fin_dfs[idx], on=["Ticker", "Year", "Index"], how="inner") for idx in esg_dfs], ignore_index=True)

# Onglets
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Aperçu des données", "Configuration DEA", "Vue globale", "Vue par indice", "Exporter", "Tickers S&P500", "Tickers CAC40", "Tickers DAX40"])

with tab1:
    st.subheader("Aperçu des données")
    st.dataframe(df)
    st.write(f"Total lignes : {len(df)}, Tickers : {len(df['Ticker'].unique())}, Années : {df['Year'].min()}–{df['Year'].max()}")

with tab2:
    st.subheader("Configuration DEA")
    st.write("**Choisir le type de modèle DEA** : Sélectionnez le modèle approprié en fonction de vos besoins d'analyse. Voici des détails pour guider votre décision :")
    model_type = st.selectbox("Type de modèle", ["CRS", "VRS"], index=0, 
                             help="CRS suppose des rendements d'échelle constants sans contraintes sur l'efficacité d'échelle. VRS suppose des rendements d'échelle variables avec une contrainte de convexité.")
    yearly = st.checkbox("Exécuter la DEA par année", value=True)
    if st.button("Exécuter DEA"):
        dea_results = run_dea(df, model_type, "output", yearly)
        st.session_state["dea_results"] = dea_results
        st.success("DEA terminée !")
    if "dea_results" in st.session_state:
        st.dataframe(st.session_state["dea_results"])
    st.write("**CRS (Rendements d'échelle constants)** : Suppose que la sortie augmente proportionnellement à l'entrée. Convient lorsque l'efficacité d'échelle n'est pas une préoccupation, sans contrainte de convexité. Utilisez ceci si vous supposez que toutes les entreprises opèrent à une échelle optimale.")
    st.write("**VRS (Rendements d'échelle variables)** : Suppose que la sortie n'augmente pas proportionnellement à l'entrée, incluant une contrainte de convexité (somme des lambdas = 1). Idéal pour analyser des entreprises avec des échelles opérationnelles variables ou lorsque l'efficacité d'échelle varie.")

with tab3:
    st.subheader("Vue globale")
    if "dea_results" in st.session_state:
        dea_df = st.session_state["dea_results"]
        c1, c2 = st.columns(2)
        with c1:
            top_eff = dea_df.sort_values("Efficiency", ascending=False).head(20)
            fig = px.bar(top_eff, x="Efficiency", y="Ticker", color="Index", title="Top 20 Efficacité")
            st.plotly_chart(fig, use_container_width=True)
            st.write("**Description** : Ce graphique en barres affiche les 20 premières entreprises avec les scores d'efficacité les plus élevés à travers tous les indices (S&P500, CAC40, DAX40). La hauteur de chaque barre représente le score d'efficacité, et la couleur indique l'indice auquel appartient l'entreprise. Utilisez ceci pour identifier les meilleurs performers globaux.")
        with c2:
            fig = px.scatter(dea_df, x="Total", y="Efficiency", color="Index", hover_data=["Ticker", "Year"], title="Efficacité vs Score ESG Total")
            st.plotly_chart(fig, use_container_width=True)
            st.write("**Description** : Ce graphique de dispersion montre la relation entre les scores ESG totaux et l'efficacité. Chaque point représente une entreprise, colorée par indice. Passez la souris sur les points pour voir les détails du ticker et de l'année, aidant à analyser l'impact ESG sur l'efficacité.")
        fig = px.line(dea_df.groupby(["Year", "Index"])["Efficiency"].mean().reset_index(), x="Year", y="Efficiency", color="Index", title="Tendances d'efficacité")
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Description** : Ce graphique linéaire suit l'efficacité moyenne sur les années pour chaque indice. Il met en évidence les tendances dans la performance d'efficacité, permettant de voir les améliorations ou déclins au fil du temps.")
        fig = px.imshow(dea_df.pivot_table(index="Ticker", columns="Year", values="Efficiency").fillna(0), title="Carte thermique d'efficacité")
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Description** : Cette carte thermique visualise les scores d'efficacité pour chaque ticker sur les années. Les couleurs plus foncées indiquent une efficacité plus élevée, aidant à repérer les motifs ou anomalies dans la performance au fil du temps.")

with tab4:
    st.subheader("Vue par indice")
    idx = st.selectbox("Sélectionner un indice", ["S&P500", "CAC40", "DAX40"])
    if "dea_results" in st.session_state:
        dea_df = st.session_state["dea_results"]
        df_idx = dea_df[dea_df["Index"] == idx]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df_idx.sort_values("Efficiency", ascending=False).head(20), x="Efficiency", y="Ticker", title=f"Top Efficacité ({idx})")
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"**Description** : Ce graphique en barres montre les 20 premières entreprises dans l'indice {idx} avec les scores d'efficacité les plus élevés. La hauteur de la barre représente l'efficacité, utile pour identifier les leaders dans ce marché spécifique.")
        with c2:
            fig = px.scatter(df_idx, x="Total", y="ROE", size="Efficiency", hover_data=["Ticker", "Year"], title=f"ROE vs Score ESG Total ({idx})")
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"**Description** : Ce graphique de dispersion compare le Retour sur Équité (ROE) avec les scores ESG totaux pour l'indice {idx}. La taille des points reflète l'efficacité, et en passant la souris, vous voyez le ticker et l'année, aidant à évaluer la performance financière par rapport à l'ESG.")
        fig = px.imshow(df_idx.pivot_table(index="Ticker", columns="Year", values="Efficiency").fillna(0), title=f"Carte thermique d'efficacité ({idx})")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Description** : Cette carte thermique affiche les scores d'efficacité pour chaque ticker dans l'indice {idx} sur les années. Les teintes plus foncées indiquent une efficacité plus élevée, aidant à analyser la performance temporelle dans cet indice.")

with tab5:
    st.subheader("Exporter les résultats")
    if "dea_results" in st.session_state:
        dea_df = st.session_state["dea_results"]
        output = BytesIO()
        dea_df.to_excel(output, index=False)
        excel_data = output.getvalue()
        st.download_button(
            label="Télécharger les résultats DEA (Excel)",
            data=excel_data,
            file_name="resultats_dea.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        csv_data = dea_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les résultats DEA (CSV)",
            data=csv_data,
            file_name="resultats_dea.csv",
            mime="text/csv"
        )

with tab6:
    st.subheader("Tickers S&P500")
    if "dea_results" in st.session_state:
        dea_df = st.session_state["dea_results"]
        sp500_tickers = dea_df[dea_df["Index"] == "S&P500"]["Ticker"].unique().tolist()
        st.write(sp500_tickers)
    else:
        st.write("Exécutez la DEA d'abord pour voir les données.")

with tab7:
    st.subheader("Tickers CAC40")
    if "dea_results" in st.session_state:
        dea_df = st.session_state["dea_results"]
        cac40_tickers = dea_df[dea_df["Index"] == "CAC40"]["Ticker"].unique().tolist()
        st.write(cac40_tickers)
    else:
        st.write("Exécutez la DEA d'abord pour voir les données.")

with tab8:
    st.subheader("Tickers DAX40")
    if "dea_results" in st.session_state:
        dea_df = st.session_state["dea_results"]
        dax40_tickers = dea_df[dea_df["Index"] == "DAX40"]["Ticker"].unique().tolist()
        st.write(dax40_tickers)
    else:
        st.write("Exécutez la DEA d'abord pour voir les données.")