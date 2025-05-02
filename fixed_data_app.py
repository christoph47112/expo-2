import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.figure import Figure
from threading import RLock

# Thread-Lock für Matplotlib (wichtig für Streamlit Cloud)
_lock = RLock()

# Seitentitel und Konfiguration
st.set_page_config(
    page_title="Exponentielle Glättung 1. Ordnung", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS für besseres Aussehen und optimiertes Layout
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stApp {
        max-width: 100%;
    }
    h1, h2, h3 {
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #e6f3ff;
        border-radius: 0.5rem;
        padding: 1rem;
        flex: 1 1 200px;
    }
    .forecast-box {
        background-color: #e6ffe6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    /* Sidebar-Anpassung */
    .css-1d391kg, .css-1lcbmhc {
        width: 18rem !important;
    }
    /* Volle Breite für den Hauptinhalt */
    .css-18e3th9 {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Optimiertes Layout für mobile Geräte */
    @media (max-width: 768px) {
        .css-1d391kg, .css-1lcbmhc {
            width: 14rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Titel und Einführung
st.title("Exponentielle Glättung 1. Ordnung")
st.markdown("""
Diese App demonstriert die exponentielle Glättung 1. Ordnung für eine Zeitreihe von Wochendaten.
Passen Sie den Glättungsfaktor an, um zu sehen, wie sich dies auf die Glättung und Vorhersage auswirkt.
""")

# Funktionen für die exponentielle Glättung und Fehlermetriken
def exponential_smoothing(data, alpha):
    """
    Berechnet die exponentielle Glättung 1. Ordnung für eine Zeitreihe
    
    Parameters:
    -----------
    data : array-like
        Die Originaldaten der Zeitreihe
    alpha : float
        Der Glättungsfaktor (zwischen 0 und 1)
        
    Returns:
    --------
    smoothed : array-like
        Die geglätteten Werte
    """
    smoothed = np.zeros(len(data))
    smoothed[0] = data[0]  # Erster Wert bleibt gleich
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed

def calculate_error_metrics(actual, predicted):
    """
    Berechnet verschiedene Fehlermetriken
    
    Parameters:
    -----------
    actual : array-like
        Die tatsächlichen Werte
    predicted : array-like
        Die vorhergesagten Werte
        
    Returns:
    --------
    metrics : dict
        Ein Dictionary mit den berechneten Fehlermetriken
    """
    # Stellen Sie sicher, dass wir mit numpy arrays arbeiten
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Fehlerberechnung
    error = actual - predicted
    abs_error = np.abs(error)
    squared_error = error ** 2
    
    # MAE - Mean Absolute Error
    mae = np.mean(abs_error)
    
    # MSE - Mean Squared Error
    mse = np.mean(squared_error)
    
    # RMSE - Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # MAPE - Mean Absolute Percentage Error
    # Nur für Nicht-Null-Werte berechnen
    non_zero = actual != 0
    mape = np.mean(abs_error[non_zero] / np.abs(actual[non_zero])) * 100 if np.any(non_zero) else np.nan
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def get_download_link(df, filename, text):
    """
    Erstellt einen Download-Link für ein DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Initialisierung von Session State für persistente Daten
if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.2
if 'chart_height' not in st.session_state:
    st.session_state.chart_height = 500
if 'show_grid' not in st.session_state:
    st.session_state.show_grid = True
if 'download_format' not in st.session_state:
    st.session_state.download_format = "CSV"

# Seitenlayout: Sidebar für Parameter
with st.sidebar:
    st.header("Parameter")
    
    # Glättungsparameter
    st.session_state.alpha = st.slider(
        "Glättungsfaktor (α)",
        min_value=0.01,
        max_value=0.99,
        value=st.session_state.alpha,
        step=0.01,
        help="Niedrigere Werte (nahe 0) glätten stärker, höhere Werte (nahe 1) folgen den Originaldaten enger."
    )
    
    # Optionen für das Diagramm in einen Expander auslagern
    with st.expander("Diagramm-Optionen"):
        st.session_state.chart_height = st.slider(
            "Diagrammhöhe", 
            300, 800, 
            st.session_state.chart_height, 
            50
        )
        st.session_state.show_grid = st.checkbox(
            "Gitter anzeigen", 
            st.session_state.show_grid
        )
    
    # Download-Optionen
    with st.expander("Download-Optionen"):
        st.session_state.download_format = st.radio(
            "Format", 
            ["CSV", "Excel"],
            index=0 if st.session_state.download_format == "CSV" else 1
        )

# Die festen Daten aus dem React-Code
raw_data = """Jahr Woche Menge 
2023 30 21 
2023 31 28 
2023 32 20 
2023 33 26 
2023 34 17 
2023 35 17 
2023 36 2 
2023 37 21 
2023 38 48 
2023 39 17 
2023 40 27 
2023 41 3 
2023 42 8 
2023 43 1 
2023 44 10 
2023 45 17 
2023 46 26 
2023 47 6 
2023 48 12 
2023 49 27 
2023 50 13 
2023 51 14 
2023 52 1 
2024 01 23 
2024 02 6 
2024 03 21 
2024 04 15 
2024 05 29 
2024 06 7 
2024 07 27 
2024 08 3 
2024 09 7 
2024 10 8 
2024 11 12 
2024 12 10 
2024 13 10 
2024 14 15 
2024 15 29 
2024 16 19 
2024 17 10 
2024 19 33 
2024 20 15 
2024 21 12 
2024 22 19 
2024 23 29 
2024 24 17 
2024 25 13 
2024 26 26 
2024 27 20 
2024 28 16 
2024 29 15 
2024 30 19 
2024 31 36 
2024 32 35 
2024 33 19 
2024 34 26 
2024 35 28 
2024 36 51 
2024 37 5 
2024 38 38 
2024 39 51 
2024 40 8 
2024 41 40 
2024 42 12 
2024 43 21 
2024 44 15 
2024 46 8 
2024 47 10 
2024 48 12 
2024 49 18 
2024 50 23 
2024 51 21 
2024 52 9 
2025 01 6 
2025 02 23 
2025 03 28 
2025 04 3 
2025 05 29 
2025 06 20 
2025 07 9 
2025 08 39 
2025 09 20 
2025 10 8 
2025 11 11 
2025 12 13 
2025 13 26 
2025 14 11 
2025 15 23"""

# Daten verarbeiten
lines = raw_data.strip().split('\n')
data_list = []

# Kopfzeile überspringen
for i in range(1, len(lines)):
    parts = lines[i].strip().split()
    if len(parts) >= 3:
        year = parts[0]
        week = parts[1].zfill(2)  # Führende Null hinzufügen
        amount = int(parts[2])
        data_list.append({
            'Periode': f"{year}-W{week}",
            'Jahr': year,
            'Woche': week,
            'Menge': amount
        })

# Konvertierung in DataFrame
df = pd.DataFrame(data_list)

# Berechnungen
original_data = df['Menge'].values
with _lock:  # Thread-Lock für Matplotlib
    smoothed_data = exponential_smoothing(original_data, st.session_state.alpha)

# Ergebnisse in DataFrame speichern
result_df = pd.DataFrame()
result_df['Periode'] = df['Periode']
result_df['Originalwert'] = original_data
result_df['Geglätteter Wert'] = smoothed_data

# Für Vorhersagefehler: Geglätteter Wert des vorherigen Zeitpunkts als Prognose
one_step_forecast = np.zeros_like(smoothed_data)
one_step_forecast[1:] = smoothed_data[:-1]  # Verschieben um 1
one_step_forecast[0] = original_data[0]  # Erster Wert hat keine Vorhersage

result_df['Vorhersage (t+1)'] = one_step_forecast

# Fehler berechnen
errors = calculate_error_metrics(original_data[1:], one_step_forecast[1:])

# Layout mit optimierten Spaltenbreiten
col1, col2 = st.columns([7, 3])

with col1:
    # Diagramm
    st.header("Visualisierung")
    
    with _lock:  # Thread-Lock für Matplotlib
        fig, ax = plt.subplots(figsize=(10, st.session_state.chart_height/100))
        
        # Seaborn-Styling für besseres Aussehen
        sns.set_style("whitegrid" if st.session_state.show_grid else "white")
        
        # Originaldaten plotten
        ax.plot(range(len(original_data)), original_data, 'o-', label='Originaldaten', color='#8884d8', markersize=4)
        
        # Geglättete Daten plotten
        ax.plot(range(len(smoothed_data)), smoothed_data, 'o-', label='Geglättete Werte', color='#82ca9d', markersize=4)
        
        # X-Achsenbeschriftungen
        ticks = list(range(0, len(result_df), 10))  # Alle 10 Einträge
        if len(result_df) - 1 not in ticks:
            ticks.append(len(result_df) - 1)  # Letzten Eintrag hinzufügen
            
        ax.set_xticks(ticks)
        ax.set_xticklabels([result_df['Periode'].iloc[i] for i in ticks], rotation=45)
        
        # Titel und Labels
        ax.set_title("Exponentielle Glättung 1. Ordnung", fontsize=14)
        ax.set_xlabel("Zeitperiode")
        ax.set_ylabel("Menge")
        
        # Legende
        ax.legend()
        
        # Grid
        ax.grid(st.session_state.show_grid)
        
        # Diagramm anzeigen
        plt.tight_layout()
        st.pyplot(fig)
    
    # Tabelle mit den Ergebnissen
    st.header("Tabelle der Werte")
    
    # Formatierung der geglätteten Werte auf 2 Dezimalstellen
    formatted_result_df = result_df.copy()
    formatted_result_df['Geglätteter Wert'] = formatted_result_df['Geglätteter Wert'].round(2)
    formatted_result_df['Vorhersage (t+1)'] = formatted_result_df['Vorhersage (t+1)'].round(2)
    
    # Nur die ersten 10 Einträge anzeigen
    st.dataframe(formatted_result_df.head(10), use_container_width=True)
    
    # Hinweis auf weitere Einträge
    if len(formatted_result_df) > 10:
        st.info(f"... und {len(formatted_result_df) - 10} weitere Einträge")
    
    # Download-Links
    st.markdown("### Ergebnisse herunterladen")
    
    if st.session_state.download_format == "CSV":
        st.markdown(get_download_link(formatted_result_df, 
                                     f"exponentielle_glaettung_alpha_{st.session_state.alpha}.csv",
                                     "Ergebnisse als CSV herunterladen"), 
                   unsafe_allow_html=True)
    else:  # Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            formatted_result_df.to_excel(writer, sheet_name="Ergebnisse", index=False)
            # Zweites Blatt mit Metriken
            pd.DataFrame([errors]).to_excel(writer, sheet_name="Metriken", index=False)
            # Parameter-Blatt
            params_df = pd.DataFrame({
                'Parameter': ['Alpha'],
                'Wert': [st.session_state.alpha]
            })
            params_df.to_excel(writer, sheet_name="Parameter", index=False)
        
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="exponentielle_glaettung_alpha_{st.session_state.alpha}.xlsx">Ergebnisse als Excel herunterladen</a>'
        st.markdown(href, unsafe_allow_html=True)

with col2:
    # Fehlermetriken
    st.header("Fehlermetriken")
    
    # Metriken in einem Grid anzeigen
    metrics_cols = st.columns(2)
    for i, (metric, value) in enumerate(errors.items()):
        with metrics_cols[i % 2]:
            st.metric(
                label=metric,
                value=f"{value:.2f}" + ("%" if metric == "MAPE" else "")
            )
    
    # Prognose für den nächsten Zeitpunkt
    st.header("Prognose")
    
    st.markdown("""
    <div style="background-color: #e6ffe6; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <p style="font-weight: 600;">Vorhersage für den nächsten Zeitpunkt:</p>
        <p style="font-size: 24px; font-weight: bold;">{:.2f}</p>
    </div>
    """.format(smoothed_data[-1]), unsafe_allow_html=True)
    
    # Erklärung der Metriken
    st.header("Erklärung")
    
    st.markdown("""
    <div style="background-color: #f0f2f6; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <p><strong>Exponentielle Glättung 1. Ordnung</strong></p>
        <p>Formel: S<sub>t</sub> = α × Y<sub>t</sub> + (1 - α) × S<sub>t-1</sub></p>
        <p>Wobei:</p>
        <ul>
            <li>S<sub>t</sub> = Geglätteter Wert zum Zeitpunkt t</li>
            <li>Y<sub>t</sub> = Beobachteter Wert zum Zeitpunkt t</li>
            <li>α = Glättungsfaktor (zwischen 0 und 1)</li>
            <li>S<sub>t-1</sub> = Geglätteter Wert des vorherigen Zeitpunkts</li>
        </ul>
    </div>
    
    <div style="background-color: #f0f2f6; border-radius: 0.5rem; padding: 1rem;">
        <p><strong>Fehlermetriken:</strong></p>
        <ul>
            <li><strong>MAE</strong>: Mean Absolute Error - Durchschnittlicher absoluter Fehler</li>
            <li><strong>MSE</strong>: Mean Squared Error - Durchschnittlicher quadratischer Fehler</li>
            <li><strong>RMSE</strong>: Root Mean Squared Error - Wurzel aus dem MSE</li>
            <li><strong>MAPE</strong>: Mean Absolute Percentage Error - Durchschnittlicher prozentualer Fehler</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Füge eine Fußzeile hinzu
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Entwickelt mit Streamlit • Exponentielle Glättung 1. Ordnung</p>
</div>
""", unsafe_allow_html=True)
