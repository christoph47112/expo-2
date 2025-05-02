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
Diese App ermöglicht die Berechnung der exponentiellen Glättung 1. Ordnung für Zeitreihendaten.
Laden Sie eine CSV- oder Excel-Datei hoch und passen Sie die Parameter an, um Ihre Daten zu analysieren.
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
if 'df' not in st.session_state:
    st.session_state.df = None
if 'value_col' not in st.session_state:
    st.session_state.value_col = None
if 'date_col' not in st.session_state:
    st.session_state.date_col = None
if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.2
if 'chart_height' not in st.session_state:
    st.session_state.chart_height = 500
if 'show_grid' not in st.session_state:
    st.session_state.show_grid = True
if 'download_format' not in st.session_state:
    st.session_state.download_format = "CSV"
if 'header_row' not in st.session_state:
    st.session_state.header_row = 0
if 'has_selected_columns' not in st.session_state:
    st.session_state.has_selected_columns = False

# Sidebar für Datei-Upload und Parameter
with st.sidebar:
    st.header("Daten hochladen")
    
    # Optionen für den Datei-Import
    st.subheader("Import-Optionen")
    st.session_state.header_row = st.number_input(
        "Überschriftenzeile (0 = erste Zeile)", 
        min_value=0, 
        value=st.session_state.header_row,
        help="Wählen Sie die Zeile, in der die Spaltenüberschriften stehen. 0 = erste Zeile, 1 = zweite Zeile, usw."
    )
    
    uploaded_file = st.file_uploader("Zeitreihendaten hochladen", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Dateiformat erkennen und einlesen
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file, header=int(st.session_state.header_row))
            else:  # Excel-Dateien
                st.session_state.df = pd.read_excel(uploaded_file, header=int(st.session_state.header_row))
            
            st.success(f"Datei erfolgreich geladen")
            
            # Kompaktere Datenvorschau
            with st.expander("Datenübersicht"):
                st.write(f"Zeilen: {st.session_state.df.shape[0]} | Spalten: {st.session_state.df.shape[1]}")
                st.dataframe(st.session_state.df.head(5), height=200)
            
            # Reset Spaltenwahl wenn neue Datei hochgeladen wird
            st.session_state.has_selected_columns = False
        
        except Exception as e:
            st.error(f"Fehler beim Laden der Datei: {e}")
            st.session_state.df = None
    
    # Parameter nur anzeigen, wenn Daten geladen wurden
    if st.session_state.df is not None:
        st.header("Parameter")
        
        # Spaltenauswahl für Werte
        numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        date_cols = [col for col in st.session_state.df.columns if any(word in str(col).lower() for word in ['date', 'zeit', 'jahr', 'monat', 'woche', 'period'])]
        
        # Sichere Standardwerte festlegen
        default_value_index = 0
        if numeric_cols and not st.session_state.has_selected_columns:
            st.session_state.value_col = numeric_cols[0]
            
        if not numeric_cols:
            st.warning("Keine numerischen Spalten gefunden. Bitte wählen Sie eine Spalte, die Zahlen enthält.")
        
        value_col_options = st.session_state.df.columns.tolist()
        value_col_index = 0
        if st.session_state.value_col in value_col_options:
            value_col_index = value_col_options.index(st.session_state.value_col)
        
        st.session_state.value_col = st.selectbox(
            "Wertspalte auswählen",
            options=value_col_options,
            index=value_col_index,
            help="Wählen Sie die Spalte mit den zu glättenden Werten aus."
        )
        
        # Optional: Spalte für Zeitperioden
        all_columns = ['Keine (Index verwenden)'] + st.session_state.df.columns.tolist()
        default_date_index = 0
        
        if date_cols and not st.session_state.has_selected_columns:
            st.session_state.date_col = date_cols[0]
            default_date_index = all_columns.index(date_cols[0]) if date_cols[0] in all_columns else 0
        
        date_col_index = 0
        if st.session_state.date_col in all_columns[1:]:  # Skip 'Keine (Index verwenden)'
            date_col_index = all_columns.index(st.session_state.date_col)
        
        selected_date_col = st.selectbox(
            "Datums-/Periodenspalte (optional)",
            options=all_columns,
            index=date_col_index,
            help="Wählen Sie optional eine Spalte für Zeitperioden/Datumsangaben aus."
        )
        
        if selected_date_col == 'Keine (Index verwenden)':
            st.session_state.date_col = None
        else:
            st.session_state.date_col = selected_date_col
        
        # Merken, dass Spalten ausgewählt wurden
        st.session_state.has_selected_columns = True
        
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

# Hauptbereich - nur anzeigen, wenn Daten und Spalten ausgewählt wurden
if st.session_state.df is not None and st.session_state.value_col is not None:
    try:
        df = st.session_state.df.copy()
        
        # Daten vorbereiten
        if st.session_state.date_col:
            # Sortieren nach Datum/Periode, wenn ausgewählt
            try:
                df = df.sort_values(by=st.session_state.date_col)
            except:
                st.warning(f"Sortierung nach {st.session_state.date_col} nicht möglich. Verwende Originalreihenfolge.")
        
        # Originaldaten extrahieren
        try:
            original_data = pd.to_numeric(df[st.session_state.value_col], errors='coerce').fillna(0).values
        except:
            st.error(f"Die Spalte '{st.session_state.value_col}' enthält nicht-numerische Werte, die nicht konvertiert werden können.")
            st.stop()
            
        # Prüfen, ob genügend Daten vorhanden sind
        if len(original_data) < 2:
            st.error("Zu wenige Datenpunkte für die exponentielle Glättung. Mindestens 2 Datenpunkte werden benötigt.")
            st.stop()
            
        # Exponentielle Glättung berechnen
        with _lock:  # Thread-Lock für Matplotlib
            smoothed_data = exponential_smoothing(original_data, st.session_state.alpha)
        
        # Ergebnisse in DataFrame speichern
        result_df = pd.DataFrame()
        
        if st.session_state.date_col:
            result_df['Periode'] = df[st.session_state.date_col]
        else:
            result_df['Periode'] = range(1, len(original_data) + 1)
        
        result_df['Originalwert'] = original_data
        result_df['Geglätteter Wert'] = smoothed_data
        
        # Für Vorhersagefehler müssen wir die verschobenen Werte betrachten
        # (geglätteter Wert von t-1 als Vorhersage für t)
        one_step_forecast = np.zeros_like(smoothed_data)
        one_step_forecast[1:] = smoothed_data[:-1]  # Verschieben um 1
        one_step_forecast[0] = original_data[0]  # Erster Wert hat keine Vorhersage
        
        result_df['Vorhersage (t+1)'] = one_step_forecast
        
        # Fehler berechnen
        errors = calculate_error_metrics(original_data[1:], one_step_forecast[1:])
        
        # Optimiertes Layout mit angepassten Spaltenbreiten
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
                if len(result_df) > 20:
                    ticks = list(range(0, len(result_df), max(1, len(result_df) // 10)))  # Ca. 10 Ticks
                else:
                    ticks = list(range(len(result_df)))
                    
                if len(result_df) - 1 not in ticks:
                    ticks.append(len(result_df) - 1)  # Letzten Eintrag hinzufügen
                
                ax.set_xticks(ticks)
                
                if st.session_state.date_col:
                    x_labels = [str(result_df['Periode'].iloc[i]) for i in ticks]
                    ax.set_xticklabels(x_labels, rotation=45)
                
                # Titel und Labels
                ax.set_title("Exponentielle Glättung 1. Ordnung", fontsize=14)
                ax.set_xlabel("Zeitperiode")
                ax.set_ylabel(st.session_state.value_col)
                
                # Legende
                ax.legend()
                
                # Grid
                ax.grid(st.session_state.show_grid)
                
                # Diagramm anzeigen
                plt.tight_layout()
                st.pyplot(fig)
            
            # Tabelle mit den Ergebnissen
            st.header("Ergebnistabelle")
            
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
                        'Parameter': ['Alpha', 'Wertspalte', 'Periodenspalte'],
                        'Wert': [st.session_state.alpha, st.session_state.value_col, st.session_state.date_col if st.session_state.date_col else 'Index']
                    })
                    params_df.to_excel(writer, sheet_name="Parameter", index=False)
                
                buffer.seek(0)
                b64 = base64.b64encode(buffer.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="exponentielle_glaettung_alpha_{st.session_state.alpha}.xlsx">Ergebnisse als Excel herunterladen</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Fehlermetriken
            st.header("Fehlermetriken")
            
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
    
    except Exception as e:
        st.error(f"Fehler bei der Berechnung: {e}")
        import traceback
        st.code(traceback.format_exc())

# Fügt ein README hinzu, wenn keine Daten geladen wurden
else:
    # Zweispalten-Layout für das Intro
    intro_col1, intro_col2 = st.columns([3, 2])
    
    with intro_col1:
        st.markdown("""
        ## Anleitung zur Verwendung
        
        Diese App berechnet die exponentielle Glättung 1. Ordnung für Zeitreihendaten und bietet:
        
        1. **Datenanalyse**: Laden Sie Ihre CSV- oder Excel-Datei hoch
        2. **Parameterkonfiguration**: Wählen Sie die relevanten Spalten und den Glättungsfaktor
        3. **Visualisierung**: Sehen Sie die Originaldaten und geglätteten Werte im Diagramm
        4. **Fehlermetriken**: Bewerten Sie die Qualität der Glättung anhand verschiedener Metriken
        5. **Prognose**: Erhalten Sie eine Vorhersage für den nächsten Zeitpunkt
        6. **Export**: Laden Sie die Ergebnisse als CSV oder Excel herunter
        
        ### Besonderheiten dieser App:
        
        - **Flexible Tabellenstruktur**: Sie können angeben, in welcher Zeile die Spaltenüberschriften stehen
        - **Automatische Spaltenauswahl**: Die App erkennt potenzielle Wert- und Zeitperiodenspalten
        - **Robuste Fehlerbehandlung**: Nicht-numerische Werte werden automatisch verarbeitet
        """)
    
    with intro_col2:
        st.markdown("### Beispieldatenformat")
        
        st.markdown("""
        **Format 1: Mit Datum/Periode**
        
        | Datum | Wert |
        |-------|------|
        | 2023-01 | 100 |
        | 2023-02 | 120 |
        | 2023-03 | 90 |
        
        **Format 2: Mit Jahr und Woche**
        
        | Jahr | Woche | Menge |
        |------|-------|-------|
        | 2023 | 1 | 100 |
        | 2023 | 2 | 120 |
        | 2023 | 3 | 90 |
        """)
        
        st.info("Laden Sie im linken Seitenmenü eine CSV- oder Excel-Datei mit Ihren Zeitreihendaten hoch, um zu beginnen.")

# Füge eine Fußzeile hinzu
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Entwickelt mit Streamlit • Exponentielle Glättung 1. Ordnung</p>
</div>
""", unsafe_allow_html=True)
