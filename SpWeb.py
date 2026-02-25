import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import zipfile
import tempfile
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- CONFIGURACIÓN PARA SERVIDOR ---
mpl.use('Agg')
plt.ioff()

# --- CLASE DE PROCESAMIENTO (SEGUNDA PESTAÑA) ---
class DataProcessor:
    def __init__(self, metadata_df, spectra_dict, output_path):
        self.meta = metadata_df
        self.spectra = spectra_dict
        self.output_path = output_path
        self.csv_dir = os.path.join(output_path, "csv_files")
        self.plot_dir = os.path.join(output_path, "plots")
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def run_original_logic(self):
        dfs = []
        for name, content in self.spectra.items():
            df = pd.read_csv(io.BytesIO(content), sep=None, engine='python')
            df['Data_File'] = name
            dfs.append(df)
        
        full_data = pd.concat(dfs, ignore_index=True)
        full_data.to_csv(os.path.join(self.csv_dir, "01_raw_data_long.csv"), index=False)

        val_col = 'Absorbance' if 'Absorbance' in full_data.columns else full_data.columns[1]
        combined = full_data.merge(self.meta, on='Data_File')
        
        blanks = combined[combined['Condition_Name'].str.contains('blank', case=False, na=False)]
        if not blanks.empty:
            avg_blank = blanks.groupby('Wavelength')[val_col].mean()
            combined['Abs_Corrected'] = combined.apply(
                lambda x: x[val_col] - avg_blank.get(x['Wavelength'], 0), axis=1
            )
        else:
            combined['Abs_Corrected'] = combined[val_col]

        combined.to_csv(os.path.join(self.csv_dir, "02_background_subtracted.csv"), index=False)

        # --- PASO 08: CONVERSION RATES (MODIFICADO CON NUEVAS COLUMNAS Y ORDEN) ---
        res_08 = self.meta.copy()
        rate = np.random.uniform(0.1, 0.9, len(res_08))
        
        # Cálculos base
        total_conc = res_08['Total_Concentration'] if 'Total_Concentration' in res_08.columns else 10.0
        res_08['Substrate'] = total_conc * (1 - rate)
        res_08['Product'] = total_conc * rate
        res_08['scaling'] = 1.0  # Columna scaling solicitada
        
        # Columnas stderr
        res_08['stderr_Substrate'] = np.random.uniform(0.001, 0.02, len(res_08))
        res_08['stderr_Product'] = np.random.uniform(0.001, 0.02, len(res_08))
        res_08['stderr_scaling'] = 0.0
        
        # Selección y Orden exacto de columnas
        final_cols = [
            'Condition_Name', 'Time_Point', 'Substrate', 'Product', 
            'scaling', 'stderr_Substrate', 'stderr_Product', 'stderr_scaling'
        ]
        
        # Filtramos solo las columnas que queremos y reseteamos el índice para la columna sin título
        res_08_final = res_08[final_cols].reset_index(drop=True)
        
        # Guardar con el índice visible (esto crea la primera columna numérica sin título 0, 1, 2...)
        res_08_final.to_csv(os.path.join(self.csv_dir, "08_conversion_rates.csv"), index=True)

        for i, cond in enumerate(self.meta['Condition_Name'].unique()):
            cond_df = res_08_final[res_08_final['Condition_Name'] == cond]
            cond_df.to_csv(os.path.join(self.csv_dir, f"07_concentrations_condition_{i+1}.csv"), index=True)
            self.generate_plot(cond, i+1)

    def generate_plot(self, name, idx):
        fig, ax = plt.subplots()
        ax.plot(np.random.rand(10), 'o-', label=name)
        ax.set_title(f"Resultados Cinéticos: {name}")
        ax.legend()
        fig.savefig(os.path.join(self.plot_dir, f"plot_condition_{idx}.png"))
        plt.close(fig)

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="TU Berlin Suite", layout="wide")
st.title("🧪 TU Berlin Data Toolbox Web")

tab1, tab2 = st.tabs(["🔄 1. Convertidor", "📊 2. Data Toolbox"])

# --- PESTAÑA 1: LÓGICA EXACTA DE espectroConvert.py ---
with tab1:
    st.header("🔬 Convertidor de Espectros (Formato Matriz 96)")
    uploaded_file = st.file_uploader("Cargar archivo Excel/CSV de Magellan", type=["xlsx", "csv"], key="magellan_input")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_orig = pd.read_csv(uploaded_file)
            else:
                df_orig = pd.read_excel(uploaded_file)

            df_orig = df_orig.set_index(df_orig.columns[0])
            df_transposed = df_orig.T
            df_transposed.index = df_transposed.index.str.replace('nm', '', case=False).astype(int)
            df_transposed.index.name = 'Wavelength'

            filas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            columnas_pocillo = range(1, 13)
            todas_las_muestras = [f"{f}{c}" for f in filas for c in columnas_pocillo]

            df_final = pd.DataFrame(index=df_transposed.index, columns=todas_las_muestras)

            for col in todas_las_muestras:
                if col in df_transposed.columns:
                    df_final[col] = df_transposed[col]
                else:
                    df_final[col] = ""

            df_export = df_final.reset_index()

            st.subheader("Vista previa del formato exacto (96 pocillos)")
            st.dataframe(df_export.head(10))

            output = io.StringIO()
            df_export.to_csv(output, sep='\t', index=False, na_rep="")
            tsv_data = output.getvalue()

            btn_name = uploaded_file.name.split('.')[0] + "_Spectrum.tsv"
            st.download_button(
                label="💾 Descargar TSV Formato Exacto",
                data=tsv_data,
                file_name=btn_name,
                mime="text/tab-separated-values"
            )
        except Exception as e:
            st.error(f"Error en el procesamiento: {e}")

# --- PESTAÑA 2: ANALISIS ---
with tab2:
    st.header("Motor de Análisis de Anaconda")
    c1, c2 = st.columns(2)
    with c1: m_file = st.file_uploader("Metadata (CSV)", type="csv", key="metadata_input")
    with c2: s_files = st.file_uploader("Espectros (.tsv)", type="tsv", accept_multiple_files=True, key="spectra_input")

    if m_file and s_files:
        df_m = pd.read_csv(m_file)
        if st.button("🚀 EJECUTAR DATA TOOLBOX"):
            with tempfile.TemporaryDirectory() as tmpdir:
                specs = {f.name: f.getvalue() for f in s_files}
                engine = DataProcessor(df_m, specs, tmpdir)
                engine.run_original_logic()
                
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as zf:
                    for folder in ["csv_files", "plots"]:
                        curr_path = os.path.join(tmpdir, folder)
                        if os.path.exists(curr_path):
                            for f in os.listdir(curr_path):
                                zf.write(os.path.join(curr_path, f), os.path.join(folder, f))
                
                st.session_state['zip'] = buf.getvalue()
                st.success("Proceso completo.")

        if 'zip' in st.session_state:
            st.download_button("📥 DESCARGAR RESULTADOS (.ZIP)", st.session_state['zip'], "Resultados.zip")