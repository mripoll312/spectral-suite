import streamlit as st
import os
import shutil
import sys
import pandas as pd
import hashlib
import traceback
import io
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Spectral Suite - Final Fix", layout="wide")
import matplotlib
matplotlib.use('Agg') 

# --- 2. PARCHES ---
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = lambda self, other, ignore_index=True: pd.concat([self, other], ignore_index=ignore_index)

original_new = hashlib.new
def patched_new(name, data=b'', **kwargs):
    kwargs.pop('digest_size', None)
    return original_new(name, data, **kwargs)
hashlib.new = patched_new

if 'file_cache' not in st.session_state:
    st.session_state.file_cache = {}

from pandas.io.parsers.readers import read_csv as _original_pandas_read_func
def patched_read_csv(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, str):
        requested_name = os.path.basename(filepath_or_buffer).strip()
        if requested_name in st.session_state.file_cache:
            return _original_pandas_read_func(io.BytesIO(st.session_state.file_cache[requested_name]), *args, **kwargs)
    return _original_pandas_read_func(filepath_or_buffer, *args, **kwargs)
pd.read_csv = patched_read_csv

def force_metadata_compatibility(dt_instance):
    df = dt_instance._meta_data
    actual_time_col = next((c for c in ['Time_Point', 'Time_Points', 'time_point'] if c in df.columns), None)
    if actual_time_col:
        df['Time_Point'] = df[actual_time_col]
        df['Time_Points'] = df[actual_time_col]
    
    actual_cond_col = next((c for c in ['Condition_Name', 'condition_name'] if c in df.columns), None)
    if actual_cond_col:
        df['Condition_Name'] = df[actual_cond_col]

    for col in df.columns:
        df[col] = df[col].apply(lambda x: os.path.basename(x) if isinstance(x, str) and (('/' in x) or ('\\' in x)) else x)
    dt_instance._meta_data = df

from data_toolbox import Data

# --- 3. INTERFAZ ---
tab1, tab2 = st.tabs(["🔄 Conversor Magellan", "📊 Spectral Analysis"])

with tab1:
    st.header("🔄 Convertidor")
    uploaded_xlsx = st.file_uploader("Subir archivo Magellan (.xlsx)", type=["xlsx"])
    if uploaded_xlsx:
        df_mag = pd.read_excel(uploaded_xlsx)
        csv_buf = io.StringIO()
        df_mag.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Descargar CSV", csv_buf.getvalue(), f"conv_{uploaded_xlsx.name}.csv")

with tab2:
    st.header("🔬 Spectral Analysis")
    col1, col2, col3 = st.columns(3)
    with col1: meta_file = st.file_uploader("1. Metadata (.csv)", type=["csv"])
    with col2: ref_files = st.file_uploader("2. Muestras (.tsv)", type=["tsv"], accept_multiple_files=True)
    with col3: data_files = st.file_uploader("3. Estándares (.csv)", type=["csv"], accept_multiple_files=True)

    if st.button("🚀 Iniciar Análisis"):
        if meta_file and ref_files and data_files:
            st.session_state.file_cache = {'metadata.csv': meta_file.getvalue()}
            for f in ref_files: st.session_state.file_cache[os.path.basename(f.name)] = f.getvalue()
            for f in data_files: st.session_state.file_cache[os.path.basename(f.name)] = f.getvalue()

            tmp_dir = "workspace"
            if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
            os.makedirs(os.path.join(tmp_dir, "csv_files"), exist_ok=True)
            os.makedirs(os.path.join(tmp_dir, "graphs"), exist_ok=True)

            for f_name, f_content in st.session_state.file_cache.items():
                with open(os.path.join(tmp_dir, f_name), "wb") as out: out.write(f_content)

            try:
                with st.spinner("Procesando datos..."):
                    old_cwd = os.getcwd()
                    os.chdir(tmp_dir)
                    sys.argv = ["data_toolbox.py", "-m", "metadata.csv"]
                    dt = Data()
                    dt.parse_args()
                    force_metadata_compatibility(dt)
                    dt.read_data(); dt.sub_background(); dt.rearrange_data(); dt.group_data(); dt.conversion_rate()
                    meta_map = dt._meta_data.set_index('Unique_Sample_ID')[['Condition_Name', 'Time_Points']].to_dict('index')
                    os.chdir(old_cwd)

                # --- BLOQUE 1: RATES (RECUPERADOS) ---
                st.subheader("📉 Tasas de Conversión (Rates)")
                
                csv_rates = os.path.join(tmp_dir, "csv_files", "08_conv_rates.csv")
                if os.path.exists(csv_rates):
                    df_r = pd.read_csv(csv_rates)
                    for cond in df_r.iloc[:, 0].unique():
                        df_cond = df_r[df_r.iloc[:, 0] == cond].sort_values(df_r.columns[1])
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(df_cond.iloc[:,1], df_cond.iloc[:,2], 'o-', label='Substrate', color='#1f77b4')
                        ax.plot(df_cond.iloc[:,1], df_cond.iloc[:,3], 's-', label='Product', color='#ff7f0e')
                        ax.set_title(f"Rates: {cond}")
                        ax.set_xlabel("Time (min)"); ax.set_ylabel("Concentration / Conversion")
                        ax.legend(); ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        fig.savefig(os.path.join(tmp_dir, "graphs", f"rates_{cond}.png"))
                        plt.close(fig)

                # --- BLOQUE 2: ESPECTROS MRP AGRUPADOS ---
                st.subheader("📊 Evolución de Espectros")
                
                master_csv = os.path.join(tmp_dir, "csv_files", "00b_df_complete_background_subtracted_and_dropped.csv")
                if os.path.exists(master_csv):
                    df_master = pd.read_csv(master_csv)
                    wavelengths = df_master.iloc[:, 0]
                    mrp_cols = [c for c in df_master.columns[1:] if not any(f"spectrum_{l}" in c for l in "ABCDEFGH")]
                    
                    grouped_spectra = {}
                    for col in mrp_cols:
                        if col in meta_map:
                            c_n = meta_map[col]['Condition_Name']
                            t_p = meta_map[col]['Time_Points']
                            if c_n not in grouped_spectra: grouped_spectra[c_n] = []
                            grouped_spectra[c_n].append((t_p, col))

                    for cond_name, curves in sorted(grouped_spectra.items()):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for t_val, col_name in sorted(curves, key=lambda x: x[0]):
                            ax.plot(wavelengths, df_master[col_name], label=f"{t_val} min", linewidth=1.5)
                        ax.set_xlabel("Wavelength (nm)", fontweight='bold'); ax.set_ylabel("Absorbance", fontweight='bold')
                        ax.set_title(f"Espectros: {cond_name}"); ax.legend(title="Tiempo", loc='upper right'); ax.grid(True, alpha=0.5)
                        st.pyplot(fig)
                        fig.savefig(os.path.join(tmp_dir, "graphs", f"spectrum_{cond_name}.png"))
                        plt.close(fig)

                shutil.make_archive("resultados_completos", 'zip', tmp_dir)
                st.success("✅ ¡Análisis completo!")
                with open("resultados_completos.zip", "rb") as f:
                    st.download_button("⬇️ Descargar Resultados", f, "spectral_results.zip")

            except Exception as e:
                st.error(f"Error: {e}"); st.code(traceback.format_exc())
                if 'old_cwd' in locals(): os.chdir(old_cwd)
