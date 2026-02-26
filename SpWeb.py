import streamlit as st
import os, shutil, sys, io, traceback, hashlib
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Spectral Suite - Full Restore", layout="wide")
import matplotlib
matplotlib.use('Agg') 

if 'file_cache' not in st.session_state:
    st.session_state.file_cache = {}

# --- 2. PARCHES Y UTILIDADES ---
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = lambda self, other, ignore_index=True: pd.concat([self, other], ignore_index=ignore_index)

# Parche para hashlib (Python 3.13)
original_new = hashlib.new
def patched_new(name, data=b'', **kwargs):
    kwargs.pop('digest_size', None)
    return original_new(name, data, **kwargs)
hashlib.new = patched_new

# Parche de Pandas interceptor para Streamlit
from pandas.io.parsers.readers import read_csv as _original_pandas_read_func
def patched_read_csv(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, str):
        fname = os.path.basename(filepath_or_buffer).strip()
        if 'file_cache' in st.session_state and fname in st.session_state.file_cache:
            return _original_pandas_read_func(io.BytesIO(st.session_state.file_cache[fname]), *args, **kwargs)
    return _original_pandas_read_func(filepath_or_buffer, *args, **kwargs)
pd.read_csv = patched_read_csv

def force_metadata_compatibility(dt_instance, uploaded_filenames):
    df = dt_instance._meta_data
    
    # 1. Normalizar nombres de columnas clave
    t_col = next((c for c in ['Time_Points', 'Time_Point', 'time_point'] if c in df.columns), 'Time_Points')
    c_col = next((c for c in ['Condition_Name', 'condition_name'] if c in df.columns), 'Condition_Name')
    df['Time_Points'], df['Condition_Name'] = df[t_col], df[c_col]

    # 2. Sincronizar nombres de archivos (Fix TSV)
    def find_real_name(val):
        if not isinstance(val, str) or not any(x in val.lower() for x in ['.tsv', '.csv']): return val
        base = os.path.basename(val).strip().lower()
        for real in uploaded_filenames:
            if real.lower() == base: return real
        return os.path.basename(val).strip()

    for col in df.columns:
        df[col] = df[col].apply(find_real_name)
    dt_instance._meta_data = df

from data_toolbox import Data

# --- 3. INTERFAZ ---
st.title("🔬 Spectral Analysis Suite (Restore Mode)")
tab1, tab2 = st.tabs(["🔄 Conversor", "📊 Análisis y Gráficos"])

with tab1:
    uploaded_xlsx = st.file_uploader("Subir Magellan (.xlsx)", type=["xlsx"])
    if uploaded_xlsx:
        df_mag = pd.read_excel(uploaded_xlsx)
        csv_buf = io.StringIO()
        df_mag.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Descargar CSV", csv_buf.getvalue(), f"conv_{uploaded_xlsx.name}.csv")

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1: meta_file = st.file_uploader("1. Metadata (.csv)", type=["csv"])
    with col2: ref_files = st.file_uploader("2. Muestras (.tsv)", type=["tsv"], accept_multiple_files=True)
    with col3: data_files = st.file_uploader("3. Estándares (.csv)", type=["csv"], accept_multiple_files=True)

    if st.button("🚀 Ejecutar Análisis Completo"):
        if meta_file and ref_files and data_files:
            base_path = os.path.abspath("workspace")
            if os.path.exists(base_path): shutil.rmtree(base_path)
            graphs_dir = os.path.join(base_path, "graphs"); os.makedirs(graphs_dir, exist_ok=True)
            csv_dir = os.path.join(base_path, "csv_files"); os.makedirs(csv_dir, exist_ok=True)

            # Llenar caché y disco
            st.session_state.file_cache = {}
            up_names = []
            for f in ref_files + data_files:
                fn = os.path.basename(f.name).strip()
                content = f.getvalue()
                st.session_state.file_cache[fn] = content
                up_names.append(fn)
                with open(os.path.join(base_path, fn), "wb") as out: out.write(content)

            with open(os.path.join(base_path, "metadata.csv"), "wb") as f:
                f.write(meta_file.getvalue())

            try:
                with st.spinner("Calculando espectros y tasas..."):
                    old_cwd = os.getcwd()
                    os.chdir(base_path)
                    sys.argv = ["data_toolbox.py", "-m", "metadata.csv"]
                    dt = Data(); dt.parse_args()
                    force_metadata_compatibility(dt, up_names)
                    
                    dt.read_data(); dt.sub_background(); dt.rearrange_data(); dt.group_data(); dt.conversion_rate()
                    
                    # Mapa para etiquetas de gráficos
                    m_map = dt._meta_data.set_index('Unique_Sample_ID')[['Condition_Name', 'Time_Points']].to_dict('index')
                    os.chdir(old_cwd)

                # --- GENERACIÓN DE GRÁFICOS ---
                
                # A. RATES
                st.subheader("📉 Conversión: Sustrato vs Producto")
                
                rates_path = os.path.join(csv_dir, "08_conv_rates.csv")
                if os.path.exists(rates_path):
                    df_r = pd.read_csv(rates_path)
                    for cond in df_r.iloc[:, 0].unique():
                        df_c = df_r[df_r.iloc[:, 0] == cond].sort_values(df_r.columns[1])
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(df_c.iloc[:,1], df_c.iloc[:,2], 'o-', label='Substrate', color='#1f77b4')
                        ax.plot(df_c.iloc[:,1], df_c.iloc[:,3], 's-', label='Product', color='#ff7f0e')
                        ax.set_title(f"Rates: {cond}"); ax.set_xlabel("Time (min)"); ax.legend(); ax.grid(True, alpha=0.3)
                        fig.savefig(os.path.join(graphs_dir, f"rates_{cond}.png"), bbox_inches='tight')
                        st.pyplot(fig)
                        plt.close(fig)

                # B. ESPECTROS MRP
                st.subheader("📊 Espectros MRP por Condición")
                
                master_csv = os.path.join(csv_dir, "00b_df_complete_background_subtracted_and_dropped.csv")
                if os.path.exists(master_csv):
                    df_m = pd.read_csv(master_csv)
                    wv = df_m.iloc[:, 0]
                    # Filtrar solo columnas de muestras (MRP)
                    cols = [c for c in df_m.columns[1:] if not any(f"spectrum_{l}" in c for l in "ABCDEFGH")]
                    
                    grp = {}
                    for c in cols:
                        if c in m_map:
                            cn, tp = m_map[c]['Condition_Name'], m_map[c]['Time_Points']
                            if cn not in grp: grp[cn] = []
                            grp[cn].append((tp, c))

                    for cond_name, curves in sorted(grp.items()):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for t_v, col_n in sorted(curves, key=lambda x: x[0]):
                            ax.plot(wv, df_m[col_n], label=f"{t_v} min")
                        ax.set_title(f"Espectros: {cond_name}"); ax.set_xlabel("Wavelength (nm)"); ax.legend(title="Tiempo")
                        fig.savefig(os.path.join(graphs_dir, f"espectro_{cond_name}.png"), bbox_inches='tight')
                        st.pyplot(fig)
                        plt.close(fig)

                # --- ZIP FINAL ---
                shutil.make_archive("analisis_total", 'zip', base_path)
                st.success("✅ ¡Gráficos y datos listos!")
                with open("analisis_total.zip", "rb") as f:
                    st.download_button("⬇️ Descargar Reporte Completo (ZIP)", f, "analisis_completo.zip")

            except Exception as e:
                st.error(f"Error: {e}"); st.code(traceback.format_exc())
                if 'old_cwd' in locals(): os.chdir(old_cwd)
