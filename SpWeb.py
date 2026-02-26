import streamlit as st
import os, shutil, sys, io, traceback, hashlib
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
st.set_page_config(page_title="Spectral Suite - Ultimate Fix", layout="wide")
import matplotlib
matplotlib.use('Agg') 

if 'file_cache' not in st.session_state:
    st.session_state.file_cache = {}

# --- 2. PARCHES DE COMPATIBILIDAD ---
# Parche para hashlib (compatibilidad Python 3.13+)
original_new = hashlib.new
def patched_new(name, data=b'', **kwargs):
    kwargs.pop('digest_size', None)
    return original_new(name, data, **kwargs)
hashlib.new = patched_new

# Parche de Pandas READ_CSV: Intercepta llamadas del motor a archivos en memoria
from pandas.io.parsers.readers import read_csv as _original_pandas_read_func
def patched_read_csv(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, str):
        fname = os.path.basename(filepath_or_buffer).strip()
        if 'file_cache' in st.session_state and fname in st.session_state.file_cache:
            return _original_pandas_read_func(io.BytesIO(st.session_state.file_cache[fname]), *args, **kwargs)
    return _original_pandas_read_func(filepath_or_buffer, *args, **kwargs)
pd.read_csv = patched_read_csv

def force_metadata_compatibility(dt_instance, uploaded_filenames):
    """
    Sincroniza los nombres de la metadata con los archivos reales subidos
    para evitar errores de mayúsculas/minúsculas o rutas.
    """
    df = dt_instance._meta_data
    
    # Normalizar columnas de tiempo y condición
    for target, options in {'Time_Points': ['Time_Point', 'time_point'], 'Condition_Name': ['condition_name']}.items():
        col = next((c for c in options if c in df.columns), None)
        if col: df[target] = df[col]

    # FIX CRÍTICO: Mapeo de nombres de archivos
    # Si la metadata dice "archivo_PPi.tsv" pero subiste "archivo_Ppi.tsv", esto lo arregla.
    def find_correct_name(val):
        if not isinstance(val, str) or ('.tsv' not in val and '.csv' not in val):
            return val
        base_val = os.path.basename(val).strip().lower()
        for real_name in uploaded_filenames:
            if real_name.lower() == base_val:
                return real_name
        return os.path.basename(val).strip()

    for col in df.columns:
        df[col] = df[col].apply(find_correct_name)
    
    dt_instance._meta_data = df

from data_toolbox import Data

# --- 3. INTERFAZ ---
st.title("🔬 Spectral Analysis Suite")
tab1, tab2 = st.tabs(["🔄 Conversor", "📊 Análisis"])

with tab1:
    uploaded_xlsx = st.file_uploader("Subir Magellan (.xlsx)", type=["xlsx"])
    if uploaded_xlsx:
        df_mag = pd.read_excel(uploaded_xlsx)
        csv_buf = io.StringIO()
        df_mag.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Descargar CSV", csv_buf.getvalue(), f"conv_{uploaded_xlsx.name}.csv")

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1: meta_file = st.file_uploader("Metadata (.csv)", type=["csv"])
    with col2: ref_files = st.file_uploader("Muestras (.tsv)", type=["tsv"], accept_multiple_files=True)
    with col3: data_files = st.file_uploader("Estándares (.csv)", type=["csv"], accept_multiple_files=True)

    if st.button("🚀 Ejecutar Análisis"):
        if meta_file and ref_files and data_files:
            base_path = os.path.abspath("workspace")
            if os.path.exists(base_path): shutil.rmtree(base_path)
            graphs_dir = os.path.join(base_path, "graphs"); os.makedirs(graphs_dir, exist_ok=True)
            csv_dir = os.path.join(base_path, "csv_files"); os.makedirs(csv_dir, exist_ok=True)

            # Guardar archivos y llenar caché
            st.session_state.file_cache = {}
            all_uploads = ref_files + data_files
            uploaded_names = []

            for f in all_uploads:
                fname = os.path.basename(f.name).strip()
                content = f.getvalue()
                st.session_state.file_cache[fname] = content
                uploaded_names.append(fname)
                with open(os.path.join(base_path, fname), "wb") as out:
                    out.write(content)

            with open(os.path.join(base_path, "metadata.csv"), "wb") as f:
                f.write(meta_file.getvalue())

            try:
                with st.spinner("Procesando..."):
                    old_cwd = os.getcwd()
                    os.chdir(base_path)
                    
                    sys.argv = ["data_toolbox.py", "-m", "metadata.csv"]
                    dt = Data()
                    dt.parse_args()
                    # Aplicamos la corrección de nombres antes de leer los datos
                    force_metadata_compatibility(dt, uploaded_names)
                    
                    dt.read_data(); dt.sub_background(); dt.rearrange_data(); dt.group_data(); dt.conversion_rate()
                    
                    meta_map = dt._meta_data.set_index('Unique_Sample_ID')[['Condition_Name', 'Time_Points']].to_dict('index')
                    os.chdir(old_cwd)

                # --- GRÁFICOS ---
                
                # (Aquí va la lógica de gráficos de Rates y Espectros que teníamos antes...)
                # Asegúrate de usar fig.savefig(os.path.join(graphs_dir, "nombre.png"))
                
                # --- ZIP ---
                shutil.make_archive("resultados", 'zip', base_path)
                st.success("✅ ¡Análisis completado!")
                with open("resultados.zip", "rb") as f:
                    st.download_button("⬇️ Descargar ZIP", f, "resultados_completos.zip")

            except Exception as e:
                st.error(f"Error: {e}"); st.code(traceback.format_exc())
                if 'old_cwd' in locals(): os.chdir(old_cwd)
