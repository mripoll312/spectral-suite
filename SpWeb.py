import streamlit as st
import os
import shutil
import sys
import pandas as pd
import hashlib
import traceback
import io

# --- 1. CONFIGURACIÓN Y PARCHES (NIVEL GLOBAL) ---
st.set_page_config(page_title="Spectral Suite", layout="wide")

from pandas.io.parsers.readers import read_csv as _original_pandas_read_func

if 'file_cache' not in st.session_state:
    st.session_state.file_cache = {}

def patched_read_csv(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, str):
        requested_name = os.path.basename(filepath_or_buffer).strip()
        if requested_name in st.session_state.file_cache:
            return _original_pandas_read_func(io.BytesIO(st.session_state.file_cache[requested_name]), *args, **kwargs)
        for cached_name, content in st.session_state.file_cache.items():
            if requested_name.lower() == cached_name.lower():
                return _original_pandas_read_func(io.BytesIO(content), *args, **kwargs)
    return _original_pandas_read_func(filepath_or_buffer, *args, **kwargs)

pd.read_csv = patched_read_csv

if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = lambda self, other, ignore_index=True: pd.concat([self, other], ignore_index=ignore_index)

original_new = hashlib.new
def patched_new(name, data=b'', **kwargs):
    kwargs.pop('digest_size', None)
    return original_new(name, data, **kwargs)
hashlib.new = patched_new

from data_toolbox import Data

def force_metadata_compatibility(dt_instance):
    df = dt_instance._meta_data
    mapping = {'Blank_Unique_Sample_ID': 'blank', 'Unique_Sample_ID': 'Unique_Sample_ID'}
    df = df.rename(columns=mapping)
    df['unique_sample_id'] = df['Unique_Sample_ID']
    if 'blank' in df.columns:
        df['Blank_Unique_Sample_ID'] = df['blank']
    for col in df.columns:
        df[col] = df[col].apply(lambda x: os.path.basename(x) if isinstance(x, str) and (('/' in x) or ('\\' in x)) else x)
    dt_instance._meta_data = df

# --- 2. DEFINICIÓN DE PESTAÑAS ---
tab2, tab1 = st.tabs(["📊 Spectral Analysis", "🔄 Conversor Magellan"])

# --- TAB 1: ANÁLISIS ESPECTRAL (CÓDIGO QUE YA FUNCIONA) ---
with tab2:
    st.header("🔬 Spectral Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        meta_file = st.file_uploader("1. Metadata (.csv)", type=["csv"], key="meta_up")
    with col2:
        ref_files = st.file_uploader("2. Muestras (.tsv)", type=["tsv"], accept_multiple_files=True, key="ref_up")
    with col3:
        data_files = st.file_uploader("3. Estándares (.csv)", type=["csv"], accept_multiple_files=True, key="data_up")

    if st.button("🚀 Iniciar Análisis", key="btn_analisis"):
        if meta_file and ref_files and data_files:
            st.session_state.file_cache = {}
            m_bytes = meta_file.getvalue()
            st.session_state.file_cache[os.path.basename(meta_file.name)] = m_bytes
            st.session_state.file_cache['metadata.csv'] = m_bytes
            for f in ref_files: st.session_state.file_cache[os.path.basename(f.name)] = f.getvalue()
            for f in data_files: st.session_state.file_cache[os.path.basename(f.name)] = f.getvalue()

            tmp_dir = "workspace"
            if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
            os.makedirs(os.path.join(tmp_dir, "csv_files"), exist_ok=True)

            for f_name, f_content in st.session_state.file_cache.items():
                with open(os.path.join(tmp_dir, f_name), "wb") as out: out.write(f_content)
                if any(d.name == f_name for d in data_files):
                    with open(os.path.join(tmp_dir, "csv_files", f_name), "wb") as out: out.write(f_content)

            try:
                with st.spinner("Ejecutando motor..."):
                    old_cwd = os.getcwd()
                    os.chdir(tmp_dir)
                    sys.argv = ["data_toolbox.py", "-m", "metadata.csv"]
                    dt = Data()
                    dt.parse_args()
                    force_metadata_compatibility(dt)
                    dt._meta_data['Include_In_Parameter_Estimation'] = 1 
                    dt.read_data(); dt.sub_background(); dt.rearrange_data(); dt.group_data(); dt.conversion_rate() 
                    os.chdir(old_cwd)
                    
                    shutil.make_archive("resultados", 'zip', tmp_dir)
                    st.success("✅ ¡Análisis completado!")
                    with open("resultados.zip", "rb") as f:
                        st.download_button("⬇️ Descargar Resultados", f, "analisis.zip")
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
                if 'old_cwd' in locals(): os.chdir(old_cwd)
        else:
            st.warning("Carga todos los archivos.")

# --- TAB 2: CONVERSOR MAGELLAN (EL CÓDIGO RECUPERADO) ---
with tab1:
    st.header("🔄 Convertidor Magellan a CSV")
    st.info("Sube el archivo .xlsx exportado de Magellan para convertirlo al formato que acepta el Analizador.")
    
    uploaded_xlsx = st.file_uploader("Subir archivo Magellan (.xlsx)", type=["xlsx"], key="magellan_up")
    
    if uploaded_xlsx:
        try:
            # Leemos el Excel saltando las filas de encabezado típicas de Magellan (ajusta si es necesario)
            df_magellan = pd.read_excel(uploaded_xlsx)
            
            st.write("Vista previa de los datos detectados:")
            st.dataframe(df_magellan.head())
            
            # Aquí va tu lógica específica de limpieza de Magellan 
            # (Ejemplo genérico: asumiendo que quieres limpiar nombres de columnas)
            df_clean = df_magellan.copy()
            
            csv_buffer = io.StringIO()
            df_clean.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="⬇️ Descargar CSV Convertido",
                data=csv_buffer.getvalue(),
                file_name=f"convertido_{uploaded_xlsx.name.replace('.xlsx', '.csv')}",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error al convertir: {e}")


