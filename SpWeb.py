import streamlit as st
import os
import shutil
import sys
import pandas as pd
import hashlib
import traceback
import io

# --- 1. INTERCEPTOR DE MEMORIA (PROTEGIDO) ---
from pandas.io.parsers.readers import read_csv as _original_pandas_read_func

if 'file_cache' not in st.session_state:
    st.session_state.file_cache = {}

def patched_read_csv(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, str):
        requested_name = os.path.basename(filepath_or_buffer).strip()
        if requested_name in st.session_state.file_cache:
            return _original_pandas_read_func(io.BytesIO(st.session_state.file_cache[requested_name]), *args, **kwargs)
        
        # Búsqueda difusa (insensible a mayúsculas)
        for cached_name, content in st.session_state.file_cache.items():
            if requested_name.lower() == cached_name.lower():
                return _original_pandas_read_func(io.BytesIO(content), *args, **kwargs)

    return _original_pandas_read_func(filepath_or_buffer, *args, **kwargs)

pd.read_csv = patched_read_csv

# --- PARCHES DE COMPATIBILIDAD ---
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = lambda self, other, ignore_index=True: pd.concat([self, other], ignore_index=ignore_index)

original_new = hashlib.new
def patched_new(name, data=b'', **kwargs):
    kwargs.pop('digest_size', None)
    return original_new(name, data, **kwargs)
hashlib.new = patched_new

from data_toolbox import Data

# --- 3. FUNCIÓN DE COMPATIBILIDAD DE METADATA ---
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

# --- 4. INTERFAZ ---
st.set_page_config(page_title="Spectral Analysis Tool", layout="wide")
st.title("🔬 Procesador Espectral")

col1, col2, col3 = st.columns(3)
with col1:
    meta_file = st.file_uploader("1. Metadata (.csv)", type=["csv"])
with col2:
    ref_files = st.file_uploader("2. Estándares (.tsv)", type=["tsv", "csv"], accept_multiple_files=True)
with col3:
    data_files = st.file_uploader("3. Muestras (.csv)", type=["csv"], accept_multiple_files=True)

if st.button("🚀 Iniciar Análisis"):
    if meta_file and ref_files and data_files:
        # A. PREPARAR CACHÉ
        st.session_state.file_cache = {}
        meta_bytes = meta_file.getvalue()
        st.session_state.file_cache[os.path.basename(meta_file.name)] = meta_bytes
        st.session_state.file_cache['metadata.csv'] = meta_bytes
        
        for f in ref_files:
            st.session_state.file_cache[os.path.basename(f.name)] = f.getvalue()
        for f in data_files:
            st.session_state.file_cache[os.path.basename(f.name)] = f.getvalue()

        # B. PREPARAR DIRECTORIOS (ANTES de escribir archivos)
        tmp_dir = "workspace"
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
        os.makedirs(os.path.join(tmp_dir, "csv_files"), exist_ok=True)

        # C. ESCRIBIR ARCHIVOS FÍSICOS (Usando el caché recién llenado)
        for f_name, f_content in st.session_state.file_cache.items():
            # Ruta base
            path_raiz = os.path.join(tmp_dir, f_name)
            with open(path_raiz, "wb") as out:
                out.write(f_content)
            
            # Si es una muestra, también va en csv_files
            if any(d.name == f_name for d in data_files):
                path_data = os.path.join(tmp_dir, "csv_files", f_name)
                with open(path_data, "wb") as out:
                    out.write(f_content)

        # D. EJECUCIÓN
        try:
            with st.spinner("Ejecutando motor de Niels..."):
                old_cwd = os.getcwd()
                os.chdir(tmp_dir)
                
                sys.argv = ["data_toolbox.py", "-m", "metadata.csv"]
                dt = Data()
                dt.parse_args()
                force_metadata_compatibility(dt)
                dt._meta_data['Include_In_Parameter_Estimation'] = 1 
                
                dt.read_data()
                dt.sub_background()
                dt.rearrange_data()
                dt.group_data()
                dt.conversion_rate() 
                
                os.chdir(old_cwd)
                
                # ZIP y Descarga
                zip_path = "resultados_analisis"
                shutil.make_archive(zip_path, 'zip', tmp_dir)
                st.success("✅ ¡Completado!")
                
                with open(f"{zip_path}.zip", "rb") as f:
                    st.download_button("⬇️ Descargar ZIP", f, "analisis.zip", "application/zip")

        except Exception as e:
            st.error(f"Error crítico: {e}")
            st.code(traceback.format_exc())
            if 'old_cwd' in locals(): os.chdir(old_cwd)
    else:
        st.warning("Faltan archivos.")
