import streamlit as st
import os
import shutil
import sys
import pandas as pd
import hashlib
import traceback
import io

# --- 1. INTERCEPTOR DE MEMORIA (MONKEY PATCH) ---
uploaded_files_cache = {}

# Guardamos la función REAL de pandas en una variable privada que nadie más toque
from pandas.io.parsers.readers import read_csv as pandas_real_read_csv

def patched_read_csv(filepath_or_buffer, *args, **kwargs):
    # 1. Si es un string (una ruta de archivo)
    if isinstance(filepath_or_buffer, str):
        filename = os.path.basename(filepath_or_buffer)
        
        # 2. ¿Lo tenemos en la memoria (subido por el usuario)?
        if filename in uploaded_files_cache:
            # st.write(f"DEBUG: Cargando desde memoria: {filename}") # Opcional para debug
            return pandas_real_read_csv(io.BytesIO(uploaded_files_cache[filename]), *args, **kwargs)
        
        # 3. Si NO está en memoria, buscamos en el disco (solo para archivos internos del sistema)
        # Pero usamos la función REAL para evitar la recursión infinita
        return pandas_real_read_csv(filepath_or_buffer, *args, **kwargs)
    
    # 4. Si no es un string (es un buffer), usamos la función real directamente
    return pandas_real_read_csv(filepath_or_buffer, *args, **kwargs)

# Aplicamos el parche globalmente
pd.read_csv = patched_read_csv

# --- NUEVO: PARCHE DE COMPATIBILIDAD PANDAS 2.0 (DEPRECATED APPEND) ---
# El motor de Niels usa .append() que ya no existe en Pandas moderno.
# Lo re-inyectamos usando ._append interno para que el motor no falle.
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = lambda self, other, ignore_index=True: pd.concat([self, other], ignore_index=ignore_index)
# ---------------------------------------------------------------------

# --- 2. PARCHE hashlib ---
original_new = hashlib.new
def patched_new(name, data=b'', **kwargs):
    kwargs.pop('digest_size', None)
    return original_new(name, data, **kwargs)
hashlib.new = patched_new

from data_toolbox import Data

# --- 3. FUNCIÓN DE COMPATIBILIDAD DE METADATA ---
def force_metadata_compatibility(dt_instance):
    df = dt_instance._meta_data
    # Normalizar nombres de columnas que el motor pide de forma inconsistente
    mapping = {
        'Blank_Unique_Sample_ID': 'blank',
        'Unique_Sample_ID': 'Unique_Sample_ID'
    }
    df = df.rename(columns=mapping)
    df['unique_sample_id'] = df['Unique_Sample_ID']
    if 'blank' in df.columns:
        df['Blank_Unique_Sample_ID'] = df['blank']
    
    # Limpiamos las rutas en todas las celdas para que coincidan con nuestro cache
    for col in df.columns:
        df[col] = df[col].apply(lambda x: os.path.basename(x) if isinstance(x, str) and (('/' in x) or ('\\' in x)) else x)
    
    dt_instance._meta_data = df

# --- 4. INTERFAZ DE USUARIO ---
st.set_page_config(page_title="Spectral Analysis Tool", layout="wide")
st.title("🔬 Procesador Espectral (Modo Interceptor)")
st.info("Este sistema intercepta las llamadas del motor original y le entrega los archivos directamente desde la memoria.")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("1. Metadata")
    meta_file = st.file_uploader("Subir Metadata (.csv)", type=["csv"])

with col2:
    st.header("2. Estándares y Referencias")
    ref_files = st.file_uploader("Subir .tsv y .csv de REFERENCIA", type=["tsv", "csv"], accept_multiple_files=True)

with col3:
    st.header("3. Datos de la Placa")
    data_files = st.file_uploader("Subir .csv de MUESTRAS", type=["csv"], accept_multiple_files=True)

# --- 5. LÓGICA DE EJECUCIÓN ---
if st.button("🚀 Iniciar Análisis"):
    if meta_file and ref_files and data_files:
        # Limpiar y llenar el Cache de Memoria
        uploaded_files_cache.clear()
        
        # Guardamos la metadata con su nombre original y como 'metadata.csv' (que pide el motor)
        meta_bytes = meta_file.getvalue()
        uploaded_files_cache[os.path.basename(meta_file.name)] = meta_bytes
        uploaded_files_cache['metadata.csv'] = meta_bytes
        
        for f in ref_files:
            content = f.getvalue()
            uploaded_files_cache[f.name] = content
            uploaded_files_cache[os.path.basename(f.name)] = content
            
        for f in data_files:
            content = f.getvalue()
            uploaded_files_cache[f.name] = content
            uploaded_files_cache[os.path.basename(f.name)] = content

        # Crear estructura física básica (el motor la necesita para existir)
        tmp_dir = "workspace"
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
        os.makedirs(os.path.join(tmp_dir, "csv_files"), exist_ok=True)
        
        # Escribimos los archivos físicamente como respaldo
        with open(os.path.join(tmp_dir, "metadata.csv"), "wb") as f: f.write(meta_bytes)
        for f_name, f_content in uploaded_files_cache.items():
            if f_name != 'metadata.csv':
                # Si es un dato de placa, va en csv_files
                path = os.path.join(tmp_dir, "csv_files", f_name) if any(d.name == f_name for d in data_files) else os.path.join(tmp_dir, f_name)
                with open(path, "wb") as out: out.write(f_content)

        try:
            with st.spinner("El motor está procesando..."):
                old_cwd = os.getcwd()
                os.chdir(tmp_dir)
                
                # Configurar argumentos de sistema para el motor
                sys.argv = ["data_toolbox.py", "-m", "metadata.csv"]
                
                dt = Data()
                dt.parse_args()
                # INYECCIÓN DE COMPATIBILIDAD + FORZAR INCLUSIÓN
                force_metadata_compatibility(dt)
                # Forzamos a que todas las muestras sean procesadas
                dt._meta_data['Include_In_Parameter_Estimation'] = 1 
                
                dt.read_data()
                dt.sub_background()
                dt.rearrange_data()
                dt.group_data()
                
                # Ejecución del cálculo
                dt.conversion_rate() 
                
                # --- NUEVA LÓGICA DE LOCALIZACIÓN DE ARCHIVOS ---
                os.chdir(old_cwd) # Volvemos a la carpeta de la app
                
                # --- LÓGICA DE DESCARGA ZIP (TODO EL CONTENIDO) ---
                zip_path = "resultados_analisis"
                shutil.make_archive(zip_path, 'zip', tmp_dir)
                
                st.success("✅ ¡Procesamiento completado con éxito!")
                
                with open(f"{zip_path}.zip", "rb") as f:
                    st.download_button(
                        label="⬇️ Descargar Todos los Resultados (.ZIP)",
                        data=f,
                        file_name="analisis_completo.zip",
                        mime="application/zip"
                    )
                
                # Opcional: Mostrar lista de archivos generados para control
                with st.expander("Ver lista de archivos generados"):
                    st.write(os.listdir(tmp_dir) + os.listdir(os.path.join(tmp_dir, "csv_files")))

        except Exception as e:
            st.error(f"Error crítico: {e}")
            st.code(traceback.format_exc())
            if 'old_cwd' in locals(): os.chdir(old_cwd)
    else:
        st.warning("Por favor, sube todos los archivos antes de continuar.")
