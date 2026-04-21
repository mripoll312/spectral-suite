import streamlit as st
import os, shutil, sys, io, traceback, hashlib, glob
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Spectral Suite - V1.9 Carousel", layout="wide")
import matplotlib
matplotlib.use('Agg') 

# Inicializar estados para el carrusel y caché
if 'file_cache' not in st.session_state: st.session_state.file_cache = {}
if 'img_idx' not in st.session_state: st.session_state.img_idx = 0
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

# --- 2. PARCHES Y UTILIDADES (Mantenemos toda la robustez V1.8) ---
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = lambda self, other, ignore_index=True: pd.concat([self, other], ignore_index=ignore_index)

original_new = hashlib.new
def patched_new(name, data=b'', **kwargs):
    kwargs.pop('digest_size', None)
    return original_new(name, data, **kwargs)
hashlib.new = patched_new

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
    t_col = next((c for c in ['Time_Points', 'Time_Point', 'time_point', 'Time'] if c in df.columns), 'Time_Points')
    c_col = next((c for c in ['Condition_Name', 'condition_name', 'Condition'] if c in df.columns), 'Condition_Name')
    df['Time_Points'], df['Condition_Name'] = df[t_col], df[c_col]
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
st.title("🔬 Spectral Analysis Suite 2.0")
tab1, tab2 = st.tabs(["🔄 Magellan Convert", "📊 Spectral Analysis"])

with tab1:
    st.info("Convertidor Magellan: Transpone datos horizontales a formato vertical .tsv compatible.")
    uploaded_xlsx = st.file_uploader("Upload Magellan (.xlsx)", type=["xlsx"])
    
    if uploaded_xlsx:
        try:
            # 1. Leer el Excel
            df_mag = pd.read_excel(uploaded_xlsx)
            
            # Limpieza: Si la primera columna se llama '*' o similar, la usamos como índice
            first_col = df_mag.columns[0]
            df_mag = df_mag.set_index(first_col)
            
            # 2. TRANSPOSICIÓN
            df_transposed = df_mag.T
            df_transposed.index.name = 'Wavelength'
            df_transposed = df_transposed.reset_index()
            
            # Aseguramos que Wavelength sea numérico
            df_transposed['Wavelength'] = pd.to_numeric(df_transposed['Wavelength'], errors='coerce')
            df_transposed = df_transposed.dropna(subset=['Wavelength'])

            # 3. Crear la estructura completa de 96 pozos (A1 -> H12)
            filas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            columnas_placa = [f"{f}{c}" for f in filas for c in range(1, 13)]
            
            # Crear DataFrame final con Wavelength primero
            df_final = pd.DataFrame(columns=['Wavelength'] + columnas_placa)
            df_final['Wavelength'] = df_transposed['Wavelength']
            
            # 4. Mapear datos originales a sus posiciones en la placa
            for col in df_transposed.columns:
                if col in columnas_placa:
                    df_final[col] = df_transposed[col]
            
            # 5. Generar el TSV fiel al original
            tsv_buf = io.StringIO()
            df_final.to_csv(tsv_buf, sep='\t', index=False, decimal='.', float_format='%.18f')
            
            output_filename = f"conv_{uploaded_xlsx.name.replace('.xlsx', '')}.tsv"
            
            st.success(f"✅ Conversión exitosa. Se detectaron los pozos: {', '.join([c for c in df_transposed.columns if c in columnas_placa])}")
            
            st.download_button(
                label="⬇️ Descargar TSV para Análisis",
                data=tsv_buf.getvalue(),
                file_name=output_filename,
                mime="text/tab-separated-values"
            )
            
            st.subheader("Vista previa de datos convertidos")
            st.dataframe(df_final.head())

        except Exception as e:
            st.error(f"Error al procesar el Excel: {e}")
            st.code(traceback.format_exc())

with tab2:
    col_a, col_b, col_c = st.columns(3)
    with col_a: meta_file = st.file_uploader("1. Metadata (.csv)", type=["csv"])
    with col_b: ref_files = st.file_uploader("2. Samples (.tsv)", type=["tsv"], accept_multiple_files=True)
    with col_c: data_files = st.file_uploader("3. Standards (.csv)", type=["csv"], accept_multiple_files=True)

    if st.button("🚀 Execute Analysis"):
        if meta_file and ref_files and data_files:
            base_path = os.path.abspath("workspace")
            if os.path.exists(base_path): shutil.rmtree(base_path)
            graphs_dir = os.path.join(base_path, "graphs"); os.makedirs(graphs_dir, exist_ok=True)
            csv_dir = os.path.join(base_path, "csv_files"); os.makedirs(csv_dir, exist_ok=True)

            up_names = []
            for f in ref_files + data_files:
                fn = os.path.basename(f.name).strip()
                st.session_state.file_cache[fn] = f.getvalue()
                up_names.append(fn)
                with open(os.path.join(base_path, fn), "wb") as out: out.write(f.getvalue())

            # --- CAMBIO REALIZADO AQUÍ: LIMPIEZA DE METADATA ANTES DE GUARDAR ---
            try:
                # Leemos la metadata cargada
                df_meta_clean = pd.read_csv(io.BytesIO(meta_file.getvalue()))
                # Eliminamos filas que sean totalmente nulas o que no tengan nombre de condición
                df_meta_clean = df_meta_clean.dropna(how='all')
                if 'Condition_Name' in df_meta_clean.columns:
                    df_meta_clean = df_meta_clean.dropna(subset=['Condition_Name'])
                
                # Guardamos la versión limpia en el disco para el motor Data()
                df_meta_clean.to_csv(os.path.join(base_path, "metadata.csv"), index=False)
            except Exception as e:
                st.error(f"Error limpiando el archivo de Metadata: {e}")
                with open(os.path.join(base_path, "metadata.csv"), "wb") as f: f.write(meta_file.getvalue())

            try:
                with st.spinner("Processing and generating graphics..."):
                    old_cwd = os.getcwd(); os.chdir(base_path)
                    sys.argv = ["data_toolbox.py", "-m", "metadata.csv"]
                    dt = Data(); dt.parse_args()
                    force_metadata_compatibility(dt, up_names)
                    dt.read_data(); dt.sub_background(); dt.rearrange_data(); dt.group_data(); dt.conversion_rate()
                    
                    df_meta = dt._meta_data.copy()
                    m_map = df_meta.set_index('Unique_Sample_ID')[['Condition_Name', 'Time_Points']].to_dict('index')
                    os.chdir(old_cwd)

                # --- GENERACIÓN DE IMÁGENES ---
                rates_csv = os.path.join(csv_dir, "08_conv_rates.csv")
                if os.path.exists(rates_csv):
                    df_rates = pd.read_csv(rates_csv, index_col=0)
                    for cond, group in df_rates.groupby('Condition_Name'):
                        group = group.sort_values(by='Time_Point')
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(group['Time_Point'], group['Substrate']*100, 'o-', label='Substrate', color='#1f77b4')
                        ax.plot(group['Time_Point'], group['Product']*100, 's-', label='Product', color='#ff7f0e')
                        ax.set_title(f"Kinetic: {cond}"); ax.set_ylabel("Conversion (%)"); ax.set_ylim(-5, 105); ax.legend(); ax.grid(True, alpha=0.3)
                        fig.savefig(os.path.join(graphs_dir, f"kinetic_{cond}.png"), bbox_inches='tight'); plt.close(fig)

                master_csv = os.path.join(csv_dir, "00b_df_complete_background_subtracted_and_dropped.csv")
                if os.path.exists(master_csv):
                    df_m = pd.read_csv(master_csv); wv = df_m.iloc[:, 0]
                    cols = [c for c in df_m.columns[1:] if not any(f"spectrum_{l}" in c for l in "ABCDEFGH")]
                    grp = {}
                    for c in cols:
                        if c in m_map:
                            cn, tp = m_map[c]['Condition_Name'], m_map[c]['Time_Points']
                            if cn not in grp: grp[cn] = []
                            grp[cn].append((tp, c))
                    for cond_name, curves in grp.items():
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for t_v, col_n in sorted(curves): ax.plot(wv, df_m[col_n], label=f"{t_v} min")
                        ax.set_title(f"Spectrum: {cond_name}"); ax.legend(); ax.grid(True, alpha=0.3)
                        fig.savefig(os.path.join(graphs_dir, f"spectrum_{cond_name}.png"), bbox_inches='tight'); plt.close(fig)

                shutil.make_archive("analisis_v1.9", 'zip', base_path)
                st.session_state.analysis_ready = True
                st.session_state.img_idx = 0

            except Exception as e:
                st.error(f"Error: {e}"); st.code(traceback.format_exc())
                if 'old_cwd' in locals(): os.chdir(old_cwd)

    if st.session_state.analysis_ready:
        st.divider()
        st.subheader("Results")
        img_list = sorted(glob.glob("workspace/graphs/*.png"))
        
        if img_list:
            total_imgs = len(img_list)
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                if st.button("⬅️ Previous"):
                    st.session_state.img_idx = (st.session_state.img_idx - 1) % total_imgs
            with c3:
                if st.button("Next ➡️"):
                    st.session_state.img_idx = (st.session_state.img_idx + 1) % total_imgs
            
            current_img_path = img_list[st.session_state.img_idx]
            file_name = os.path.basename(current_img_path)
            with c2:
                st.markdown(f"<p style='text-align: center;'><b>Archivo {st.session_state.img_idx + 1} de {total_imgs}:</b> {file_name}</p>", unsafe_allow_html=True)
                st.image(current_img_path, use_container_width=True)
        
        st.divider()
        with open("analisis_v1.9.zip", "rb") as f:
            st.download_button("⬇️ Download as Zip", f, "completeAnalysis.zip")