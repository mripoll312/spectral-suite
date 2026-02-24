import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import zipfile

# Configuración de página
st.set_page_config(
    page_title="Spectral Analysis Suite PRO",
    page_icon="🧪",
    layout="wide"
)

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #fcfcfc; }
    div.stButton > button {
        width: 100%; border-radius: 8px; height: 3.5em;
        background-color: #004b87; color: white; font-weight: bold; border: none;
    }
    div.stButton > button:hover { background-color: #007bff; color: white; }
    .stTabs [aria-selected="true"] { background-color: #004b87 !important; color: white !important; }
    [data-testid="stSidebar"] { background-color: #002b4d; color: white; }
    [data-testid="stSidebar"] * { color: white !important; }
    .sidebar-section { 
        background-color: rgba(255,255,255,0.1); 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        border-left: 5px solid #007bff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE PROCESAMIENTO ---

def convertir_formato_matriz(uploaded_file):
    try:
        df_orig = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df_orig = df_orig.set_index(df_orig.columns[0])
        df_transposed = df_orig.T
        df_transposed.index = df_transposed.index.str.replace('nm', '', case=False).astype(int)
        filas, cols = ['A','B','C','D','E','F','G','H'], range(1, 13)
        todas = [f"{f}{c}" for f in filas for c in cols]
        df_final = pd.DataFrame(index=df_transposed.index, columns=todas)
        for col in todas:
            df_final[col] = df_transposed[col] if col in df_transposed.columns else ""
        return df_final.reset_index().rename(columns={'index': 'Wavelength'})
    except Exception as e:
        st.error(f"Error en conversión: {e}")
        return None

def procesar_data_toolbox(df_meta, dict_archivos):
    base_datos = {}
    for nombre, contenido in dict_archivos.items():
        try: base_datos[nombre] = pd.read_csv(io.BytesIO(contenido), sep='\t').dropna(how='all')
        except: continue
    resultados, espectros_procesados = [], {}
    for _, row in df_meta.iterrows():
        sample_id = str(row['Unique_Sample_ID'])
        try:
            f_name, well, blank_id = str(row['Data_File']), str(row['Well_Number']), str(row['Blank_Unique_Sample_ID'])
            if f_name in base_datos and well in base_datos[f_name].columns:
                df_spec = base_datos[f_name]
                waves = df_spec.iloc[:, 0].values
                abs_sample = df_spec[well].values
                abs_final = abs_sample
                meta_blank = df_meta[df_meta['Unique_Sample_ID'] == blank_id]
                if not meta_blank.empty:
                    b_f, b_w = str(meta_blank.iloc[0]['Data_File']), str(meta_blank.iloc[0]['Well_Number'])
                    if b_f in base_datos and b_w in base_datos[b_f].columns:
                        abs_final = abs_sample - base_datos[b_f][b_w].values
                espectros_procesados[sample_id] = (waves, abs_final)
                w_s, w_e = row['Wavelength_Start'], row['Wavelength_End']
                v_s = abs_final[waves == w_s][0] if any(waves == w_s) else 0
                v_e = abs_final[waves == w_e][0] if any(waves == w_e) else 0
                resultados.append({
                    "Sample_ID": sample_id, "Condition": row['Condition_Name'],
                    f"Abs_{w_s}": round(v_s, 4), f"Abs_{w_e}": round(v_e, 4),
                    "Ratio": round(v_s / v_e, 4) if v_e != 0 else 0
                })
        except: continue
    return pd.DataFrame(resultados), espectros_procesados

def generar_pdf(df_res, specs):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16); pdf.cell(200, 20, "Reporte de Analisis Espectral", ln=True, align='C')
    for _, row in df_res.iterrows():
        pdf.set_font("Arial", '', 9)
        pdf.cell(200, 7, f"{row['Sample_ID']} | {row['Condition']} | Ratio: {row['Ratio']}", ln=True)
    for sid, (w, a) in specs.items():
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12); pdf.cell(200, 10, f"Muestra: {sid}", ln=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(w, a, color='#004b87'); ax.grid(True, linestyle='--')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, dpi=100); pdf.image(tmp.name, x=15, y=30, w=180)
        plt.close(fig)
    return pdf.output(dest='S').encode('latin-1')

def generar_zip(df_res, specs):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("resultados_analisis.csv", df_res.to_csv(index=False))
        for sid, (w, a) in specs.items():
            df_ind = pd.DataFrame({'Wavelength': w, 'Absorbance': a})
            zf.writestr(f"data/{sid}_data.csv", df_ind.to_csv(index=False))
            fig, ax = plt.subplots(); ax.plot(w, a); ax.set_title(sid); ax.grid(True)
            img_buf = io.BytesIO(); fig.savefig(img_buf, format='png'); plt.close(fig)
            zf.writestr(f"plots/{sid}_plot.png", img_buf.getvalue())
    return buf.getvalue()

# --- BARRA LATERAL ESTÁTICA (LOGO) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Logo_ORT_Uruguay.png/800px-Logo_ORT_Uruguay.png", width=200)
    st.markdown("### Spectral Suite v2.2")
    st.caption("Facultad de Ingeniería")
    st.divider()

# --- CUERPO PRINCIPAL CON PESTAÑAS ---
tab1, tab2 = st.tabs(["🔄 1. Convertidor Magellan", "📊 2. Análisis & Exportación"])

# --- CONTENIDO PESTAÑA 1 ---
with tab1:
    # Instrucciones específicas para Tab 1 en la Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("📖 Guía: Convertidor")
        st.write("""
        **Funcionalidad:** Prepara tus datos. Magellan exporta las longitudes de onda en filas; esta herramienta las transpone a columnas para el análisis masivo.
        
        **Pasos:**
        1. Sube el **Excel/CSV** original.
        2. Revisa la tabla generada.
        3. Descarga el archivo **.tsv**.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.header("🔄 Convertidor de Formato Magellan")
    file_raw = st.file_uploader("Subir archivo de Magellan (Excel/CSV)", type=["xlsx", "csv"], key="uploader_tab1")
    if file_raw:
        df_conv = convertir_formato_matriz(file_raw)
        if df_conv is not None:
            st.success("Matriz generada con éxito.")
            st.dataframe(df_conv.head(10), width='stretch')
            output = io.StringIO()
            df_conv.to_csv(output, sep='\t', index=False, na_rep="")
            st.download_button("💾 Descargar Espectro (.tsv)", output.getvalue(), file_raw.name.split('.')[0] + "_Spectrum.tsv", key="btn_tab1")

# --- CONTENIDO PESTAÑA 2 ---
with tab2:
    # Instrucciones específicas para Tab 2 en la Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("📖 Guía: Data Toolbox")
        st.write("""
        **Funcionalidad:** Procesa múltiples archivos simultáneamente aplicando resta de blancos y cálculo de ratios.
        
        **Pasos:**
        1. Sube el CSV de **Metadata**.
        2. Sube todos los archivos **.tsv** (Paso 1).
        3. Pulsa **Ejecutar**.
        4. Descarga el **Reporte PDF** o el **ZIP** con gráficos PNG.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.header("📊 Análisis Masivo & Reportes")
    c1, c2 = st.columns(2)
    with c1: meta_up = st.file_uploader("1. Subir Metadata (CSV)", type="csv", key="meta_tab2")
    with c2: specs_up = st.file_uploader("2. Subir Espectros (.tsv)", type="tsv", accept_multiple_files=True, key="specs_tab2")

    if meta_up and specs_up:
        df_meta = pd.read_csv(meta_up, sep=None, engine='python')
        df_meta.columns = df_meta.columns.str.strip()
        dict_archivos = {f.name: f.getvalue() for f in specs_up}
        
        if st.button("🚀 EJECUTAR PROCESAMIENTO", key="btn_run_tab2"):
            res, specs = procesar_data_toolbox(df_meta, dict_archivos)
            st.session_state['res_t2'] = res
            st.session_state['specs_t2'] = specs

        if 'res_t2' in st.session_state:
            st.divider()
            st.subheader("📈 Ratios Calculados")
            st.dataframe(st.session_state['res_t2'], width='stretch')
            
            # Visualizador interactivo
            muestras = list(st.session_state['specs_t2'].keys())
            st.subheader("🔍 Visualizador de Curvas")
            sel = st.selectbox("Muestra para previsualizar:", muestras, key="select_viz")
            if sel:
                w_v, a_v = st.session_state['specs_t2'][sel]
                fig_v, ax_v = plt.subplots(figsize=(10, 4))
                ax_v.plot(w_v, a_v, color='#004b87', linewidth=2)
                ax_v.set_title(f"Muestra: {sel}")
                ax_v.set_xlabel("nm")
                ax_v.set_ylabel("Abs")
                ax_v.grid(True, alpha=0.3)
                st.pyplot(fig_v)
            
            st.divider()
            st.subheader("📥 Exportar")
            col_pdf, col_zip = st.columns(2)
            with col_pdf:
                pdf_bytes = generar_pdf(st.session_state['res_t2'], st.session_state['specs_t2'])
                st.download_button("📄 Reporte PDF", pdf_bytes, "Reporte.pdf", "application/pdf", key="btn_pdf")
            with col_zip:
                zip_bytes = generar_zip(st.session_state['res_t2'], st.session_state['specs_t2'])
                st.download_button("📦 Paquete ZIP", zip_bytes, "Resultados.zip", "application/zip", key="btn_zip")