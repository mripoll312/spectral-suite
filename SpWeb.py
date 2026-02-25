import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import nnls
import io
import os
import zipfile
import tempfile
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuración para servidor (sin display)
mpl.use('Agg')
plt.ioff()

# ────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL DE PROCESAMIENTO
# ────────────────────────────────────────────────────────────────
class DataProcessor:
    def __init__(self, metadata_df, spectra_dict, output_path, ref_dict):
        self.meta = metadata_df
        self.spectra = spectra_dict
        self.output_path = output_path
        self.ref_dict = ref_dict
        self.csv_dir = os.path.join(output_path, "csv_files")
        self.plot_dir = os.path.join(output_path, "plots")
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def run_original_logic(self):
        # 1. Leer espectros experimentales (TSV de placa)
        dfs = []
        for name, content in self.spectra.items():
            try:
                df = pd.read_csv(io.BytesIO(content), sep='\t')
                df = df.dropna(axis=1, how='all')

                # Renombrar columna de longitud de onda de forma tolerante
                wave_candidates = [c for c in df.columns if 'wave' in str(c).lower() or str(c).isdigit()]
                if wave_candidates:
                    df = df.rename(columns={wave_candidates[0]: 'Wavelength'})

                if 'Wavelength' in df.columns:
                    well_cols = [c for c in df.columns if c != 'Wavelength' and str(c).strip()]
                    if not well_cols:
                        st.warning(f"Archivo {name}: no se detectaron columnas de pozos (A1, B2, etc.)")
                        continue
                    df_melted = df.melt(id_vars=['Wavelength'],
                                        value_vars=well_cols,
                                        var_name='Well',
                                        value_name='Absorbance')
                    df_melted['Data_File'] = name
                    dfs.append(df_melted)
                else:
                    st.warning(f"Archivo {name}: no tiene columna Wavelength")
            except Exception as e:
                st.error(f"Error leyendo espectro {name}: {str(e)}")

        if not dfs:
            st.error("❌ No se pudieron leer ninguno de los archivos de espectros.")
            return

        # Debug: mostrar primer archivo procesado
        st.subheader("Debug — Primer espectro leído")
        primer = dfs[0]
        st.write(f"**Archivo:** {primer['Data_File'].iloc[0]}")
        st.write(f"**Forma:** {primer.shape}")
        st.dataframe(primer.head(12))
        pozos_unicos = sorted(primer['Well'].unique())
        st.write(f"**Pozos detectados ({len(pozos_unicos)}):** {pozos_unicos[:25]}")

        full_data = pd.concat(dfs, ignore_index=True)
        full_data['Wavelength'] = pd.to_numeric(full_data['Wavelength'], errors='coerce').round(0).astype(float)
        full_data['Absorbance'] = pd.to_numeric(full_data['Absorbance'], errors='coerce')
        full_data = full_data.dropna(subset=['Wavelength', 'Absorbance', 'Well', 'Data_File'])

        full_data.to_csv(os.path.join(self.csv_dir, "01_raw_spectral_data.csv"), index=False)

        # Normalización agresiva de nombres de archivo
        def normalize_filename(s):
            if pd.isna(s):
                return ""
            s = str(s).strip().lower()
            s = s.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            s = s.replace(".tsv", "").replace(".csv", "")
            return s

        full_data['Data_File_norm'] = full_data['Data_File'].apply(normalize_filename)

        # 2. Identificar columnas de referencia
        col_s_list = [c for c in self.meta.columns if 'substrate' in c.lower() and ('ref' in c.lower() or 'spectrum' in c.lower())]
        col_p_list = [c for c in self.meta.columns if 'product' in c.lower() and ('ref' in c.lower() or 'spectrum' in c.lower())]

        if not col_s_list or not col_p_list:
            st.error("No se encontraron columnas de referencia (Substrate/Product Ref/Spectrum)")
            return

        col_s, col_p = col_s_list[0], col_p_list[0]

        # Detectar columna de well automáticamente
        well_col = next((c for c in self.meta.columns if 'well' in c.lower() or 'pozo' in c.lower()), None)
        if well_col is None:
            st.error("No se encontró columna de Well / Well_Number / Pozo en metadata")
            return
        st.info(f"Columna de pocillos detectada: **{well_col}**")

        res_list = []

        for idx, row in self.meta.iterrows():
            file_name_raw = row.get('Data_File', '')
            file_name_norm = normalize_filename(file_name_raw)

            well_name = str(row.get(well_col, '')).strip().upper()

            exp_spectrum = full_data[
                (full_data['Data_File_norm'] == file_name_norm) &
                (full_data['Well'].str.strip().str.upper() == well_name)
            ].copy()

            if exp_spectrum.empty:
                st.warning(f"No datos encontrados → {row.get('Condition_Name','?')} | File: {file_name_raw} | Well: {well_name}")
                substrate_conc = product_conc = 0.0
            else:
                ref_s_name = os.path.basename(str(row.get(col_s, ''))).strip().lower() if pd.notnull(row.get(col_s)) else None
                ref_p_name = os.path.basename(str(row.get(col_p, ''))).strip().lower() if pd.notnull(row.get(col_p)) else None

                df_s = self.ref_dict.get(ref_s_name)
                df_p = self.ref_dict.get(ref_p_name)

                if df_s is None or df_p is None:
                    st.warning(f"Referencia faltante para {row.get('Condition_Name','?')}")
                    substrate_conc = product_conc = 0.0
                else:
                    m1 = pd.merge(exp_spectrum, df_s[['Wavelength', 'Abs']], on='Wavelength', how='inner')
                    merged = pd.merge(m1, df_p[['Wavelength', 'Abs']], on='Wavelength', how='inner', suffixes=('_S', '_P'))
                    final_df = merged.dropna(subset=['Abs_S', 'Abs_P', 'Absorbance'])

                    if final_df.empty:
                        st.warning(f"Sin solapamiento de longitudes de onda → {well_name}")
                        substrate_conc = product_conc = 0.0
                    else:
                        A = final_df[['Abs_S', 'Abs_P']].values
                        y = final_df['Absorbance'].values
                        sol, rnorm = nnls(A, y)
                        substrate_conc, product_conc = sol[0], sol[1]
                        st.success(f"✓ {row.get('Condition_Name','?')} | {well_name} → S={substrate_conc:.4f}  P={product_conc:.4f}")

            res_list.append({
                'Condition_Name': row.get('Condition_Name'),
                'Time_Point': row.get('Time_Point'),
                'Substrate': substrate_conc,
                'Product': product_conc,
                'Well': well_name,
                'Data_File': file_name_raw
            })

        # Guardar resultado
        res_final = pd.DataFrame(res_list)
        res_final.to_csv(os.path.join(self.csv_dir, "08_conversion_rates.csv"), index=False)

        # Generar gráficos
        for i, cond in enumerate(res_final['Condition_Name'].unique()):
            cond_df = res_final[res_final['Condition_Name'] == cond].sort_values('Time_Point')
            self.generate_all_plots(cond, cond_df, i + 1)

        # Limpieza columna temporal
        full_data.drop(columns=['Data_File_norm'], errors='ignore', inplace=True)

    def generate_all_plots(self, name, df, idx):
        if df.empty:
            return

        # 1. Substrate + Product
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df['Time_Point'], df['Substrate'], 'r-o', label='Substrate')
        ax.plot(df['Time_Point'], df['Product'], 'b-o', label='Product')
        ax.set_title(f"{name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.plot_dir, f"01_condition_{idx:02d}.png"), dpi=140, bbox_inches='tight')
        plt.close(fig)

        # 2. Solo Product (cinética)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df['Time_Point'], df['Product'], 'b-o', markersize=5)
        ax.set_title(f"Producto – {name}")
        ax.set_xlabel("Time point")
        ax.set_ylabel("Concentración producto")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.plot_dir, f"02_product_{idx:02d}.png"), dpi=140, bbox_inches='tight')
        plt.close(fig)

        # 3. Barras finales
        fig, ax = plt.subplots(figsize=(6, 5))
        last = df.iloc[-1]
        ax.bar(['Sustrato', 'Producto'], [last['Substrate'], last['Product']],
               color=['#e74c3c', '#3498db'])
        ax.set_title(f"Final – {name}")
        ax.set_ylabel("Concentración final")
        fig.savefig(os.path.join(self.plot_dir, f"03_barras_{idx:02d}.png"), dpi=140, bbox_inches='tight')
        plt.close(fig)


# ────────────────────────────────────────────────────────────────
# INTERFAZ STREAMLIT
# ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spectral Analysis & Biotransformation",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .custom-header {
        background: #f8f9fa;
        padding: 1.5rem;
        border-bottom: 2px solid #dee2e6;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .custom-header h2 {
        margin: 0;
        color: #2c3e50;
    }
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    div.stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-header"><h2>🧪 Spectral Analysis & Biotransformation Toolbox</h2></div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://png.pngtree.com/png-clipart/20241117/original/pngtree-bacteria-illustration-png-image_17164212.png", width=90)
    st.markdown("<h4 style='text-align:center; color:#64748b;'>Workstation Control</h4>", unsafe_allow_html=True)
    st.info("PhD Engine v2.1.5 – Kinetic regression optimized")

tab1, tab2 = st.tabs(["Converter (Magellan → TSV)", "Data Engine (Kinetic Analysis)"])

# ── Pestaña 1 ──
with tab1:
    st.subheader("Matrix-to-TSV Transformation")
    uploaded = st.file_uploader("Magellan output (xlsx/csv)", type=["xlsx","csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            df = df.set_index(df.columns[0]).T
            df.index = df.index.str.replace('nm','',case=False).astype(int)

            wells = [f"{r}{c}" for r in 'ABCDEFGH' for c in range(1,13)]
            df_out = pd.DataFrame(index=df.index, columns=wells)
            for w in wells:
                df_out[w] = df.get(w, "")

            df_export = df_out.reset_index().rename(columns={'index':'Wavelength'})

            c1, c2 = st.columns([3,1])
            with c1:
                st.success("Procesado correctamente")
                st.dataframe(df_export.head(10))
            with c2:
                output = io.StringIO()
                df_export.to_csv(output, sep='\t', index=False, na_rep="")
                st.download_button("Descargar TSV", output.getvalue(),
                                  f"{uploaded.name.split('.')[0]}_Spectrum.tsv",
                                  "text/tab-separated-values")
        except Exception as e:
            st.error(f"Error: {e}")

# ── Pestaña 2 ──
with tab2:
    cols = st.columns([2,1,1])
    with cols[0]:
        st.markdown("### Kinetic Analysis")
    with cols[1]:
        run_button = st.button("🚀 Ejecutar análisis", use_container_width=True)
    with cols[2]:
        if 'zip_data' in st.session_state:
            st.download_button("📥 Descargar ZIP", st.session_state.zip_data,
                              "Biotrans_Report.zip", use_container_width=True)

    with st.expander("Carga de datos", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Metadata**")
            metadata_file = st.file_uploader("CSV metadata", type="csv", key="meta")
        with c2:
            st.markdown("**Espectros experimentales**")
            spectra_files = st.file_uploader("Archivos TSV", type="tsv", accept_multiple_files=True, key="spectra")
        with c3:
            st.markdown("**Espectros de referencia**")
            ref_files = st.file_uploader("CSV referencias", type="csv", accept_multiple_files=True, key="refs")

    if run_button:
        if metadata_file and spectra_files and ref_files:
            with st.spinner("Procesando..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    df_meta = pd.read_csv(metadata_file)

                    # ── Carga y limpieza de referencias ───────────────────────────────
                    ref_dict = {}
                    for f in ref_files:
                        name = f.name.strip()
                        name_key = name.lower()
                        try:
                            df = pd.read_csv(f, sep=None, engine='python')
                            wave_col = next((c for c in df.columns if 'wave' in str(c).lower() or str(c).isdigit()), df.columns[0])
                            abs_col  = next((c for c in df.columns if 'abs' in str(c).lower()), df.columns[1] if len(df.columns)>1 else None)

                            if abs_col is None:
                                st.warning(f"No columna Abs en {name}")
                                continue

                            df_clean = df[[wave_col, abs_col]].copy()
                            df_clean.columns = ['Wavelength', 'Abs']
                            df_clean['Wavelength'] = pd.to_numeric(df_clean['Wavelength'], errors='coerce').round(0).astype(float)
                            df_clean['Abs'] = pd.to_numeric(df_clean['Abs'], errors='coerce')
                            df_clean = df_clean.dropna(subset=['Wavelength','Abs'])

                            ref_dict[name_key] = df_clean
                            st.info(f"Referencia cargada: {name}  ({len(df_clean)} λ)")
                        except Exception as e:
                            st.error(f"Error leyendo referencia {name}: {e}")

                    # ── Procesamiento ────────────────────────────────────────────────
                    engine = DataProcessor(df_meta, {f.name: f.getvalue() for f in spectra_files}, tmpdir, ref_dict)
                    engine.run_original_logic()

                    # ── Generar ZIP ──────────────────────────────────────────────────
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for folder in ["csv_files", "plots"]:
                            p = os.path.join(tmpdir, folder)
                            if os.path.exists(p):
                                for file in os.listdir(p):
                                    zf.write(os.path.join(p, file), os.path.join(folder, file))

                    buf.seek(0)
                    st.session_state.zip_data = buf.read()
                    st.success("Procesamiento finalizado")
                    st.rerun()

        else:
            st.warning("Faltan archivos: metadata, espectros y/o referencias")