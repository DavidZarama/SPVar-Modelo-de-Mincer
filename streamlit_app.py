# -*- coding: utf-8 -*-
"""
Interfaz STREAMLIT del modelo de Mincer (GEIH-DANE). Reutiliza mincer_3d_nocturno.py (mismo repositorio, misma carpeta).
Local:  streamlit run streamlit_app.py     ·     Streamlit Cloud: "Main file path" = streamlit_app.py
"""
import streamlit as st
import streamlit.components.v1 as components
import mincer_3d_nocturno as m

st.set_page_config(page_title="Mincer Colombia — visión nocturna", layout="wide")
st.title("▓ MINCER COLOMBIA — Retornos a la educación")
st.caption("ln(w) = β0 + β1·esc + β2·exp + β3·exp² + β4·mujer · MCO con EE robustos HC1 · microdatos reales GEIH-DANE (marco 2018) · límites MGN-DANE · sin datos simulados")

with st.sidebar:
    st.header("Parámetros")
    year = st.selectbox("Año GEIH (catálogos verificados)", [2022, 2023, 2024], index=2)
    n_meses = st.slider("Meses a descargar", 1, 12, 3, help="Cada mes es un .zip grande. En Streamlit Cloud la memoria/disco son limitados: empiece con pocos meses.")
    min_n = st.slider("n mínimo para estimar por departamento", 100, 1000, 300, 50)
    ejecutar = st.button("DESCARGAR Y ESTIMAR ▶", type="primary")

@st.cache_data(show_spinner="Descargando y leyendo microdatos GEIH del DANE…", max_entries=2)
def cargar(year, n_meses):
    df = m.cargar_geih(year, limite=n_meses, verbose=False)
    return m.construir_muestra(df)

@st.cache_data(show_spinner="Descargando límites oficiales MGN-DANE…")
def geometria():
    return m.load_geometry()

if ejecutar:
    try:
        muestra, cascada = cargar(year, n_meses)
        geo = geometria()
        st.session_state["res"] = (m.analizar(muestra, geo, year, min_n), cascada)
    except Exception as ex:
        st.session_state.pop("res", None)
        st.error(f"{type(ex).__name__}: {str(ex)[:600]}")

if "res" in st.session_state:
    r, cascada = st.session_state["res"]
    t = st.tabs(["3D · dinámica", "Mapa nocturno 2D", "Superficie 3D", "Nube 3D", "Interpretación automática", "Resultados"])
    with t[0]:
        components.html(r["html3d"], height=700)
        st.caption("Nota: dentro de Streamlit el navegador no concede cámara/WebXR al iframe, por lo que el modo AR no funciona aquí; use la versión Gradio (Colab/Spaces) para AR.")
    with t[1]: st.plotly_chart(r["fig_mapa"], use_container_width=True)
    with t[2]: st.plotly_chart(r["fig_sup"], use_container_width=True)
    with t[3]: st.plotly_chart(r["fig_nube"], use_container_width=True)
    with t[4]: st.markdown(r["texto"])
    with t[5]:
        st.subheader("Nacional (MCO, HC1)"); st.dataframe(r["tabla_nac"])
        st.subheader("Por departamento"); st.dataframe(r["tabla_dep"])
        st.subheader("Construcción de la muestra"); st.dataframe(cascada)
else:
    st.info("Elija año y número de meses en la barra lateral y pulse «Descargar y estimar».")
