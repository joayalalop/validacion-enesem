from __future__ import annotations

import hashlib
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from enesem_validaciones import (
    FRIENDLY_COLUMNS,
    build_case_table,
    build_validation_catalog,
    dataset_fingerprint,
    friendly_frame,
    prepare_history,
    read_workbook_sheet,
    workbook_sheet_names,
)
from enesem_validaciones.export_utils import (
    DECISION_COLUMNS,
    DECISION_LABELS,
    EVIDENCE_COLUMNS,
    PROFILE_OPTIONS,
    TREATMENT_OPTIONS,
    export_analysis_workbook,
    initialize_decisions,
    merge_decisions,
    read_decision_file,
)


st.set_page_config(
    page_title="ENESEM · Distribución de validaciones",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = ["#116B8C", "#1FA6A8", "#6BC4B8", "#F2A900", "#D95D39", "#76528B"]
OUTCOME_COLORS = {
    "Pendiente": "#F2A900",
    "Corrección parcial": "#D95D39",
    "Corregida": "#1FA6A8",
    "Justificada": "#116B8C",
    "Regla o sistema": "#76528B",
    "Otro estado": "#8A94A0",
    "Sin estado": "#C4CBD2",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stMetric"] {background: #F6FAFC; border: 1px solid #D9E6EC; padding: 0.8rem 1rem; border-radius: 0.55rem;}
      [data-testid="stMetricLabel"] {font-weight: 600; color: #37505E;}
      [data-testid="stSidebar"] {background: #F7FAFC;}
      .enesem-kicker {font-size: 0.82rem; font-weight: 700; letter-spacing: .08em; color: #116B8C; text-transform: uppercase;}
      .enesem-subtitle {color: #536872; margin-top: -.45rem; margin-bottom: 1rem;}
      .enesem-note {border-left: 4px solid #1FA6A8; background: #EEF7F8; padding: .85rem 1rem; border-radius: .25rem; margin: .4rem 0 1rem 0;}
      .small-muted {font-size: .88rem; color: #687D87;}
      div[data-testid="stDataFrame"] {border: 1px solid #E1E8EC; border-radius: .4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_sheet_names(file_bytes: bytes) -> list[str]:
    return workbook_sheet_names(file_bytes)


@st.cache_data(show_spinner=False)
def cached_dataset(file_bytes: bytes, sheet_name: str):
    source = read_workbook_sheet(file_bytes, sheet_name)
    history, report = prepare_history(source)
    cases = build_case_table(history)
    catalog = build_validation_catalog(cases)
    return history, cases, catalog, report


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}".replace(",", ".")


def fmt_pct(value: float | int, denominator: float | int | None = None) -> str:
    if denominator is not None:
        if not denominator:
            return "0,0 %"
        value = float(value) / float(denominator) * 100
    if value is None or pd.isna(value):
        return "n.d."
    return f"{float(value):.1f} %".replace(".", ",")


def chapter_sort_key(value: str):
    if value.startswith("Capítulo "):
        try:
            return (0, int(value.split()[-1]))
        except ValueError:
            pass
    return (1, value)


def nonblank_options(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).str.strip()
    return sorted(values[values.ne("")].unique().tolist())


def apply_case_filters(
    cases: pd.DataFrame,
    filters: dict,
    *,
    ignore_chapter: bool = False,
) -> pd.DataFrame:
    result = cases.copy()
    if not ignore_chapter:
        chapter_group = filters.get("chapter_group", "Todos")
        if chapter_group != "Todos":
            result = result[result["capitulo_agrupado"].eq(chapter_group)]
        raw_chapters = filters.get("raw_chapters", [])
        if raw_chapters:
            result = result[result["capitulo"].isin(raw_chapters)]

    zones = filters.get("zones", [])
    if zones:
        result = result[result["cz_final"].isin(zones)]
    outcomes = filters.get("outcomes", [])
    if outcomes:
        result = result[result["resultado_final"].isin(outcomes)]
    cod_values = filters.get("cod_values", [])
    if cod_values:
        result = result[result["resultado_cod_val"].isin(cod_values)]

    interviewer = filters.get("interviewer", "Todos")
    if interviewer != "Todos":
        result = result[result["encuestador_nombre"].fillna("").eq(interviewer)]
    load_division = filters.get("load_division", "Todas")
    if load_division != "Todas":
        result = result[result["division_carga"].fillna("").eq(load_division)]
    if filters.get("only_reappearing", False):
        result = result[result["reaparece"]]

    start_date = filters.get("start_date")
    end_date = filters.get("end_date")
    if start_date is not None and end_date is not None:
        dated = result["ultima_aparicion"].dt.date.between(start_date, end_date)
        if filters.get("include_undated", True):
            dated = dated | result["ultima_aparicion"].isna()
        result = result[dated]

    search = str(filters.get("search", "")).strip().casefold()
    if search:
        columns = [
            "id_empresa",
            "id_error",
            "error_n",
            "mensaje",
            "ruc",
            "nombre_comercial",
            "razon_social",
            "encuestador_nombre",
            "division_carga",
        ]
        search_mask = pd.Series(False, index=result.index)
        for column in columns:
            if column in result.columns:
                search_mask |= result[column].fillna("").astype(str).str.casefold().str.contains(
                    search, regex=False
                )
        result = result[search_mask]
    return result


def plot_or_message(figure, *, height: int | None = None):
    if height:
        figure.update_layout(height=height)
    figure.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(family="Arial", size=12, color="#233642"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def show_case_table(frame: pd.DataFrame, show_identity: bool, height: int = 420):
    identity = ["ruc", "nombre_comercial", "razon_social"] if show_identity else []
    columns = [
        "id_empresa",
        *identity,
        "encuestador_nombre",
        "division_carga",
        "id_error",
        "error_n",
        "cz_final",
        "capitulo",
        "mensaje",
        "estado_final",
        "resultado_cod_val",
        "apariciones_malla",
        "reaparece",
        "ultima_aparicion",
    ]
    display = friendly_frame(frame, columns)
    if "Reaparece en otra malla" in display:
        display["Reaparece en otra malla"] = display["Reaparece en otra malla"].map(
            {True: "Sí", False: "No"}
        )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=height,
        column_config={
            "Mensaje de la validación": st.column_config.TextColumn(width="large"),
            "Razón social": st.column_config.TextColumn(width="medium"),
            "Nombre comercial": st.column_config.TextColumn(width="medium"),
            "Última aparición": st.column_config.DateColumn(format="DD/MM/YYYY"),
        },
    )


st.markdown('<div class="enesem-kicker">Encuesta Estructural Empresarial</div>', unsafe_allow_html=True)
st.title("Distribución de validaciones por perfil")
st.markdown(
    '<div class="enesem-subtitle">Diagnóstico histórico, priorización y acuerdos para la mesa de trabajo.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Archivo semanal")
    uploaded = st.file_uploader(
        "Cargar histórico de validaciones",
        type=["xlsx", "xlsm"],
        help="El archivo se procesa en memoria y no forma parte de la aplicación.",
    )

if uploaded is None:
    st.info("Carga el archivo histórico semanal en el panel izquierdo para iniciar el análisis.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. Reconocimiento automático")
        st.write(
            "La aplicación identifica los nombres habituales de las columnas y puede generar `id_error` "
            "si existen `id_empresa` y `error_n`."
        )
    with c2:
        st.subheader("2. Datos empresariales opcionales")
        st.write(
            "RUC, razón social, nombre comercial, encuestador y división de carga se muestran cuando "
            "existen. Si faltan, se crean vacíos."
        )
    with c3:
        st.subheader("3. Resultado de la mesa")
        st.write(
            "Las decisiones sobre responsables, necesidad de consultar al informante y tratamiento "
            "pueden descargarse en Excel."
        )
    st.markdown(
        """
        #### Columnas mínimas

        El archivo debe contener `id_empresa` y `error_n`. Las demás columnas enriquecen el análisis, pero su ausencia no bloquea la aplicación.
        """
    )
    st.stop()

file_bytes = uploaded.getvalue()
try:
    sheet_names = cached_sheet_names(file_bytes)
except Exception as exc:
    st.error(f"No fue posible leer el libro de Excel: {exc}")
    st.stop()

with st.sidebar:
    selected_sheet = st.selectbox("Hoja de datos", sheet_names)

try:
    with st.spinner("Preparando casos, reapariciones y catálogo de validaciones..."):
        history, cases, full_catalog, report = cached_dataset(file_bytes, selected_sheet)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"El archivo no pudo procesarse. Revisa su estructura. Detalle: {exc}")
    st.stop()

fingerprint = dataset_fingerprint(file_bytes, selected_sheet)
if st.session_state.get("decision_dataset") != fingerprint:
    st.session_state["decision_dataset"] = fingerprint
    st.session_state["decisions"] = initialize_decisions(full_catalog)

with st.sidebar:
    st.divider()
    st.header("Filtros")
    chapter_options = sorted(cases["capitulo_agrupado"].dropna().unique().tolist(), key=chapter_sort_key)
    chapter_group = st.selectbox("Capítulo agrupado", ["Todos", *chapter_options])
    raw_scope = cases if chapter_group == "Todos" else cases[cases["capitulo_agrupado"].eq(chapter_group)]
    raw_chapters = st.multiselect(
        "Sección o capítulo original",
        nonblank_options(raw_scope["capitulo"]),
        help="Déjalo vacío para incluir todas las secciones del capítulo agrupado.",
    )
    zone_options = nonblank_options(cases["cz_final"])
    zones = st.multiselect("Coordinación Zonal", zone_options, default=zone_options)
    outcome_options = nonblank_options(cases["resultado_final"])
    outcomes = st.multiselect("Resultado final", outcome_options, default=outcome_options)
    cod_options = nonblank_options(cases["resultado_cod_val"])
    cod_values = st.multiselect("Clasificación cod_val", cod_options, default=cod_options)

    interviewer_options = nonblank_options(cases["encuestador_nombre"])
    interviewer = st.selectbox(
        "Encuestador/a",
        ["Todos", *interviewer_options] if interviewer_options else ["Todos"],
        disabled=not interviewer_options,
    )
    division_options = nonblank_options(cases["division_carga"])
    load_division = st.selectbox(
        "División de carga",
        ["Todas", *division_options] if division_options else ["Todas"],
        disabled=not division_options,
    )

    valid_last_dates = cases["ultima_aparicion"].dropna()
    start_date = end_date = None
    include_undated = True
    if not valid_last_dates.empty:
        min_date = valid_last_dates.min().date()
        max_date = valid_last_dates.max().date()
        date_range = st.date_input(
            "Fecha de última aparición",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
        include_undated = st.checkbox("Incluir casos sin fecha", value=True)

    only_reappearing = st.checkbox("Solo casos que reaparecen", value=False)
    search = st.text_input(
        "Buscar",
        placeholder="ID, RUC, empresa, error o texto del mensaje",
    )
    show_identity = st.checkbox("Mostrar datos empresariales", value=True)

filters = {
    "chapter_group": chapter_group,
    "raw_chapters": raw_chapters,
    "zones": zones,
    "outcomes": outcomes,
    "cod_values": cod_values,
    "interviewer": interviewer,
    "load_division": load_division,
    "start_date": start_date,
    "end_date": end_date,
    "include_undated": include_undated,
    "only_reappearing": only_reappearing,
    "search": search,
}

filtered_cases = apply_case_filters(cases, filters)
filtered_catalog = build_validation_catalog(filtered_cases)
case_ids = set(filtered_cases["id_error"].astype(str))
filtered_history = history[history["id_error"].astype(str).isin(case_ids)].copy()

with st.sidebar:
    st.divider()
    st.caption(f"{fmt_int(len(filtered_cases))} casos después de aplicar los filtros")
    if report.missing_optional_columns:
        missing_labels = [FRIENDLY_COLUMNS.get(column, column) for column in report.missing_optional_columns]
        st.caption("Campos opcionales ausentes: " + ", ".join(missing_labels))

if filtered_cases.empty:
    st.warning("Los filtros seleccionados no dejan casos para analizar. Ajusta el panel izquierdo.")
    st.stop()

st.markdown(
    '<div class="enesem-note"><b>Unidad de análisis:</b> los indicadores cuentan combinaciones únicas de empresa y validación mediante <code>id_error</code>. Las filas históricas se utilizan únicamente para reconstruir estados y reapariciones.</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Casos", fmt_int(filtered_cases["id_error"].nunique()))
m2.metric("Empresas", fmt_int(filtered_cases["id_empresa"].nunique()))
m3.metric("Validaciones", fmt_int(filtered_cases["validation_key"].nunique()))
m4.metric(
    "Reaparición",
    fmt_pct(filtered_cases["reaparece"].sum(), len(filtered_cases)),
    help="Casos que aparecen en más de una fecha de malla.",
)
m5.metric(
    "Evaluados",
    fmt_pct(filtered_cases["evaluada"].sum(), len(filtered_cases)),
    help="Casos cuyo último resultado registrado tiene código 1, 2 o 3.",
)

tabs = st.tabs(
    [
        "Panorama",
        "Validaciones prioritarias",
        "Ficha de validación",
        "Comparación zonal",
        "Capítulo 5",
        "Asignación por perfil",
        "Calidad de datos y guía",
    ]
)

with tabs[0]:
    left, right = st.columns([1.15, 1])
    with left:
        by_chapter = (
            filtered_cases.groupby("capitulo_agrupado", as_index=False)["id_error"]
            .nunique()
            .rename(columns={"id_error": "Casos", "capitulo_agrupado": "Capítulo"})
            .sort_values("Casos", ascending=True)
        )
        fig = px.bar(
            by_chapter,
            x="Casos",
            y="Capítulo",
            orientation="h",
            title="Casos únicos por capítulo",
            color_discrete_sequence=[COLORS[0]],
            text_auto=True,
        )
        plot_or_message(fig, height=420)
    with right:
        by_outcome = (
            filtered_cases["resultado_final"].value_counts().rename_axis("Resultado").reset_index(name="Casos")
        )
        fig = px.pie(
            by_outcome,
            names="Resultado",
            values="Casos",
            hole=0.55,
            title="Último resultado registrado",
            color="Resultado",
            color_discrete_map=OUTCOME_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        plot_or_message(fig, height=420)

    st.subheader("Validaciones con mayor volumen")
    top = filtered_catalog.head(20)
    display_columns = [
        "error_n",
        "capitulo_agrupado",
        "capitulo",
        "mensaje",
        "casos",
        "empresas",
        "evaluadas",
        "pct_reaparicion",
        "pct_codigo_2",
        "pct_codigo_3",
    ]
    st.dataframe(
        friendly_frame(top, display_columns),
        hide_index=True,
        width="stretch",
        height=530,
        column_config={
            "Mensaje de la validación": st.column_config.TextColumn(width="large"),
            "% de reaparición": st.column_config.NumberColumn(format="%.1f %%"),
            "% código 2 entre evaluadas": st.column_config.NumberColumn(format="%.1f %%"),
            "% código 3 entre evaluadas": st.column_config.NumberColumn(format="%.1f %%"),
        },
    )

with tabs[1]:
    st.subheader("Priorización basada en evidencia")
    st.caption(
        "Los porcentajes de códigos 1, 2 y 3 utilizan como denominador únicamente los casos evaluados."
    )
    p1, p2, p3 = st.columns([1.2, 1, 1])
    sort_options = {
        "Mayor volumen": "casos",
        "Mayor reaparición": "pct_reaparicion",
        "Mayor proporción de código 3": "pct_codigo_3",
        "Mayor proporción de código 2": "pct_codigo_2",
        "Mayor proporción de código 1": "pct_codigo_1",
    }
    with p1:
        sort_label = st.selectbox("Ordenar por", list(sort_options))
    with p2:
        minimum_cases = st.number_input("Mínimo de casos", min_value=1, value=20, step=1)
    with p3:
        top_n = st.slider("Número de validaciones", min_value=10, max_value=100, value=30, step=5)

    priority = filtered_catalog[filtered_catalog["casos"].ge(minimum_cases)].copy()
    priority = priority.sort_values(sort_options[sort_label], ascending=False, na_position="last").head(top_n)
    if priority.empty:
        st.info("No existen validaciones que cumplan el mínimo de casos seleccionado.")
    else:
        scatter_source = filtered_catalog[filtered_catalog["casos"].ge(minimum_cases)].copy()
        scatter_source["mensaje_corto"] = scatter_source["mensaje"].fillna("").str.slice(0, 170)
        scatter_source["tamano"] = scatter_source["empresas"].clip(lower=1)
        fig = px.scatter(
            scatter_source,
            x="casos",
            y="pct_reaparicion",
            size="tamano",
            color="capitulo_agrupado",
            hover_name="error_n",
            hover_data={
                "mensaje_corto": True,
                "casos": ":,",
                "empresas": ":,",
                "pct_reaparicion": ":.1f",
                "tamano": False,
            },
            labels={
                "casos": "Casos únicos",
                "pct_reaparicion": "% de reaparición",
                "capitulo_agrupado": "Capítulo",
                "mensaje_corto": "Mensaje",
            },
            title="Volumen y reaparición por validación",
            color_discrete_sequence=COLORS,
        )
        plot_or_message(fig, height=500)
        st.dataframe(
            friendly_frame(
                priority,
                [
                    "error_n",
                    "capitulo_agrupado",
                    "capitulo",
                    "mensaje",
                    "casos",
                    "empresas",
                    "evaluadas",
                    "pct_reaparicion",
                    "pct_codigo_1",
                    "pct_codigo_2",
                    "pct_codigo_3",
                ],
            ),
            hide_index=True,
            width="stretch",
            height=560,
            column_config={
                "Mensaje de la validación": st.column_config.TextColumn(width="large"),
                "% de reaparición": st.column_config.NumberColumn(format="%.1f %%"),
                "% código 1 entre evaluadas": st.column_config.NumberColumn(format="%.1f %%"),
                "% código 2 entre evaluadas": st.column_config.NumberColumn(format="%.1f %%"),
                "% código 3 entre evaluadas": st.column_config.NumberColumn(format="%.1f %%"),
            },
        )

with tabs[2]:
    st.subheader("Ficha de una validación")
    labels = {
        row.validation_key: f"{row.error_n} · {row.capitulo} · {str(row.mensaje)[:105]}"
        for row in filtered_catalog.itertuples()
    }
    selected_key = st.selectbox(
        "Selecciona una validación",
        filtered_catalog["validation_key"].tolist(),
        format_func=lambda key: labels.get(key, key),
    )
    selected_catalog = filtered_catalog[filtered_catalog["validation_key"].eq(selected_key)].iloc[0]
    selected_cases = filtered_cases[filtered_cases["validation_key"].eq(selected_key)].copy()

    st.markdown(f"**{selected_catalog['mensaje']}**")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Casos", fmt_int(selected_catalog["casos"]))
    f2.metric("Empresas", fmt_int(selected_catalog["empresas"]))
    f3.metric("Reaparición", fmt_pct(selected_catalog["pct_reaparicion"]))
    f4.metric("Código 2", fmt_pct(selected_catalog["pct_codigo_2"]))
    f5.metric("Código 3", fmt_pct(selected_catalog["pct_codigo_3"]))

    chart_left, chart_right = st.columns(2)
    with chart_left:
        zone_outcome = (
            selected_cases.groupby(["cz_final", "resultado_final"], as_index=False)["id_error"]
            .nunique()
            .rename(columns={"id_error": "Casos", "cz_final": "Coordinación Zonal", "resultado_final": "Resultado"})
        )
        fig = px.bar(
            zone_outcome,
            x="Coordinación Zonal",
            y="Casos",
            color="Resultado",
            barmode="stack",
            title="Resultados por Coordinación Zonal",
            color_discrete_map=OUTCOME_COLORS,
        )
        plot_or_message(fig, height=390)
    with chart_right:
        zone_rep = (
            selected_cases.groupby("cz_final", as_index=False)
            .agg(casos=("id_error", "nunique"), reapariciones=("reaparece", "sum"))
        )
        zone_rep["Porcentaje"] = zone_rep["reapariciones"].div(zone_rep["casos"]).mul(100)
        fig = px.bar(
            zone_rep,
            x="cz_final",
            y="Porcentaje",
            title="Reaparición por Coordinación Zonal",
            labels={"cz_final": "Coordinación Zonal", "Porcentaje": "% de reaparición"},
            color_discrete_sequence=[COLORS[1]],
            text_auto=".1f",
        )
        plot_or_message(fig, height=390)

    st.subheader("Casos asociados")
    show_case_table(selected_cases, show_identity, height=400)

    case_labels = {}
    for row in selected_cases.itertuples():
        company = row.razon_social if pd.notna(row.razon_social) and str(row.razon_social).strip() else row.id_empresa
        case_labels[row.id_error] = f"{company} · {row.id_error}"
    selected_case_id = st.selectbox(
        "Ver el historial de un caso",
        selected_cases["id_error"].astype(str).tolist(),
        format_func=lambda key: case_labels.get(key, key),
    )
    timeline = filtered_history[filtered_history["id_error"].astype(str).eq(str(selected_case_id))].sort_values(
        "_source_row"
    )
    timeline_columns = [
        "_source_row",
        "fecha_malla_dt",
        "estado",
        "cod_val_num",
        "observacion_encuestador",
        "observacion_critico",
        "observacion_revisor",
        "observaciones_zonal",
    ]
    timeline_display = timeline[timeline_columns].rename(
        columns={
            "_source_row": "Fila en el archivo",
            "fecha_malla_dt": "Fecha de malla",
            "estado": "Estado registrado",
            "cod_val_num": "cod_val",
            "observacion_encuestador": "Observación del encuestador",
            "observacion_critico": "Observación del crítico",
            "observacion_revisor": "Observación del revisor",
            "observaciones_zonal": "Observación zonal",
        }
    )
    st.dataframe(
        timeline_display,
        hide_index=True,
        width="stretch",
        column_config={
            "Fecha de malla": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
            "Observación del encuestador": st.column_config.TextColumn(width="large"),
            "Observación del crítico": st.column_config.TextColumn(width="large"),
            "Observación del revisor": st.column_config.TextColumn(width="large"),
            "Observación zonal": st.column_config.TextColumn(width="large"),
        },
    )

with tabs[3]:
    st.subheader("Comparación entre Coordinaciones Zonales")
    comparison = (
        filtered_cases.groupby("cz_final", as_index=False)
        .agg(
            casos=("id_error", "nunique"),
            empresas=("id_empresa", "nunique"),
            validaciones=("validation_key", "nunique"),
            evaluadas=("evaluada", "sum"),
            reapariciones=("reaparece", "sum"),
            codigo_1=("codigo_1", "sum"),
            codigo_2=("codigo_2", "sum"),
            codigo_3=("codigo_3", "sum"),
        )
    )
    comparison["pct_evaluadas"] = comparison["evaluadas"].div(comparison["casos"]).mul(100)
    comparison["pct_reaparicion"] = comparison["reapariciones"].div(comparison["casos"]).mul(100)
    denominator = comparison["evaluadas"].replace(0, np.nan)
    for code in [1, 2, 3]:
        comparison[f"pct_codigo_{code}"] = comparison[f"codigo_{code}"].div(denominator).mul(100)

    st.dataframe(
        comparison.rename(
            columns={
                "cz_final": "Coordinación Zonal",
                "casos": "Casos",
                "empresas": "Empresas",
                "validaciones": "Validaciones",
                "evaluadas": "Evaluados",
                "reapariciones": "Reapariciones",
                "pct_evaluadas": "% evaluados",
                "pct_reaparicion": "% reaparición",
                "pct_codigo_1": "% código 1",
                "pct_codigo_2": "% código 2",
                "pct_codigo_3": "% código 3",
            }
        )[
            [
                "Coordinación Zonal",
                "Casos",
                "Empresas",
                "Validaciones",
                "Evaluados",
                "% evaluados",
                "Reapariciones",
                "% reaparición",
                "% código 1",
                "% código 2",
                "% código 3",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "% evaluados": st.column_config.NumberColumn(format="%.1f %%"),
            "% reaparición": st.column_config.NumberColumn(format="%.1f %%"),
            "% código 1": st.column_config.NumberColumn(format="%.1f %%"),
            "% código 2": st.column_config.NumberColumn(format="%.1f %%"),
            "% código 3": st.column_config.NumberColumn(format="%.1f %%"),
        },
    )

    zleft, zright = st.columns(2)
    with zleft:
        outcome_zone = (
            filtered_cases.groupby(["cz_final", "resultado_final"], as_index=False)["id_error"]
            .nunique()
            .rename(columns={"id_error": "Casos", "cz_final": "Coordinación Zonal", "resultado_final": "Resultado"})
        )
        outcome_zone["Porcentaje"] = outcome_zone["Casos"].div(
            outcome_zone.groupby("Coordinación Zonal")["Casos"].transform("sum")
        ).mul(100)
        fig = px.bar(
            outcome_zone,
            x="Coordinación Zonal",
            y="Porcentaje",
            color="Resultado",
            barmode="stack",
            title="Composición de resultados por zonal",
            color_discrete_map=OUTCOME_COLORS,
            labels={"Porcentaje": "% de casos"},
        )
        plot_or_message(fig, height=430)
    with zright:
        fig = px.bar(
            comparison.sort_values("pct_reaparicion", ascending=False),
            x="cz_final",
            y="pct_reaparicion",
            title="Reaparición dentro de cada zonal",
            labels={"cz_final": "Coordinación Zonal", "pct_reaparicion": "% de reaparición"},
            color_discrete_sequence=[COLORS[1]],
            text_auto=".1f",
        )
        plot_or_message(fig, height=430)
    st.caption(
        "La comparación es descriptiva. Para evaluar desempeño debe considerarse la mezcla de capítulos, validaciones, empresas y fechas de cada zonal."
    )

with tabs[4]:
    st.subheader("Análisis específico del capítulo 5")
    st.info(
        "El capítulo 5 se mantiene dentro del diagnóstico, pero se presenta por separado porque parte de su información puede provenir de la precarga. "
        "El histórico no permite determinar automáticamente si una inconsistencia nació en la precarga o en una modificación posterior."
    )
    cap5_base = apply_case_filters(cases, filters, ignore_chapter=True)
    cap5_cases = cap5_base[cap5_base["capitulo_agrupado"].eq("Capítulo 5")].copy()
    if cap5_cases.empty:
        st.warning("No existen casos del capítulo 5 bajo los demás filtros seleccionados.")
    else:
        cap5_catalog = build_validation_catalog(cap5_cases)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Casos", fmt_int(len(cap5_cases)))
        c2.metric("Empresas", fmt_int(cap5_cases["id_empresa"].nunique()))
        c3.metric("Validaciones", fmt_int(cap5_cases["validation_key"].nunique()))
        c4.metric("Reaparición", fmt_pct(cap5_cases["reaparece"].sum(), len(cap5_cases)))

        cleft, cright = st.columns(2)
        with cleft:
            subsection = (
                cap5_cases.groupby("capitulo", as_index=False)["id_error"]
                .nunique()
                .rename(columns={"id_error": "Casos", "capitulo": "Sección"})
                .sort_values("Casos", ascending=True)
            )
            fig = px.bar(
                subsection,
                x="Casos",
                y="Sección",
                orientation="h",
                title="Casos por sección del capítulo 5",
                color_discrete_sequence=[COLORS[0]],
                text_auto=True,
            )
            plot_or_message(fig, height=390)
        with cright:
            cap5_outcome = cap5_cases["resultado_cod_val"].value_counts().rename_axis("Resultado").reset_index(name="Casos")
            fig = px.pie(
                cap5_outcome,
                names="Resultado",
                values="Casos",
                hole=0.5,
                title="Clasificación de los casos del capítulo 5",
                color_discrete_sequence=COLORS,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            plot_or_message(fig, height=390)

        st.subheader("Validaciones del capítulo 5")
        st.dataframe(
            friendly_frame(
                cap5_catalog,
                [
                    "error_n",
                    "capitulo",
                    "mensaje",
                    "casos",
                    "empresas",
                    "evaluadas",
                    "pct_reaparicion",
                    "pct_codigo_1",
                    "pct_codigo_2",
                    "pct_codigo_3",
                ],
            ),
            hide_index=True,
            width="stretch",
            height=500,
            column_config={
                "Mensaje de la validación": st.column_config.TextColumn(width="large"),
                "% de reaparición": st.column_config.NumberColumn(format="%.1f %%"),
                "% código 1 entre evaluadas": st.column_config.NumberColumn(format="%.1f %%"),
                "% código 2 entre evaluadas": st.column_config.NumberColumn(format="%.1f %%"),
                "% código 3 entre evaluadas": st.column_config.NumberColumn(format="%.1f %%"),
            },
        )
        with st.expander("Ver casos y empresas del capítulo 5"):
            show_case_table(cap5_cases, show_identity, height=470)

with tabs[5]:
    st.subheader("Matriz de distribución por perfil")
    st.write(
        "Utiliza la evidencia precargada para acordar quién debe resolver cada validación, cuándo se requiere contactar al informante y qué tratamiento debe aplicarse."
    )

    import_col, import_button_col = st.columns([3, 1])
    with import_col:
        previous_file = st.file_uploader(
            "Incorporar acuerdos de una sesión anterior",
            type=["xlsx", "csv"],
            key="previous_decisions",
            help="Puedes volver a cargar el Excel exportado por esta aplicación la semana siguiente.",
        )
    with import_button_col:
        st.write("")
        st.write("")
        import_clicked = st.button("Incorporar acuerdos", width="stretch")
    if import_clicked:
        if previous_file is None:
            st.warning("Selecciona primero un archivo de acuerdos.")
        else:
            try:
                imported = read_decision_file(previous_file.getvalue(), previous_file.name)
                merged, matched = merge_decisions(st.session_state["decisions"], imported)
                st.session_state["decisions"] = merged
                st.success(f"Se incorporaron acuerdos para {matched:,} validaciones.".replace(",", "."))
            except Exception as exc:
                st.error(f"No fue posible incorporar la matriz: {exc}")

    a1, a2, a3 = st.columns([1.2, 1, 1])
    editor_sort_options = {
        "Mayor volumen": "casos",
        "Mayor reaparición": "pct_reaparicion",
        "Mayor código 3": "pct_codigo_3",
        "Mayor código 2": "pct_codigo_2",
    }
    with a1:
        editor_sort_label = st.selectbox("Orden de la matriz", list(editor_sort_options), key="editor_sort")
    with a2:
        editor_min_cases = st.number_input(
            "Mínimo de casos para la matriz", min_value=1, value=5, step=1, key="editor_min"
        )
    with a3:
        editor_limit = st.slider("Filas para trabajar", 10, 250, 60, 10)

    editor_catalog = filtered_catalog[filtered_catalog["casos"].ge(editor_min_cases)].copy()
    editor_catalog = editor_catalog.sort_values(
        editor_sort_options[editor_sort_label], ascending=False, na_position="last"
    ).head(editor_limit)
    editor_keys = editor_catalog["validation_key"].tolist()
    master = st.session_state["decisions"].set_index("validation_key")
    available_keys = [key for key in editor_keys if key in master.index]
    editor_columns = [
        "error_n",
        "capitulo",
        "mensaje",
        "casos",
        "pct_reaparicion",
        "pct_codigo_2",
        "pct_codigo_3",
        *DECISION_COLUMNS,
    ]
    editor_source = master.loc[available_keys, editor_columns].reset_index().copy()
    editor_hash = hashlib.sha1("|".join(available_keys).encode("utf-8")).hexdigest()[:10]
    edited = st.data_editor(
        editor_source,
        hide_index=True,
        width="stretch",
        height=650,
        key=f"assignment_editor_{editor_hash}",
        disabled=[
            "validation_key",
            "error_n",
            "capitulo",
            "mensaje",
            "casos",
            "pct_reaparicion",
            "pct_codigo_2",
            "pct_codigo_3",
        ],
        column_config={
            "validation_key": None,
            "error_n": st.column_config.TextColumn("Código", width="small"),
            "capitulo": st.column_config.TextColumn("Capítulo", width="small"),
            "mensaje": st.column_config.TextColumn("Mensaje", width="large"),
            "casos": st.column_config.NumberColumn("Casos", format="%d"),
            "pct_reaparicion": st.column_config.NumberColumn("% reaparición", format="%.1f %%"),
            "pct_codigo_2": st.column_config.NumberColumn("% código 2", format="%.1f %%"),
            "pct_codigo_3": st.column_config.NumberColumn("% código 3", format="%.1f %%"),
            "prioridad_acordada": st.column_config.SelectboxColumn(
                DECISION_LABELS["prioridad_acordada"], options=["", "Alta", "Media", "Baja"]
            ),
            "requiere_informante": st.column_config.SelectboxColumn(
                DECISION_LABELS["requiere_informante"], options=["", "Sí", "No", "Depende"]
            ),
            "origen_probable": st.column_config.SelectboxColumn(
                DECISION_LABELS["origen_probable"],
                options=["", "Precarga", "Ingreso de la empresa", "Ambos", "No determinado"],
            ),
            "responsable_principal": st.column_config.SelectboxColumn(
                DECISION_LABELS["responsable_principal"], options=PROFILE_OPTIONS
            ),
            "responsable_apoyo": st.column_config.SelectboxColumn(
                DECISION_LABELS["responsable_apoyo"], options=PROFILE_OPTIONS
            ),
            "tratamiento_acordado": st.column_config.SelectboxColumn(
                DECISION_LABELS["tratamiento_acordado"], options=TREATMENT_OPTIONS, width="medium"
            ),
            "criterio_cierre": st.column_config.TextColumn(
                DECISION_LABELS["criterio_cierre"], width="large"
            ),
            "observacion_acuerdo": st.column_config.TextColumn(
                DECISION_LABELS["observacion_acuerdo"], width="large"
            ),
        },
    )

    save_col, status_col = st.columns([1, 3])
    with save_col:
        save_changes = st.button("Guardar cambios de la matriz", type="primary", width="stretch")
    if save_changes:
        updated_master = st.session_state["decisions"].set_index("validation_key")
        edited_indexed = edited.set_index("validation_key")
        for column in DECISION_COLUMNS:
            updated_master.loc[edited_indexed.index, column] = edited_indexed[column].fillna("").astype(str)
        st.session_state["decisions"] = updated_master.reset_index()
        with status_col:
            st.success("Los cambios quedaron guardados en esta sesión y se incluirán en la descarga.")

    st.divider()
    export_sensitive = st.checkbox(
        "Incluir datos empresariales y del encuestador en la hoja Casos_filtrados",
        value=show_identity,
    )
    export_bytes = export_analysis_workbook(
        st.session_state["decisions"],
        filtered_cases,
        report,
        include_sensitive=export_sensitive,
    )
    max_date = filtered_cases["ultima_aparicion"].max()
    suffix = max_date.strftime("%Y%m%d") if pd.notna(max_date) else date.today().strftime("%Y%m%d")
    st.download_button(
        "Descargar matriz y resultados en Excel",
        data=export_bytes,
        file_name=f"ENESEM_distribucion_validaciones_{suffix}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

with tabs[6]:
    st.subheader("Calidad y reconocimiento de columnas")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Filas originales", fmt_int(report.raw_rows))
    q2.metric("Filas utilizadas", fmt_int(report.usable_rows))
    q3.metric("Fechas utilizables", fmt_int(report.valid_dates))
    q4.metric("Fechas vacías o inválidas", fmt_int(report.invalid_dates))

    if report.warnings:
        for warning in report.warnings:
            st.warning(warning)
    if report.key_collisions == 0 and report.pair_collisions == 0:
        st.success("No se detectaron colisiones en la relación entre empresa, código de validación e id_error.")

    st.dataframe(report.mapping_frame(), hide_index=True, width="stretch", height=610)

    st.subheader("Definiciones utilizadas")
    definitions = pd.DataFrame(
        {
            "Concepto": [
                "Caso",
                "Movimiento histórico",
                "Reaparición",
                "Último estado",
                "Código 1",
                "Código 2",
                "Código 3",
                "Capítulo 5",
            ],
            "Definición": [
                "Una combinación única de empresa y validación identificada mediante id_error.",
                "Una fila que registra una etapa del caso. Varias filas no significan necesariamente varias validaciones.",
                "El mismo id_error aparece en más de una fecha de malla distinta.",
                "Se toma el último registro según el orden de las filas del archivo.",
                "Inconsistencia confirmada que ingresó al proceso de corrección.",
                "La diferencia fue justificada mediante una observación correcta.",
                "La validación no debería considerarse o requiere revisión de la regla o del sistema.",
                "Se presenta por separado debido a la precarga. La fuente no permite atribuir automáticamente el origen de la inconsistencia.",
            ],
        }
    )
    st.dataframe(definitions, hide_index=True, width="stretch")
    st.caption(
        "Los archivos cargados no se guardan dentro del proyecto. Si el histórico contiene información sensible, despliega la aplicación únicamente en un entorno de acceso restringido."
    )
