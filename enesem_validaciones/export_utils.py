from __future__ import annotations

import re
import unicodedata
from io import BytesIO
from typing import Any

import pandas as pd

from .data_model import FRIENDLY_COLUMNS, PreparationReport, friendly_frame


DECISION_COLUMNS = [
    "prioridad_acordada",
    "requiere_informante",
    "origen_probable",
    "responsable_principal",
    "responsable_apoyo",
    "tratamiento_acordado",
    "criterio_cierre",
    "observacion_acuerdo",
]

DECISION_LABELS = {
    "prioridad_acordada": "Prioridad acordada",
    "requiere_informante": "¿Requiere al informante?",
    "origen_probable": "Origen probable",
    "responsable_principal": "Responsable principal",
    "responsable_apoyo": "Responsable de apoyo",
    "tratamiento_acordado": "Tratamiento acordado",
    "criterio_cierre": "Criterio de cierre",
    "observacion_acuerdo": "Observación o compromiso",
}

EVIDENCE_COLUMNS = [
    "validation_key",
    "error_n",
    "capitulo",
    "capitulo_agrupado",
    "mensaje",
    "casos",
    "empresas",
    "zonales",
    "evaluadas",
    "reapariciones",
    "pct_reaparicion",
    "pct_codigo_1",
    "pct_codigo_2",
    "pct_codigo_3",
]

PROFILE_OPTIONS = [
    "",
    "Encuestador/a",
    "Crítico/a",
    "Revisor/a de calidad",
    "Responsable zonal",
    "Planta Central",
    "SIPE / Informática",
    "Responsabilidad compartida",
]

TREATMENT_OPTIONS = [
    "",
    "Corregir con el informante",
    "Resolver con información disponible",
    "Aceptar justificación estandarizada",
    "Ajustar o eliminar la validación",
    "Automatizar o prevenir en el SIPE",
    "Escalar a Planta Central",
    "Fortalecer capacitación",
]


def _normalized(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def initialize_decisions(catalog: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in EVIDENCE_COLUMNS if column in catalog.columns]
    decisions = catalog[columns].copy()
    for column in DECISION_COLUMNS:
        decisions[column] = ""
    return decisions


def merge_decisions(current: pd.DataFrame, imported: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if imported is None or imported.empty:
        return current, 0

    inverse_names = {
        _normalized(label): internal for internal, label in {**FRIENDLY_COLUMNS, **DECISION_LABELS}.items()
    }
    inverse_names.update({_normalized(column): column for column in [*EVIDENCE_COLUMNS, *DECISION_COLUMNS]})
    rename = {}
    for column in imported.columns:
        internal = inverse_names.get(_normalized(column))
        if internal:
            rename[column] = internal
    imported = imported.rename(columns=rename)
    if "validation_key" not in imported.columns:
        if {"error_n", "capitulo"}.issubset(imported.columns):
            imported["validation_key"] = (
                imported["error_n"].astype(str).str.strip()
                + " | "
                + imported["capitulo"].astype(str).str.strip()
            )
        else:
            raise ValueError(
                "La matriz anterior debe contener la clave de validación o las columnas de código y capítulo."
            )

    usable_decisions = [column for column in DECISION_COLUMNS if column in imported.columns]
    if not usable_decisions:
        raise ValueError("El archivo no contiene columnas de acuerdos reconocibles.")

    imported = imported.drop_duplicates("validation_key", keep="last")
    imported = imported.set_index("validation_key")
    result = current.set_index("validation_key")
    common = result.index.intersection(imported.index)
    for column in usable_decisions:
        values = imported.loc[common, column].fillna("").astype(str)
        result.loc[common, column] = values
    return result.reset_index(), len(common)


def read_decision_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes), dtype=object)
    book = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    preferred = "Matriz_asignacion"
    sheet = preferred if preferred in book.sheet_names else book.sheet_names[0]
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet, dtype=object, engine="openpyxl")


def _friendly_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    rename = {**FRIENDLY_COLUMNS, **DECISION_LABELS}
    return decisions.rename(columns={column: rename.get(column, column) for column in decisions.columns})


def _safe_excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        if getattr(result[column].dt, "tz", None) is not None:
            result[column] = result[column].dt.tz_localize(None)
    return result


def export_analysis_workbook(
    decisions: pd.DataFrame,
    filtered_cases: pd.DataFrame,
    report: PreparationReport,
    include_sensitive: bool,
) -> bytes:
    output = BytesIO()
    identity_columns = ["ruc", "nombre_comercial", "razon_social", "encuestador_nombre", "division_carga"]
    case_columns = [
        "id_empresa",
        *identity_columns,
        "id_error",
        "error_n",
        "cz_final",
        "capitulo_agrupado",
        "capitulo",
        "mensaje",
        "estado_final",
        "resultado_final",
        "cod_val_final",
        "resultado_cod_val",
        "registros_historicos",
        "apariciones_malla",
        "reaparece",
        "primera_aparicion",
        "ultima_aparicion",
        "dias_entre_apariciones",
    ]
    if not include_sensitive:
        case_columns = [column for column in case_columns if column not in identity_columns]

    case_export = friendly_frame(filtered_cases, case_columns)
    decision_export = _friendly_decisions(decisions)
    summary = pd.DataFrame(
        {
            "Indicador": [
                "Casos empresa-validación",
                "Empresas",
                "Códigos de validación",
                "Casos que reaparecen",
                "Casos evaluados",
                "Registros históricos utilizados",
            ],
            "Valor": [
                filtered_cases["id_error"].nunique(),
                filtered_cases["id_empresa"].nunique(),
                filtered_cases["validation_key"].nunique(),
                int(filtered_cases["reaparece"].sum()),
                int(filtered_cases["evaluada"].sum()),
                report.usable_rows,
            ],
        }
    )
    guide = pd.DataFrame(
        {
            "Concepto": [
                "Caso",
                "Movimiento histórico",
                "Reaparición",
                "Porcentajes de códigos 1, 2 y 3",
                "Fecha de malla",
                "Capítulo 5",
            ],
            "Definición": [
                "Una combinación única de empresa y código de validación, identificada mediante id_error.",
                "Cada fila que registra un estado del caso. No debe contarse como una validación nueva.",
                "El mismo id_error aparece en más de una fecha de corte diferente.",
                "Se calculan solamente entre los casos que cuentan con una clasificación 1, 2 o 3.",
                "Se utiliza para identificar cortes de aparición. Los registros sin fecha se mantienen en el análisis general.",
                "Se analiza por separado porque la precarga puede influir en la generación del error. El histórico no identifica por sí solo el origen del dato.",
            ],
        }
    )

    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="dd/mm/yyyy") as writer:
        _safe_excel_frame(summary).to_excel(writer, sheet_name="Resumen", index=False)
        _safe_excel_frame(decision_export).to_excel(writer, sheet_name="Matriz_asignacion", index=False)
        _safe_excel_frame(case_export).to_excel(writer, sheet_name="Casos_filtrados", index=False)
        report.mapping_frame().to_excel(writer, sheet_name="Calidad_datos", index=False)
        guide.to_excel(writer, sheet_name="Guia", index=False)

        workbook = writer.book
        header_format = workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#145A7A",
                "border": 0,
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        )
        percent_format = workbook.add_format({"num_format": "0.0"})
        date_format = workbook.add_format({"num_format": "dd/mm/yyyy"})
        wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})

        frames = {
            "Resumen": summary,
            "Matriz_asignacion": decision_export,
            "Casos_filtrados": case_export,
            "Calidad_datos": report.mapping_frame(),
            "Guia": guide,
        }
        for sheet_name, frame in frames.items():
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            if len(frame.columns):
                worksheet.autofilter(0, 0, max(len(frame), 1), len(frame.columns) - 1)
            worksheet.set_row(0, 32, header_format)
            for index, column in enumerate(frame.columns):
                values = frame[column].fillna("").astype(str)
                sample_width = max([len(str(column)), *(len(value) for value in values.head(300))])
                width = min(max(sample_width + 2, 12), 48)
                cell_format = None
                label = str(column).lower()
                if "%" in str(column):
                    cell_format = percent_format
                elif "fecha" in label or "aparición" in label:
                    cell_format = date_format
                elif any(word in label for word in ["mensaje", "observación", "criterio", "tratamiento"]):
                    cell_format = wrap_format
                worksheet.set_column(index, index, width, cell_format)

    return output.getvalue()

