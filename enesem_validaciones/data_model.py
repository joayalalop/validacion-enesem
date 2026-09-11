from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Iterable

import numpy as np
import pandas as pd


ALIASES: dict[str, tuple[str, ...]] = {
    "id_empresa": (
        "id_empresa",
        "id empresa",
        "identificador_empresa",
        "identificador empresa",
        "inec_identificador_empresa",
        "inec identificador empresa",
    ),
    "error_n": (
        "error_n",
        "error n",
        "codigo_error",
        "código_error",
        "cod_error",
        "validacion",
        "validación",
    ),
    "cz_final": (
        "cz_final",
        "cz final",
        "coordinacion_zonal",
        "coordinación_zonal",
        "coordinacion zonal",
        "zonal",
        "cz",
    ),
    "fecha_malla": (
        "fecha_malla",
        "fecha malla",
        "fecha_validacion",
        "fecha validación",
        "fecha_corte",
        "fecha corte",
    ),
    "id_error": (
        "id_error",
        "id error",
        "identificador_error",
        "identificador error",
    ),
    "capitulo": (
        "capitulo",
        "capítulo",
        "seccion",
        "sección",
    ),
    "mensaje": (
        "mensaje",
        "mensaje_error",
        "mensaje error",
        "descripcion_error",
        "descripción_error",
    ),
    "observacion_encuestador": (
        "observacion_encuestador",
        "observación_encuestador",
        "observacion encuestador",
        "observación encuestador",
    ),
    "observacion_critico": (
        "observacion_critico",
        "observación_crítico",
        "observacion critico",
        "observación crítico",
        "observacion_critica",
    ),
    "observacion_revisor": (
        "observacion_revisor",
        "observación_revisor",
        "observacion revisor",
        "observación revisor",
    ),
    "cod_val": (
        "cod_val",
        "cod val",
        "codigo_validacion",
        "código_validación",
        "resultado_validacion",
    ),
    "observaciones_zonal": (
        "observaciones_zonal",
        "observacion_zonal",
        "observaciones zonal",
        "observación zonal",
        "observacion coordinacion zonal",
    ),
    "estado": (
        "estado",
        "estado_error",
        "estado error",
        "estado_validacion",
    ),
    "ruc": (
        "ruc",
        "numero_ruc",
        "número_ruc",
        "ruc_empresa",
    ),
    "nombre_comercial": (
        "nombre_comercial",
        "nombre comercial",
    ),
    "razon_social": (
        "razon_social",
        "razón_social",
        "razon social",
        "razón social",
    ),
    "encuestador_nombre": (
        "encuestador_nombre",
        "encuestador nombre",
        "nombre_encuestador",
        "nombre encuestador",
        "encuestador",
    ),
    "division_carga": (
        "division_carga",
        "división_carga",
        "division carga",
        "división carga",
        "carga",
    ),
}

CORE_COLUMNS = ("id_empresa", "error_n")
OPTIONAL_IDENTITY_COLUMNS = (
    "ruc",
    "nombre_comercial",
    "razon_social",
    "encuestador_nombre",
    "division_carga",
)
STANDARD_COLUMNS = tuple(ALIASES)
OBSERVATION_COLUMNS = (
    "observacion_encuestador",
    "observacion_critico",
    "observacion_revisor",
    "observaciones_zonal",
)

FRIENDLY_COLUMNS = {
    "id_empresa": "ID empresa",
    "error_n": "Código de validación",
    "cz_final": "Coordinación Zonal",
    "fecha_malla": "Fecha de malla",
    "fecha_malla_dt": "Fecha de malla",
    "fecha_corte": "Fecha de corte",
    "id_error": "ID empresa-validación",
    "capitulo": "Capítulo o sección",
    "capitulo_agrupado": "Capítulo agrupado",
    "mensaje": "Mensaje de la validación",
    "observacion_encuestador": "Observación del encuestador",
    "observacion_critico": "Observación del crítico",
    "observacion_revisor": "Observación del revisor",
    "observaciones_zonal": "Observación zonal",
    "cod_val": "Código de resultado",
    "cod_val_final": "Código de resultado final",
    "resultado_cod_val": "Resultado codificado",
    "estado": "Estado",
    "estado_final": "Estado final",
    "resultado_final": "Resultado final",
    "ruc": "RUC",
    "nombre_comercial": "Nombre comercial",
    "razon_social": "Razón social",
    "encuestador_nombre": "Encuestador/a",
    "division_carga": "División de carga",
    "registros_historicos": "Movimientos registrados",
    "apariciones_malla": "Cortes con aparición",
    "reaparece": "Reaparece en otra malla",
    "primera_aparicion": "Primera aparición",
    "ultima_aparicion": "Última aparición",
    "dias_entre_apariciones": "Días entre primera y última aparición",
    "estados_distintos": "Estados diferentes",
    "validation_key": "Clave de validación",
    "casos": "Casos",
    "empresas": "Empresas",
    "zonales": "Zonales",
    "evaluadas": "Casos evaluados",
    "reapariciones": "Casos que reaparecen",
    "pct_reaparicion": "% de reaparición",
    "codigo_1": "Código 1",
    "codigo_2": "Código 2",
    "codigo_3": "Código 3",
    "pct_codigo_1": "% código 1 entre evaluadas",
    "pct_codigo_2": "% código 2 entre evaluadas",
    "pct_codigo_3": "% código 3 entre evaluadas",
    "casos_pendientes": "Casos pendientes",
    "variantes_mensaje": "Variantes del mensaje",
}


@dataclass
class PreparationReport:
    source_columns: list[str]
    mapped_columns: dict[str, str]
    missing_optional_columns: list[str]
    added_standard_columns: list[str]
    generated_id_error_rows: int
    raw_rows: int
    usable_rows: int
    excluded_rows: int
    invalid_dates: int
    valid_dates: int
    key_collisions: int
    pair_collisions: int
    warnings: list[str] = field(default_factory=list)

    def mapping_frame(self) -> pd.DataFrame:
        rows = []
        for canonical in STANDARD_COLUMNS:
            source = self.mapped_columns.get(canonical, "No encontrada")
            status = "Reconocida" if canonical in self.mapped_columns else "Creada vacía"
            if canonical == "id_error" and self.generated_id_error_rows:
                status = "Reconocida y completada" if canonical in self.mapped_columns else "Generada"
            rows.append(
                {
                    "Campo utilizado por la aplicación": FRIENDLY_COLUMNS.get(canonical, canonical),
                    "Columna encontrada": source,
                    "Tratamiento": status,
                }
            )
        return pd.DataFrame(rows)


def _key(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _clean_headers(columns: Iterable[Any]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for value in columns:
        base = str(value).strip() or "columna_sin_nombre"
        count = seen.get(base, 0) + 1
        seen[base] = count
        output.append(base if count == 1 else f"{base}__{count}")
    return output


def _blank(series: pd.Series) -> pd.Series:
    as_text = series.astype("string").str.strip()
    return series.isna() | as_text.isin(["", "nan", "None", "<NA>"])


def _identifier_value(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".15g")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    return re.sub(r"\.0$", "", text)


def _text_series(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.isin(["", "nan", "None", "<NA>"]), pd.NA)


def _parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", dayfirst=True)


def _chapter_group(value: Any) -> str:
    if pd.isna(value):
        return "Sin dato"
    raw = str(value).strip()
    if not raw:
        return "Sin dato"
    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()
    match = re.search(r"cap(?:itulo)?[\s_-]*(\d{1,2})", normalized)
    if match:
        return f"Capítulo {int(match.group(1))}"
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    if re.fullmatch(r"a\d*", compact):
        return "Sección A"
    if re.fullmatch(r"b\d*", compact):
        return "Sección B"
    if "novedad" in compact:
        return "Esquema de novedades"
    return "Otros"


def _state_group(value: Any) -> str:
    if pd.isna(value):
        return "Sin estado"
    raw = str(value).strip()
    if not raw:
        return "Sin estado"
    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()
    if "observacion correcta" in normalized:
        return "Justificada"
    if "no se debera considerar" in normalized:
        return "Regla o sistema"
    if "solventada tanto" in normalized:
        return "Corregida"
    if "corregido solo" in normalized or "revisada pero solo" in normalized:
        return "Corrección parcial"
    if "error nuevo" in normalized or "no revisada" in normalized:
        return "Pendiente"
    return "Otro estado"


def _cod_val_label(value: Any) -> str:
    if pd.isna(value):
        return "Sin clasificación"
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return f"Otro código: {value}"
    return {
        1: "1 · Error confirmado",
        2: "2 · Observación justificada",
        3: "3 · Regla o sistema",
    }.get(code, f"Otro código: {code}")


def workbook_sheet_names(file_bytes: bytes) -> list[str]:
    book = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    return list(book.sheet_names)


def read_workbook_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(
        BytesIO(file_bytes),
        sheet_name=sheet_name,
        dtype=object,
        engine="openpyxl",
    )


def prepare_history(source: pd.DataFrame) -> tuple[pd.DataFrame, PreparationReport]:
    if source is None or source.empty:
        raise ValueError("La hoja seleccionada no contiene registros.")

    df = source.copy()
    df.columns = _clean_headers(df.columns)
    source_columns = list(df.columns)
    normalized_source: dict[str, str] = {}
    for column in source_columns:
        normalized_source.setdefault(_key(column), column)

    mapped: dict[str, str] = {}
    rename_map: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        candidates = (canonical, *aliases)
        for alias in candidates:
            found = normalized_source.get(_key(alias))
            if found is not None and found not in rename_map:
                mapped[canonical] = found
                rename_map[found] = canonical
                break

    df = df.rename(columns=rename_map)
    missing_core = [column for column in CORE_COLUMNS if column not in df.columns]
    if missing_core:
        friendly = ", ".join(FRIENDLY_COLUMNS.get(c, c) for c in missing_core)
        raise ValueError(
            "No se puede construir el seguimiento porque faltan estas columnas: " + friendly
        )

    added_columns: list[str] = []
    defaults: dict[str, Any] = {
        "cz_final": "Sin dato",
        "fecha_malla": pd.NA,
        "id_error": pd.NA,
        "capitulo": "Sin dato",
        "mensaje": pd.NA,
        "observacion_encuestador": pd.NA,
        "observacion_critico": pd.NA,
        "observacion_revisor": pd.NA,
        "cod_val": pd.NA,
        "observaciones_zonal": pd.NA,
        "estado": "Sin estado",
        "ruc": pd.NA,
        "nombre_comercial": pd.NA,
        "razon_social": pd.NA,
        "encuestador_nombre": pd.NA,
        "division_carga": pd.NA,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
            added_columns.append(column)

    df["_source_row"] = np.arange(2, len(df) + 2)
    for column in ("id_empresa", "error_n", "id_error", "ruc"):
        df[column] = df[column].map(_identifier_value).astype("string")
    for column in (
        "cz_final",
        "capitulo",
        "mensaje",
        "observacion_encuestador",
        "observacion_critico",
        "observacion_revisor",
        "observaciones_zonal",
        "estado",
        "nombre_comercial",
        "razon_social",
        "encuestador_nombre",
        "division_carga",
    ):
        df[column] = _text_series(df[column])

    invalid_key = _blank(df["id_empresa"]) | _blank(df["error_n"])
    excluded_rows = int(invalid_key.sum())
    df = df.loc[~invalid_key].copy()

    generated_mask = _blank(df["id_error"])
    generated_count = int(generated_mask.sum())
    df.loc[generated_mask, "id_error"] = (
        df.loc[generated_mask, "id_empresa"].astype(str)
        + "__"
        + df.loc[generated_mask, "error_n"].astype(str)
    )

    df["cz_final"] = df["cz_final"].fillna("Sin dato")
    df["capitulo"] = df["capitulo"].fillna("Sin dato")
    df["estado"] = df["estado"].fillna("Sin estado")
    df["fecha_malla_dt"] = _parse_dates(df["fecha_malla"])
    df["fecha_corte"] = df["fecha_malla_dt"].dt.normalize()
    df["capitulo_agrupado"] = df["capitulo"].map(_chapter_group)
    df["cod_val_num"] = pd.to_numeric(df["cod_val"], errors="coerce")
    df["resultado_cod_val"] = df["cod_val_num"].map(_cod_val_label)
    df["resultado_estado"] = df["estado"].map(_state_group)
    for column in OBSERVATION_COLUMNS:
        df[f"tiene_{column}"] = ~_blank(df[column])

    pair_counts = (
        df.groupby("id_error", dropna=False)[["id_empresa", "error_n"]]
        .apply(lambda group: group.drop_duplicates().shape[0])
    )
    key_collisions = int((pair_counts > 1).sum())
    ids_per_pair = df.groupby(["id_empresa", "error_n"], dropna=False)["id_error"].nunique()
    pair_collisions = int((ids_per_pair > 1).sum())

    warnings: list[str] = []
    if excluded_rows:
        warnings.append(
            f"Se excluyeron {excluded_rows:,} registros sin ID de empresa o código de validación."
        )
    invalid_dates = int(df["fecha_malla_dt"].isna().sum())
    if invalid_dates:
        warnings.append(
            f"Hay {invalid_dates:,} registros sin una fecha de malla utilizable. "
            "Se incluyen en los resultados generales, pero no en los cálculos temporales."
        )
    if key_collisions or pair_collisions:
        warnings.append(
            "Se detectaron inconsistencias en la relación entre ID empresa, código de validación e ID de error."
        )

    missing_optional = [
        column for column in OPTIONAL_IDENTITY_COLUMNS if column not in mapped
    ]
    report = PreparationReport(
        source_columns=source_columns,
        mapped_columns=mapped,
        missing_optional_columns=missing_optional,
        added_standard_columns=added_columns,
        generated_id_error_rows=generated_count,
        raw_rows=len(source),
        usable_rows=len(df),
        excluded_rows=excluded_rows,
        invalid_dates=invalid_dates,
        valid_dates=int(df["fecha_malla_dt"].notna().sum()),
        key_collisions=key_collisions,
        pair_collisions=pair_collisions,
        warnings=warnings,
    )
    return df, report


def build_case_table(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    ordered = history.sort_values("_source_row").copy()
    carry_columns = [
        "id_empresa",
        "error_n",
        "cz_final",
        "capitulo",
        "capitulo_agrupado",
        "mensaje",
        "ruc",
        "nombre_comercial",
        "razon_social",
        "encuestador_nombre",
        "division_carga",
    ]
    for column in carry_columns:
        if column not in ordered.columns:
            continue
        ordered[column] = ordered[column].mask(_blank(ordered[column]), pd.NA)
    ordered[carry_columns] = ordered.groupby("id_error", sort=False)[carry_columns].ffill()

    # cod_val describes the resolution of one appearance. It must not be carried
    # into a later malla where the same validation reappears as a new case pendiente.
    ordered["_cycle_key"] = ordered["fecha_corte"].fillna(pd.Timestamp("1900-01-01"))
    ordered["cod_val_num"] = ordered.groupby(
        ["id_error", "_cycle_key"], sort=False, dropna=False
    )["cod_val_num"].ffill()

    latest = ordered.groupby("id_error", sort=False, as_index=False).tail(1).copy()
    latest = latest.rename(
        columns={
            "estado": "estado_final",
            "resultado_estado": "resultado_final",
            "cod_val_num": "cod_val_final",
        }
    )

    grouped = ordered.groupby("id_error", sort=False)
    stats = grouped.agg(
        registros_historicos=("id_error", "size"),
        apariciones_malla=("fecha_corte", "nunique"),
        primera_aparicion=("fecha_corte", "min"),
        ultima_aparicion=("fecha_corte", "max"),
        estados_distintos=("estado", "nunique"),
        variantes_mensaje=("mensaje", "nunique"),
        tiene_observacion_encuestador=("tiene_observacion_encuestador", "max"),
        tiene_observacion_critico=("tiene_observacion_critico", "max"),
        tiene_observacion_revisor=("tiene_observacion_revisor", "max"),
        tiene_observaciones_zonal=("tiene_observaciones_zonal", "max"),
    ).reset_index()

    keep = [
        "id_error",
        "id_empresa",
        "error_n",
        "cz_final",
        "capitulo",
        "capitulo_agrupado",
        "mensaje",
        "ruc",
        "nombre_comercial",
        "razon_social",
        "encuestador_nombre",
        "division_carga",
        "estado_final",
        "resultado_final",
        "cod_val_final",
    ]
    cases = latest[keep].merge(stats, on="id_error", how="left", validate="one_to_one")
    cases["validation_key"] = cases["error_n"].astype(str) + " | " + cases["capitulo"].astype(str)
    cases["resultado_cod_val"] = cases["cod_val_final"].map(_cod_val_label)
    cases["reaparece"] = cases["apariciones_malla"].gt(1)
    cases["dias_entre_apariciones"] = (
        cases["ultima_aparicion"] - cases["primera_aparicion"]
    ).dt.days
    cases.loc[~cases["reaparece"], "dias_entre_apariciones"] = np.nan
    cases["evaluada"] = cases["cod_val_final"].isin([1, 2, 3])
    cases["codigo_1"] = cases["cod_val_final"].eq(1)
    cases["codigo_2"] = cases["cod_val_final"].eq(2)
    cases["codigo_3"] = cases["cod_val_final"].eq(3)
    return cases.sort_values(["ultima_aparicion", "id_error"], ascending=[False, True], na_position="last")


def _mode_or_first(series: pd.Series) -> Any:
    valid = series.dropna().astype(str).str.strip()
    valid = valid[valid.ne("")]
    if valid.empty:
        return ""
    mode = valid.mode()
    return mode.iloc[0] if not mode.empty else valid.iloc[0]


def build_validation_catalog(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()

    catalog = (
        cases.groupby("validation_key", dropna=False)
        .agg(
            error_n=("error_n", "first"),
            capitulo=("capitulo", _mode_or_first),
            capitulo_agrupado=("capitulo_agrupado", _mode_or_first),
            mensaje=("mensaje", _mode_or_first),
            casos=("id_error", "nunique"),
            empresas=("id_empresa", "nunique"),
            zonales=("cz_final", "nunique"),
            evaluadas=("evaluada", "sum"),
            reapariciones=("reaparece", "sum"),
            codigo_1=("codigo_1", "sum"),
            codigo_2=("codigo_2", "sum"),
            codigo_3=("codigo_3", "sum"),
            casos_pendientes=("resultado_final", lambda s: int(s.eq("Pendiente").sum())),
            variantes_mensaje=("mensaje", "nunique"),
            primera_aparicion=("primera_aparicion", "min"),
            ultima_aparicion=("ultima_aparicion", "max"),
            casos_obs_encuestador=("tiene_observacion_encuestador", "sum"),
            casos_obs_critico=("tiene_observacion_critico", "sum"),
            casos_obs_revisor=("tiene_observacion_revisor", "sum"),
            casos_obs_zonal=("tiene_observaciones_zonal", "sum"),
        )
        .reset_index()
    )
    catalog["pct_reaparicion"] = catalog["reapariciones"].div(catalog["casos"]).mul(100)
    denominator = catalog["evaluadas"].replace(0, np.nan)
    catalog["pct_codigo_1"] = catalog["codigo_1"].div(denominator).mul(100)
    catalog["pct_codigo_2"] = catalog["codigo_2"].div(denominator).mul(100)
    catalog["pct_codigo_3"] = catalog["codigo_3"].div(denominator).mul(100)
    catalog["pct_obs_encuestador"] = catalog["casos_obs_encuestador"].div(catalog["casos"]).mul(100)
    catalog["pct_obs_critico"] = catalog["casos_obs_critico"].div(catalog["casos"]).mul(100)
    catalog["pct_obs_revisor"] = catalog["casos_obs_revisor"].div(catalog["casos"]).mul(100)
    catalog["pct_obs_zonal"] = catalog["casos_obs_zonal"].div(catalog["casos"]).mul(100)
    catalog["es_capitulo_5"] = catalog["capitulo_agrupado"].eq("Capítulo 5")
    return catalog.sort_values(["casos", "pct_reaparicion"], ascending=[False, False])


def dataset_fingerprint(file_bytes: bytes, sheet_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(file_bytes)
    digest.update(sheet_name.encode("utf-8"))
    return digest.hexdigest()[:16]


def friendly_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    result = frame[available].copy()
    result = result.rename(columns={column: FRIENDLY_COLUMNS.get(column, column) for column in available})
    return result
