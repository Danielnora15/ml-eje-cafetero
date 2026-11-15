"""
01_preparacion_datos.py
Ingeniería de datos (limpieza, depuración y creación de variables) para los dos proyectos
de Machine Learning enfocados en el Eje Cafetero (Caldas, Risaralda y Quindío).

  Proyecto 1: Saber 11 Eje Cafetero
      ¿Qué estudiantes tendrán bajo desempeño (puntaje global < 220) en la prueba Saber 11?
  Proyecto 2: Víctimas de siniestros viales en el Eje Cafetero
      ¿Qué víctimas de siniestros viales tienen mayor probabilidad de fallecer?

Fuentes (datos.gov.co):
  kgxf-xxbe  Resultados únicos Saber 11 (ICFES)
  72sg-cybi  Reporte de lesiones personales y en accidente de tránsito (Policía Nacional)
  ha6j-pa2r  Homicidios en accidente de tránsito (Policía Nacional)

Entradas : data/raw/*.csv
Salidas  : data/processed/*_limpio.csv  y  outputs/calidad_datos.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parents[1]
RAW, PROC, OUT = BASE / "data" / "raw", BASE / "data" / "processed", BASE / "outputs"
PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

UMBRAL_BAJO = 220  # puntaje global (0-500) por debajo del cual se considera desempeño bajo


# ---------------------------------------------------------------------------
# PROYECTO 1 — SABER 11 EJE CAFETERO
# ---------------------------------------------------------------------------
EDUCACION = {  # escala ordinal del nivel educativo de los padres
    "Ninguno": 0, "Primaria incompleta": 1, "Primaria completa": 2,
    "Secundaria (Bachillerato) incompleta": 3, "Secundaria (Bachillerato) completa": 4,
    "Técnica o tecnológica incompleta": 5, "Técnica o tecnológica completa": 6,
    "Educación profesional incompleta": 7, "Educación profesional completa": 8, "Postgrado": 9,
}
CUARTOS = {"Uno": 1, "Dos": 2, "Tres": 3, "Cuatro": 4, "Cinco": 5, "Seis o mas": 6}
PERSONAS = {"1 a 2": 1.5, "3 a 4": 3.5, "5 a 6": 5.5, "7 a 8": 7.5, "9 o más": 9.5}


def preparar_saber11():
    df = pd.read_csv(RAW / "saber11_eje_cafetero.csv", low_memory=False)
    log = {"registros_originales": len(df), "columnas_originales": df.shape[1],
           "periodos": {str(k): int(v) for k, v in df["periodo"].value_counts().sort_index().items()}}
    log["nulos_por_columna"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}

    # 1. Registros inválidos
    sin_puntaje = df["punt_global"].isna() | (df["punt_global"] <= 0)
    log["eliminados_puntaje_invalido"] = int(sin_puntaje.sum())
    df = df[~sin_puntaje].copy()

    # 2. Duplicados exactos: la fuente publica DOS veces cada estudiante de los periodos 2019-2 y
    #    2022-2 (verificado con estu_consecutivo: 55.246 filas vs 27.564 estudiantes en 2019-2).
    log["duplicados_exactos"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 3. Variable objetivo
    df["bajo_desempeno"] = (df["punt_global"] < UMBRAL_BAJO).astype(int)
    df["anio"] = df["periodo"].astype(str).str[:4].astype(int)

    # 4. Edad al presentar la prueba (fecha de nacimiento dd/mm/aaaa)
    nac = pd.to_datetime(df["estu_fechanacimiento"], format="%d/%m/%Y", errors="coerce")
    df["edad"] = df["anio"] - nac.dt.year
    edad_invalida = (df["edad"] < 13) | (df["edad"] > 60) | df["edad"].isna()
    log["edades_invalidas_imputadas"] = int(edad_invalida.sum())
    df.loc[edad_invalida, "edad"] = np.nan
    df["edad"] = df["edad"].fillna(df["edad"].median())
    df["extraedad"] = (df["edad"] >= 19).astype(int)

    # 5. Nulos en variables familiares -> categoría explícita "Sin información"
    fami = ["fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre",
            "fami_cuartoshogar", "fami_personashogar", "fami_tieneautomovil",
            "fami_tienecomputador", "fami_tieneinternet", "fami_tienelavadora"]
    for c in fami:
        df[c] = df[c].fillna("Sin información")
    df["cole_bilingue"] = df["cole_bilingue"].fillna("Sin información")
    df["cole_caracter"] = df["cole_caracter"].fillna("NO APLICA")
    df["estu_genero"] = df["estu_genero"].fillna("Sin información")
    df[["estu_mcpio_reside", "estu_depto_reside"]] = df[["estu_mcpio_reside", "estu_depto_reside"]].fillna("Sin información")
    # Puntajes por área: no se usan como predictores (se conocen junto con el global), sólo se reportan
    areas = ["punt_ingles", "punt_matematicas", "punt_sociales_ciudadanas", "punt_c_naturales", "punt_lectura_critica"]
    df[areas] = df[areas].fillna(df[areas].median())

    # 6. Ingeniería de variables
    df["estrato_num"] = df["fami_estratovivienda"].str.extract(r"(\d)")[0].astype(float)
    df["estrato_num"] = df["estrato_num"].fillna(0)  # 0 = sin estrato / sin información
    edu_m = df["fami_educacionmadre"].map(EDUCACION)
    edu_p = df["fami_educacionpadre"].map(EDUCACION)
    df["edu_max_padres"] = pd.concat([edu_m, edu_p], axis=1).max(axis=1).fillna(-1)
    df["madre_profesional"] = (edu_m >= 8).astype(int)
    bienes = ["fami_tieneautomovil", "fami_tienecomputador", "fami_tieneinternet", "fami_tienelavadora"]
    df["indice_bienes"] = (df[bienes] == "Si").sum(axis=1)
    df["conectividad"] = ((df["fami_tieneinternet"] == "Si") & (df["fami_tienecomputador"] == "Si")).astype(int)
    cuartos = df["fami_cuartoshogar"].map(CUARTOS)
    personas = df["fami_personashogar"].map(PERSONAS)
    df["personas_por_cuarto"] = (personas / cuartos).round(2)
    df["hacinamiento"] = (df["personas_por_cuarto"] > 3).astype(int)  # criterio DANE: > 3 personas/cuarto
    df["personas_por_cuarto"] = df["personas_por_cuarto"].fillna(df["personas_por_cuarto"].median())
    df["reside_otro_municipio"] = (df["estu_mcpio_reside"].str.upper() != df["cole_mcpio_ubicacion"].str.upper()).astype(int)
    df["jornada_no_regular"] = df["cole_jornada"].isin(["NOCHE", "SABATINA"]).astype(int)
    df["cole_cod_dane_establecimiento"] = df["cole_cod_dane_establecimiento"].astype(str)

    df = df.drop(columns=["estu_fechanacimiento"])
    log["registros_finales"] = len(df)
    log["columnas_finales"] = df.shape[1]
    log["tasa_bajo_desempeno"] = round(float(df["bajo_desempeno"].mean()), 4)
    log["umbral_bajo_desempeno"] = UMBRAL_BAJO
    log["colegios"] = int(df["cole_cod_dane_establecimiento"].nunique())
    log["municipios"] = int(df["cole_mcpio_ubicacion"].nunique())
    log["nulos_finales"] = int(df.isna().sum().sum())
    df.to_csv(PROC / "saber11_eje_cafetero_limpio.csv", index=False)
    return log


# ---------------------------------------------------------------------------
# PROYECTO 2 — VÍCTIMAS DE SINIESTROS VIALES EJE CAFETERO
# ---------------------------------------------------------------------------
def festivos_colombia(anios):
    """Festivos de Colombia (Ley Emiliani) calculados sin librerías externas."""
    from datetime import date, timedelta

    def pascua(y):
        a, b, c = y % 19, y // 100, y % 100
        d, e = b // 4, b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = c // 4, c % 4
        l_ = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l_) // 451
        mes = (h + l_ - 7 * m + 114) // 31
        dia = ((h + l_ - 7 * m + 114) % 31) + 1
        return date(y, mes, dia)

    def lunes(d):  # traslada al lunes siguiente
        return d + timedelta(days=(7 - d.weekday()) % 7)

    fest = set()
    for y in anios:
        p = pascua(y)
        fest |= {date(y, 1, 1), date(y, 5, 1), date(y, 7, 20), date(y, 8, 7), date(y, 12, 8), date(y, 12, 25)}
        fest |= {lunes(date(y, 1, 6)), lunes(date(y, 3, 19)), lunes(date(y, 6, 29)), lunes(date(y, 8, 15)),
                 lunes(date(y, 10, 12)), lunes(date(y, 11, 1)), lunes(date(y, 11, 11))}
        fest |= {p - timedelta(days=3), p - timedelta(days=2), lunes(p + timedelta(days=39)),
                 lunes(p + timedelta(days=60)), lunes(p + timedelta(days=68))}
    return pd.to_datetime(sorted(fest))


def preparar_victimas():
    les = pd.read_csv(RAW / "lesionados_transito_eje_cafetero.csv")
    fal = pd.read_csv(RAW / "fallecidos_transito_eje_cafetero.csv").rename(columns={"grupo_etar_o": "grupo_etario"})
    log = {"registros_lesionados": len(les), "registros_fallecidos": len(fal),
           "registros_originales": len(les) + len(fal)}

    # 1. Unificar las dos fuentes y crear la variable objetivo
    les["fallecido"], fal["fallecido"] = 0, 1
    df = pd.concat([les, fal], ignore_index=True)

    # 2. Un registro de lesionados puede agrupar varias víctimas (columna "cantidad").
    #    Se expande para que cada fila sea UNA víctima.
    log["registros_con_cantidad_mayor_1"] = int((df["cantidad"] > 1).sum())
    df = df.loc[df.index.repeat(df["cantidad"].clip(lower=1))].reset_index(drop=True)
    log["victimas_tras_expandir"] = len(df)

    # 3. Fechas
    df["fecha"] = pd.to_datetime(df["fecha_hecho"], format="%d/%m/%Y", errors="coerce")
    log["fechas_invalidas"] = int(df["fecha"].isna().sum())
    df = df.dropna(subset=["fecha"])

    # 4. Medios no viales (contundentes, escopolamina) y género no reportado -> se eliminan
    no_vial = ~df["armas_medios"].isin(["VEHICULO", "MOTO", "BICICLETA", "SIN EMPLEO DE ARMAS", "NO REPORTADO", "TREN"])
    log["eliminados_medio_no_vial"] = int(no_vial.sum())
    df = df[~no_vial]
    sin_genero = ~df["genero"].isin(["MASCULINO", "FEMENINO"])
    log["eliminados_sin_genero"] = int(sin_genero.sum())
    df = df[~sin_genero]

    # 5. Sesgo de registro: "NO REPORTADO" es mucho más frecuente en el registro de fallecidos
    #    (18 %) que en el de lesionados (4 %). Dejarlo como categoría haría que el modelo aprenda
    #    la forma de registrar y no el riesgo real (fuga de información). Se unifica con
    #    "SIN EMPLEO DE ARMAS" en una sola categoría "No especificado". TREN (2 casos) -> idem.
    df["medio"] = df["armas_medios"].replace({
        "VEHICULO": "Vehículo", "MOTO": "Motocicleta", "BICICLETA": "Bicicleta",
        "SIN EMPLEO DE ARMAS": "No especificado", "NO REPORTADO": "No especificado", "TREN": "No especificado"})

    # 6. Normalizar textos
    df["departamento"] = df["departamento"].str.upper().str.replace("QUINDIO", "QUINDÍO")
    df["es_capital"] = df["municipio"].str.contains(r"\(CT\)").astype(int)
    df["municipio"] = df["municipio"].str.replace(r"\s*\(CT\)", "", regex=True).str.strip()
    df["grupo_etario"] = df["grupo_etario"].str.capitalize()
    df["genero"] = df["genero"].str.capitalize()

    # 7. Ingeniería de variables temporales
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["dia_semana"] = df["fecha"].dt.dayofweek  # 0 = lunes
    df["fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    fest = festivos_colombia(range(df["anio"].min(), df["anio"].max() + 1))
    df["festivo"] = df["fecha"].isin(fest).astype(int)
    df["puente_festivo"] = (df["festivo"] | df["fecha"].isin(fest - pd.Timedelta(days=3)) |
                            df["fecha"].isin(fest - pd.Timedelta(days=2))).astype(int)
    df["temporada_vacaciones"] = df["mes"].isin([1, 6, 7, 12]).astype(int)
    df["dia_quincena"] = df["fecha"].dt.day.isin([15, 16, 30, 31, 1]).astype(int)
    df["usuario_vulnerable"] = df["medio"].isin(["Motocicleta", "Bicicleta"]).astype(int)

    df = df.drop(columns=["fecha_hecho", "armas_medios", "descripci_n_conducta", "cantidad", "codigo_dane"])
    df = df.sort_values("fecha").reset_index(drop=True)
    log["registros_finales"] = len(df)
    log["columnas_finales"] = df.shape[1]
    log["fallecidos"] = int(df["fallecido"].sum())
    log["tasa_letalidad"] = round(float(df["fallecido"].mean()), 4)
    log["rango_fechas"] = [str(df["fecha"].min().date()), str(df["fecha"].max().date())]
    log["municipios"] = int(df["municipio"].nunique())
    log["nulos_finales"] = int(df.isna().sum().sum())
    df.to_csv(PROC / "victimas_viales_eje_cafetero_limpio.csv", index=False)
    return log


if __name__ == "__main__":
    reporte = {"saber11": preparar_saber11(), "victimas": preparar_victimas()}
    (OUT / "calidad_datos.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(reporte, indent=2, ensure_ascii=False))
