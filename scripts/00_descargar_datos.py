"""
00_descargar_datos.py
Descarga los datos abiertos de datos.gov.co (API Socrata) para el Eje Cafetero
y los guarda en data/raw/, listos para 01_preparacion_datos.py.

  kgxf-xxbe  Resultados únicos Saber 11 (ICFES), periodos 2019-2022
  72sg-cybi  Lesiones personales y en accidente de tránsito (Policía Nacional)
  ha6j-pa2r  Homicidios en accidente de tránsito (Policía Nacional)

Los datos no se suben al repositorio: son públicos, pesan ~45 MB y la fuente se actualiza.
Uso:  python scripts/00_descargar_datos.py
"""
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

API = "https://www.datos.gov.co/resource/{id}.csv"
PAGINA = 50000

SABER_COLUMNAS = [
    "periodo", "cole_area_ubicacion", "cole_bilingue", "cole_calendario", "cole_caracter",
    "cole_cod_dane_establecimiento", "cole_depto_ubicacion", "cole_genero", "cole_jornada",
    "cole_mcpio_ubicacion", "cole_naturaleza", "cole_nombre_establecimiento", "estu_fechanacimiento",
    "estu_genero", "estu_mcpio_reside", "estu_depto_reside", "fami_cuartoshogar",
    "fami_educacionmadre", "fami_educacionpadre", "fami_estratovivienda", "fami_personashogar",
    "fami_tieneautomovil", "fami_tienecomputador", "fami_tieneinternet", "fami_tienelavadora",
    "punt_ingles", "punt_matematicas", "punt_sociales_ciudadanas", "punt_c_naturales",
    "punt_lectura_critica", "punt_global",
]
VIAL_COLUMNAS = ["departamento", "municipio", "codigo_dane", "armas_medios", "fecha_hecho",
                 "genero", "grupo_etario", "descripci_n_conducta", "cantidad"]

DESCARGAS = [
    {
        "archivo": "saber11_eje_cafetero.csv",
        "id": "kgxf-xxbe",
        "select": SABER_COLUMNAS,
        "where": "cole_depto_ubicacion in ('CALDAS','RISARALDA','QUINDIO') AND "
                 "periodo in ('20191','20194','20201','20211','20221','20224')",
    },
    {
        "archivo": "lesionados_transito_eje_cafetero.csv",
        "id": "72sg-cybi",
        "select": VIAL_COLUMNAS,
        "where": "departamento in ('CALDAS','RISARALDA','QUINDÍO') AND "
                 "descripci_n_conducta like '%ACCIDENTE DE TRANSITO%'",
    },
    {
        "archivo": "fallecidos_transito_eje_cafetero.csv",
        "id": "ha6j-pa2r",
        # en esta fuente la columna se llama grupo_etar_o; 01_preparacion_datos.py la renombra
        "select": [c if c != "grupo_etario" else "grupo_etar_o" for c in VIAL_COLUMNAS],
        "where": "departamento in ('CALDAS','RISARALDA','QUINDÍO')",
    },
]


def bajar(url, intentos=3):
    for i in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # red inestable: reintenta
            if i == intentos - 1:
                raise
            print(f"   reintento por error: {e}")
            time.sleep(3)


def descargar(d):
    destino = RAW / d["archivo"]
    filas, offset, encabezado = [], 0, None
    while True:
        params = {"$select": ",".join(d["select"]), "$where": d["where"],
                  "$order": ":id", "$limit": PAGINA, "$offset": offset}
        url = API.format(id=d["id"]) + "?" + urllib.parse.urlencode(params)
        lineas = bajar(url).splitlines()
        encabezado = encabezado or lineas[0]
        datos = lineas[1:]
        filas.extend(datos)
        print(f"   {d['archivo']}: {len(filas):,} filas")
        if len(datos) < PAGINA:
            break
        offset += PAGINA
    destino.write_text("\n".join([encabezado, *filas]) + "\n", encoding="utf-8")
    return len(filas)


if __name__ == "__main__":
    for d in DESCARGAS:
        print(f"Descargando {d['id']} ...")
        n = descargar(d)
        print(f"   listo: {n:,} filas -> data/raw/{d['archivo']}")
