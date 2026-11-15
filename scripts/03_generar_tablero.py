"""
03_generar_tablero.py
Inserta los resultados del modelado en la plantilla y genera outputs/tablero.html
(se abre con doble clic; requiere internet para cargar Chart.js).
"""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"

plantilla = (BASE / "scripts" / "tablero_plantilla.html").read_text(encoding="utf-8")
datos = (OUT / "tablero_datos.json").read_text(encoding="utf-8")
metricas = (OUT / "metricas.json").read_text(encoding="utf-8")
html = plantilla.replace("/*__DATA__*/", datos).replace("/*__MET__*/", metricas)
(OUT / "tablero.html").write_text(html, encoding="utf-8")
print("Tablero generado:", OUT / "tablero.html")
