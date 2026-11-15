"""
02_modelado_validacion.py
Entrena y valida 3 algoritmos por proyecto, elige el mejor y genera:
  - outputs/metricas.json            (métricas de validación)
  - outputs/modelo_*.joblib          (modelo entrenado = entregable)
  - outputs/tablero_datos.json       (datos agregados para el tablero)
  - data/processed/*_scoring.csv     (predicciones sobre el conjunto de prueba)

Validación (ambos proyectos): partición TEMPORAL + validación cruzada 5-fold en entrenamiento.
  Saber 11 -> entrena con 2019-2021, prueba con 2022 (el modelo se usaría con cohortes futuras).
  Víctimas -> entrena con 2010-2022, prueba con 2023-2026.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parents[1]
PROC, OUT = BASE / "data" / "processed", BASE / "outputs"
SEED = 42


def construir_modelos(cat, num, te=()):
    trans = [("num", StandardScaler(), num),
             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat)]
    if te:  # codificación por objetivo (con validación cruzada interna) para alta cardinalidad
        trans.append(("te", TargetEncoder(target_type="binary",
                                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED)), list(te)))
    pre = ColumnTransformer(trans)
    return {
        "Regresión Logística": Pipeline([("pre", pre), ("clf", LogisticRegression(
            max_iter=3000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("pre", pre), ("clf", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, class_weight="balanced", n_jobs=-1,
            random_state=SEED))]),
        "Gradient Boosting": Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=300, class_weight="balanced", random_state=SEED))]),
    }


def evaluar(y, prob, umbral=0.5):
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    # Lift en el 10 % superior: cuántas veces más casos positivos hay entre el 10 % de registros
    # con mayor probabilidad, comparado con escoger al azar.
    top = np.argsort(-prob)[: max(1, int(len(prob) * 0.10))]
    y_arr = np.asarray(y)
    lift10 = y_arr[top].mean() / y_arr.mean()
    return {
        "lift_top10": round(float(lift10), 2),
        "captura_top10": round(float(y_arr[top].sum() / y_arr.sum()), 4),
        "roc_auc": round(roc_auc_score(y, prob), 4),
        "pr_auc": round(average_precision_score(y, prob), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "matriz_confusion": {"VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp)},
    }


def entrenar_proyecto(nombre, X_tr, y_tr, X_te, y_te, cat, num, te=()):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    resultados, ajustados = {}, {}
    for alg, modelo in construir_modelos(cat, num, te).items():
        auc_cv = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
        modelo.fit(X_tr, y_tr)
        prob = modelo.predict_proba(X_te)[:, 1]
        res = evaluar(y_te, prob)
        res["cv_auc_media"] = round(float(auc_cv.mean()), 4)
        res["cv_auc_std"] = round(float(auc_cv.std()), 4)
        resultados[alg], ajustados[alg] = res, (modelo, prob)
        print(f"[{nombre}] {alg:20s} CV-AUC={res['cv_auc_media']:.4f}±{res['cv_auc_std']:.4f} "
              f"Test-AUC={res['roc_auc']:.4f} PR-AUC={res['pr_auc']:.4f} P={res['precision']:.3f} "
              f"R={res['recall']:.3f} F1={res['f1']:.3f}", flush=True)
    mejor = max(resultados, key=lambda a: resultados[a]["roc_auc"])
    modelo, prob = ajustados[mejor]
    muestra = X_te.sample(min(4000, len(X_te)), random_state=SEED)
    imp = permutation_importance(modelo, muestra, y_te.loc[muestra.index], scoring="roc_auc",
                                 n_repeats=5, random_state=SEED, n_jobs=1)
    importancias = (pd.Series(imp.importances_mean, index=muestra.columns)
                    .sort_values(ascending=False).head(10).round(4))
    return resultados, mejor, modelo, prob, importancias


def segmento(p):
    return pd.cut(p, bins=[-0.01, 0.3, 0.6, 1.0], labels=["Bajo", "Medio", "Alto"]).astype(str)


def tasa_por(df, col, objetivo, orden=None, min_n=1):
    t = df.groupby(col, observed=True)[objetivo].agg(["mean", "count"]).reset_index()
    t = t[t["count"] >= min_n]
    if orden:
        t[col] = pd.Categorical(t[col], categories=orden, ordered=True)
        t = t.dropna(subset=[col]).sort_values(col)
    else:
        t = t.sort_values("mean", ascending=False)
    return {"labels": t[col].astype(str).tolist(), "tasa": (t["mean"] * 100).round(1).tolist(),
            "n": t["count"].astype(int).tolist()}


def resumen_modelo(res, mejor, n_tr, n_te, imp, extra=None):
    d = {"resultados": res, "mejor_modelo": mejor, "n_train": int(n_tr), "n_test": int(n_te),
         "importancias": imp.to_dict()}
    d.update(extra or {})
    return d


# ---------------------------------------------------------------------------
def proyecto_saber11():
    df = pd.read_csv(PROC / "saber11_eje_cafetero_limpio.csv", low_memory=False,
                     dtype={"cole_cod_dane_establecimiento": str})
    cat = ["cole_area_ubicacion", "cole_bilingue", "cole_calendario", "cole_caracter", "cole_genero",
           "cole_jornada", "cole_naturaleza", "cole_depto_ubicacion", "estu_genero",
           "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre", "fami_cuartoshogar",
           "fami_personashogar", "fami_tieneautomovil", "fami_tienecomputador", "fami_tieneinternet",
           "fami_tienelavadora"]
    num = ["edad", "extraedad", "estrato_num", "edu_max_padres", "madre_profesional", "indice_bienes",
           "conectividad", "personas_por_cuarto", "hacinamiento", "reside_otro_municipio",
           "jornada_no_regular"]
    te = ["cole_cod_dane_establecimiento", "cole_mcpio_ubicacion"]
    tr, tst = df[df.anio <= 2021], df[df.anio == 2022]
    X_tr, y_tr = tr[cat + num + te], tr["bajo_desempeno"]
    X_te, y_te = tst[cat + num + te], tst["bajo_desempeno"]
    res, mejor, modelo, prob, imp = entrenar_proyecto("Saber11", X_tr, y_tr, X_te, y_te, cat, num, te)
    joblib.dump(modelo, OUT / "modelo_saber11_bajo_desempeno.joblib", compress=3)

    score = tst[["cole_nombre_establecimiento", "cole_cod_dane_establecimiento", "cole_mcpio_ubicacion",
                 "cole_depto_ubicacion", "cole_naturaleza", "cole_area_ubicacion", "fami_estratovivienda",
                 "punt_global", "bajo_desempeno"]].copy()
    score["prob_bajo_desempeno"] = prob.round(4)
    score["riesgo"] = segmento(score["prob_bajo_desempeno"])
    score.to_csv(PROC / "saber11_scoring_2022.csv", index=False)

    # Colegios prioritarios: promedio de probabilidad de sus estudiantes en 2022
    col = (score.groupby(["cole_nombre_establecimiento", "cole_mcpio_ubicacion", "cole_naturaleza",
                          "cole_area_ubicacion"])
           .agg(estudiantes=("riesgo", "size"), prob_media=("prob_bajo_desempeno", "mean"),
                pct_alto=("riesgo", lambda s: (s == "Alto").mean()),
                pct_bajo_real=("bajo_desempeno", "mean"), puntaje_medio=("punt_global", "mean"))
           .reset_index())
    col = col[col.estudiantes >= 30].sort_values("prob_media", ascending=False)
    top_col = col.head(15).round({"prob_media": 3, "pct_alto": 3, "pct_bajo_real": 3, "puntaje_medio": 0})

    mun = (score.groupby("cole_mcpio_ubicacion")
           .agg(n=("riesgo", "size"), prob=("prob_bajo_desempeno", "mean"), real=("bajo_desempeno", "mean"))
           .reset_index())
    mun = mun[mun.n >= 150].sort_values("prob", ascending=False).head(12)

    estratos = ["Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6"]
    educ = ["Ninguno", "Primaria incompleta", "Primaria completa", "Secundaria (Bachillerato) incompleta",
            "Secundaria (Bachillerato) completa", "Técnica o tecnológica completa",
            "Educación profesional completa", "Postgrado"]
    df["conectividad_txt"] = np.where(df.conectividad == 1, "Internet y computador", "Sin internet o sin computador")
    alto = score[score.riesgo == "Alto"]
    b = res[mejor]
    tablero = {
        "kpis": {"estudiantes": int(len(df)), "colegios": int(df.cole_cod_dane_establecimiento.nunique()),
                 "municipios": int(df.cole_mcpio_ubicacion.nunique()),
                 "tasa_bajo": round(df.bajo_desempeno.mean() * 100, 1),
                 "puntaje_medio": round(df.punt_global.mean(), 1),
                 "estudiantes_prueba": int(len(score)), "alto_riesgo_prueba": int(len(alto)),
                 "recall": b["recall"], "precision": b["precision"]},
        "por_estrato": tasa_por(df, "fami_estratovivienda", "bajo_desempeno", estratos),
        "por_naturaleza": tasa_por(df, "cole_naturaleza", "bajo_desempeno"),
        "por_area": tasa_por(df, "cole_area_ubicacion", "bajo_desempeno"),
        "por_departamento": tasa_por(df, "cole_depto_ubicacion", "bajo_desempeno"),
        "por_educacion_madre": tasa_por(df, "fami_educacionmadre", "bajo_desempeno", educ),
        "por_conectividad": tasa_por(df, "conectividad_txt", "bajo_desempeno"),
        "por_jornada": tasa_por(df, "cole_jornada", "bajo_desempeno"),
        "municipios": {"labels": mun.cole_mcpio_ubicacion.str.title().tolist(),
                       "prob": (mun.prob * 100).round(1).tolist(), "real": (mun.real * 100).round(1).tolist(),
                       "n": mun.n.astype(int).tolist()},
        "top_colegios": top_col.to_dict(orient="records"),
        "segmentos": score.riesgo.value_counts().reindex(["Alto", "Medio", "Bajo"]).fillna(0).astype(int).to_dict(),
        "importancias": {"labels": imp.index.tolist(), "valores": imp.values.tolist()},
        "hist_prob": np.histogram(prob, bins=10, range=(0, 1))[0].tolist(),
    }
    extra = {"corte_temporal": "entrena 2019-2021 / prueba 2022", "colegios_priorizados": int(len(col))}
    return resumen_modelo(res, mejor, len(X_tr), len(X_te), imp, extra), tablero


def proyecto_victimas():
    df = pd.read_csv(PROC / "victimas_viales_eje_cafetero_limpio.csv", parse_dates=["fecha"])
    cat = ["departamento", "municipio", "medio", "genero", "grupo_etario"]
    num = ["es_capital", "mes", "dia_semana", "fin_de_semana", "festivo", "puente_festivo",
           "temporada_vacaciones", "dia_quincena", "usuario_vulnerable"]
    tr, tst = df[df.anio <= 2022], df[df.anio >= 2023]
    X_tr, y_tr = tr[cat + num], tr["fallecido"]
    X_te, y_te = tst[cat + num], tst["fallecido"]
    res, mejor, modelo, prob, imp = entrenar_proyecto("Victimas", X_tr, y_tr, X_te, y_te, cat, num)
    joblib.dump(modelo, OUT / "modelo_victimas_fatales.joblib", compress=3)

    score = tst[["fecha", "departamento", "municipio", "medio", "genero", "grupo_etario", "fin_de_semana",
                 "festivo", "es_capital", "fallecido"]].copy()
    score["prob_fatal"] = prob.round(4)
    score["riesgo"] = segmento(score["prob_fatal"])
    score.to_csv(PROC / "victimas_scoring_2023_2026.csv", index=False)

    # Perfiles de mayor riesgo (combinaciones de variables operativas)
    score["tipo_municipio"] = np.where(score.es_capital == 1, "Capital", "Otro municipio")
    score["momento"] = np.where(score.fin_de_semana == 1, "Fin de semana", "Entre semana")
    perf = (score.groupby(["medio", "genero", "grupo_etario", "tipo_municipio", "momento"])
            .agg(victimas=("riesgo", "size"), prob_media=("prob_fatal", "mean"), letalidad_real=("fallecido", "mean"))
            .reset_index())
    perf = perf[perf.victimas >= 40].sort_values("prob_media", ascending=False).head(12)

    mun = df.groupby("municipio").agg(n=("fallecido", "size"), tasa=("fallecido", "mean"),
                                      fallecidos=("fallecido", "sum")).reset_index()
    mun = mun[mun.n >= 300].sort_values("tasa", ascending=False).head(12)
    anual = df.groupby("anio").agg(fallecidos=("fallecido", "sum"), victimas=("fallecido", "size")).reset_index()
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    df["dia_txt"] = df.dia_semana.map(dict(enumerate(dias)))
    alto = score[score.riesgo == "Alto"]
    b = res[mejor]
    tablero = {
        "kpis": {"victimas": int(len(df)), "fallecidos": int(df.fallecido.sum()),
                 "letalidad": round(df.fallecido.mean() * 100, 2), "municipios": int(df.municipio.nunique()),
                 "victimas_prueba": int(len(score)), "fallecidos_prueba": int(score.fallecido.sum()),
                 "alto_riesgo_prueba": int(len(alto)), "fallecidos_en_alto": int(alto.fallecido.sum()),
                 "recall": b["recall"], "precision": b["precision"]},
        "por_medio": tasa_por(df, "medio", "fallecido"),
        "por_grupo": tasa_por(df, "grupo_etario", "fallecido", ["Menores", "Adolescentes", "Adultos"]),
        "por_genero": tasa_por(df, "genero", "fallecido"),
        "por_departamento": tasa_por(df, "departamento", "fallecido"),
        "por_dia": tasa_por(df, "dia_txt", "fallecido", dias),
        "por_festivo": tasa_por(df.assign(f=np.where(df.festivo == 1, "Festivo", "No festivo")), "f", "fallecido"),
        "municipios": {"labels": mun.municipio.tolist(), "tasa": (mun.tasa * 100).round(1).tolist(),
                       "n": mun.n.astype(int).tolist(), "fallecidos": mun.fallecidos.astype(int).tolist()},
        "anual": {"labels": anual.anio.astype(str).tolist(), "fallecidos": anual.fallecidos.astype(int).tolist(),
                  "victimas": anual.victimas.astype(int).tolist()},
        "perfiles": perf.round({"prob_media": 3, "letalidad_real": 3}).to_dict(orient="records"),
        "segmentos": score.riesgo.value_counts().reindex(["Alto", "Medio", "Bajo"]).fillna(0).astype(int).to_dict(),
        "importancias": {"labels": imp.index.tolist(), "valores": imp.values.tolist()},
        "hist_prob": np.histogram(prob, bins=10, range=(0, 1))[0].tolist(),
    }
    extra = {"corte_temporal": "entrena 2010-2022 / prueba 2023-2026"}
    return resumen_modelo(res, mejor, len(X_tr), len(X_te), imp, extra), tablero


if __name__ == "__main__":
    m_s, d_s = proyecto_saber11()
    m_v, d_v = proyecto_victimas()
    (OUT / "metricas.json").write_text(json.dumps({"saber11": m_s, "victimas": m_v}, indent=2,
                                                  ensure_ascii=False), encoding="utf-8")
    (OUT / "tablero_datos.json").write_text(json.dumps({"saber11": d_s, "victimas": d_v},
                                                       ensure_ascii=False, default=str), encoding="utf-8")
    print("Mejor Saber 11:", m_s["mejor_modelo"], "| Mejor Víctimas:", m_v["mejor_modelo"])
