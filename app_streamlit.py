"""
Interfaz en la nube — Detección de anomalías en válvulas industriales
=====================================================================
Despliegue típico: Streamlit Community Cloud (app pública).

Ejecución local:
  streamlit run app_streamlit.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
OUT = BASE / "salidas"
DATA = BASE / "ACTIVIDAD_2"
MODEL_PATH = OUT / "modelo_autoencoder.joblib"

st.set_page_config(
    page_title="Anomalías en válvulas | Autoencoder",
    page_icon="🔧",
    layout="wide",
)


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_one(artifact, values: dict) -> dict:
    sensors = artifact["sensors"]
    x = np.array([[values[s] for s in sensors]], dtype=float)
    xs = artifact["scaler"].transform(x)
    recon = artifact["model"].predict(xs)
    err = float(np.mean((xs - recon) ** 2))
    z = (err - artifact["err_val_mean"]) / artifact["err_val_std"]
    anom = err > artifact["umbral"]
    contrib = ((xs - recon) ** 2).ravel()
    contrib_pct = 100 * contrib / contrib.sum()
    return {
        "error": err,
        "z": float(z),
        "anomalia": anom,
        "contrib": {s: float(contrib_pct[i]) for i, s in enumerate(sensors)},
    }


def main() -> None:
    st.title("Detección de anomalías en válvulas industriales")
    st.caption(
        "Modelo Autoencoder (Deep Learning) entrenado solo con válvulas normales. "
        "R1-A2-S4 · Aplicaciones en Deep Learning · GR 608145"
    )

    artifact = load_artifact()
    if artifact is None:
        st.error(
            "No se encontró el modelo. Ejecuta primero `01_autoencoder_valvulas.py` "
            "para generar `salidas/modelo_autoencoder.joblib`."
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Inspeccionar válvula", "Lote completo", "Cómo funciona"])

    with tab1:
        st.subheader("Ingresar mediciones de una válvula")
        cols = st.columns(3)
        defaults = {
            "Diametro_mm": 25.0,
            "Peso_g": 150.0,
            "Presion_bar": 40.0,
            "Temperatura_C": 850.0,
            "Dureza_HRC": 45.0,
            "Tiempo_ciclo_s": 12.0,
        }
        values = {}
        for i, sensor in enumerate(artifact["sensors"]):
            with cols[i % 3]:
                values[sensor] = st.number_input(sensor, value=float(defaults[sensor]), format="%.3f")

        if st.button("Clasificar", type="primary"):
            res = predict_one(artifact, values)
            if res["anomalia"]:
                st.error(
                    f"ANOMALÍA detectada · error={res['error']:.4f} · z={res['z']:+.2f} "
                    f"(umbral={artifact['umbral']:.4f})"
                )
            else:
                st.success(
                    f"Normal · error={res['error']:.4f} · z={res['z']:+.2f} "
                    f"(umbral={artifact['umbral']:.4f})"
                )
            st.write("Contribución al error por sensor (%)")
            st.bar_chart(pd.Series(res["contrib"]))

    with tab2:
        st.subheader("Clasificar el lote a inspeccionar")
        lote_path = DATA / "CasoEstudioValvulas_LoteAInspeccionar.csv"
        resultados_path = OUT / "resultados_lote_autoencoder.csv"
        if resultados_path.exists():
            df = pd.read_csv(resultados_path)
            n_anom = (df["Clasificacion"] == "ANOMALIA").sum()
            st.metric("Válvulas anómalas", f"{n_anom} / {len(df)}")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Descargar resultados CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="resultados_lote_autoencoder.csv",
                mime="text/csv",
            )
            if (OUT / "fig_ranking_lote.png").exists():
                st.image(str(OUT / "fig_ranking_lote.png"), use_container_width=True)
        elif lote_path.exists():
            lote = pd.read_csv(lote_path)
            X = artifact["scaler"].transform(lote[artifact["sensors"]].to_numpy(float))
            recon = artifact["model"].predict(X)
            err = np.mean((X - recon) ** 2, axis=1)
            out = lote.copy()
            out["Error_reconstruccion"] = err
            out["Clasificacion"] = np.where(err > artifact["umbral"], "ANOMALIA", "normal")
            st.dataframe(out.sort_values("Error_reconstruccion", ascending=False))
        else:
            st.warning("No hay archivo de lote disponible.")

        uploaded = st.file_uploader("O carga un CSV con las columnas de sensores", type=["csv"])
        if uploaded is not None:
            up = pd.read_csv(uploaded)
            missing = [c for c in artifact["sensors"] if c not in up.columns]
            if missing:
                st.error(f"Faltan columnas: {missing}")
            else:
                X = artifact["scaler"].transform(up[artifact["sensors"]].to_numpy(float))
                recon = artifact["model"].predict(X)
                err = np.mean((X - recon) ** 2, axis=1)
                up = up.copy()
                up["Error_reconstruccion"] = err
                up["Clasificacion"] = np.where(err > artifact["umbral"], "ANOMALIA", "normal")
                st.dataframe(up.sort_values("Error_reconstruccion", ascending=False))

    with tab3:
        st.markdown(
            """
### Idea del modelo
1. Se entrena un **autoencoder** únicamente con válvulas **sin defecto**.
2. La red aprende a **reconstruir** el patrón correlacionado de los 6 sensores.
3. En inspección se mide el **error de reconstrucción (MSE)**.
4. Si el error supera el umbral `media + 3·std` (calculado en validación),
   la válvula se marca como **anómala**.

### ¿Por qué Deep Learning y no solo rangos por sensor?
Un filtro Excel `media ± 3σ` por columna no detecta defectos **multivariados**:
varios sensores se desplazan juntos sin salirse individualmente de rango.
El autoencoder sí captura esas correlaciones.
            """
        )
        c1, c2 = st.columns(2)
        if (OUT / "fig_error_reconstruccion.png").exists():
            c1.image(str(OUT / "fig_error_reconstruccion.png"), use_container_width=True)
        if (OUT / "fig_metricas_comparacion.png").exists():
            c2.image(str(OUT / "fig_metricas_comparacion.png"), use_container_width=True)


if __name__ == "__main__":
    main()
