"""
Script 1 — Autoencoder para detección de anomalías en válvulas industriales
===========================================================================
Entrena un autoencoder SOLO con válvulas normales (hoja Entrenamiento),
calcula el error de reconstrucción sobre el Lote a Inspeccionar y clasifica
cada válvula como normal o anómala según un umbral (media + 3·std del error
en validación).

Corresponde al plan de trabajo Python, paso 1, del caso de estudio.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "ACTIVIDAD_2"
OUT = BASE / "salidas"
OUT.mkdir(exist_ok=True)

SENSORS = [
    "Diametro_mm",
    "Peso_g",
    "Presion_bar",
    "Temperatura_C",
    "Dureza_HRC",
    "Tiempo_ciclo_s",
]


def main() -> None:
    train_df = pd.read_csv(DATA / "CasoEstudioValvulas_Entrenamiento.csv")
    lote_df = pd.read_csv(DATA / "CasoEstudioValvulas_LoteAInspeccionar.csv")

    X_all = train_df[SENSORS].to_numpy(dtype=float)
    X_lote = lote_df[SENSORS].to_numpy(dtype=float)
    ids_lote = lote_df["ID_Valvula"].astype(str).to_numpy()

    # Escalado: evita que Temperatura (~850) domine frente a Tiempo (~12)
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)
    X_lote_s = scaler.transform(X_lote)

    X_train, X_val = train_test_split(X_all_s, test_size=0.2, random_state=42)

    # Autoencoder: encoder -> latente -> decoder (mismo patrón del material S1-S4)
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(8, 3, 8),
        activation="relu",
        solver="adam",
        max_iter=3000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=40,
    )

    print("Entrenando autoencoder solo con válvulas normales...")
    autoencoder.fit(X_train, X_train)
    print(f"Entrenamiento terminado. Épocas efectivas: {autoencoder.n_iter_}\n")

    def error_por_valvula(X: np.ndarray) -> np.ndarray:
        recon = autoencoder.predict(X)
        return np.mean((X - recon) ** 2, axis=1)

    err_train = error_por_valvula(X_train)
    err_val = error_por_valvula(X_val)
    err_lote = error_por_valvula(X_lote_s)

    umbral = float(err_val.mean() + 3 * err_val.std())
    es_anomalia = err_lote > umbral
    score = (err_lote - err_val.mean()) / err_val.std()

    print(f"Error medio train: {err_train.mean():.5f}")
    print(f"Error medio val:   {err_val.mean():.5f}")
    print(f"Umbral (media+3std val): {umbral:.5f}")
    print(f"Anómalas en lote:  {es_anomalia.sum()} / {len(lote_df)}")

    resultados = lote_df.copy()
    resultados["Error_reconstruccion"] = err_lote
    resultados["Score_z"] = score
    resultados["Clasificacion"] = np.where(es_anomalia, "ANOMALIA", "normal")
    resultados = resultados.sort_values("Error_reconstruccion", ascending=False)
    resultados.to_csv(OUT / "resultados_lote_autoencoder.csv", index=False)

    anomalias = resultados[resultados["Clasificacion"] == "ANOMALIA"]
    anomalias.to_csv(OUT / "lista_valvulas_anomalas.csv", index=False)
    print("\nVálvulas anómalas (ordenadas por error):")
    for _, row in anomalias.iterrows():
        print(
            f"  {row['ID_Valvula']}: error={row['Error_reconstruccion']:.4f} "
            f"(z={row['Score_z']:+.2f})"
        )

    # Artefactos para la app en la nube
    joblib.dump(
        {
            "model": autoencoder,
            "scaler": scaler,
            "umbral": umbral,
            "err_val_mean": float(err_val.mean()),
            "err_val_std": float(err_val.std()),
            "sensors": SENSORS,
        },
        OUT / "modelo_autoencoder.joblib",
    )

    meta = {
        "umbral": umbral,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_lote": int(len(lote_df)),
        "n_anomalas": int(es_anomalia.sum()),
        "ids_anomalas": anomalias["ID_Valvula"].tolist(),
        "error_train_mean": float(err_train.mean()),
        "error_val_mean": float(err_val.mean()),
    }
    (OUT / "resumen_modelo.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Gráfico: distribución del error (val vs lote)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(err_val, bins=25, alpha=0.75, label="Validación (normales)", color="#0F6E56")
    ax.hist(err_lote, bins=25, alpha=0.75, label="Lote a inspeccionar", color="#378ADD")
    ax.axvline(umbral, color="black", linestyle="--", label=f"Umbral = {umbral:.3f}")
    ax.set_xlabel("Error de reconstrucción (MSE)")
    ax.set_ylabel("Cantidad de válvulas")
    ax.set_title("Error de reconstrucción: validación vs lote")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_reconstruccion.png", dpi=150)
    plt.close(fig)

    # Ranking visual del lote
    fig, ax = plt.subplots(figsize=(11, 5))
    orden = np.argsort(-err_lote)
    colores = ["#A32D2D" if es_anomalia[i] else "#5F5E5A" for i in orden]
    ax.bar(range(len(orden)), err_lote[orden], color=colores, width=1.0)
    ax.axhline(umbral, color="black", linestyle="--", label="Umbral")
    ax.set_xlabel("Válvulas del lote (ordenadas por error)")
    ax.set_ylabel("Error de reconstrucción")
    ax.set_title("Ranking de sospecha en el lote a inspeccionar")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_ranking_lote.png", dpi=150)
    plt.close(fig)

    # Curva de pérdida
    if hasattr(autoencoder, "loss_curve_") and autoencoder.loss_curve_:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(autoencoder.loss_curve_, color="#0F6E56")
        ax.set_title("Curva de pérdida del autoencoder")
        ax.set_xlabel("Época")
        ax.set_ylabel("Pérdida (MSE)")
        fig.tight_layout()
        fig.savefig(OUT / "fig_curva_perdida.png", dpi=150)
        plt.close(fig)

    print(f"\nSalidas guardadas en: {OUT}")


if __name__ == "__main__":
    main()
