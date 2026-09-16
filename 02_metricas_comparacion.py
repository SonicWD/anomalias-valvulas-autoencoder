"""
Script 2 — Métricas y comparación Excel (filtro univariado) vs Autoencoder
==========================================================================
Calcula al menos tres métricas apropiadas para el caso de válvulas
(sin etiquetas reales en el lote) y compara el filtro por rangos
(media ± 3·std por sensor) contra el autoencoder multivariable.

Métricas principales:
  1) Tasa de anomalías detectadas (operacional)
  2) Acuerdo / discordancias entre Excel y Autoencoder
  3) Separación del error de reconstrucción (ratio anomalía/normal)
  4) Diagnóstico por sensor (contribución al error)
  5) Estabilidad del umbral (sensibilidad)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


def filtro_excel(train_df: pd.DataFrame, lote_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce el plan Excel: marca válvulas fuera de media ± 3 std en algún sensor."""
    stats = train_df[SENSORS].agg(["mean", "std"])
    flags = pd.DataFrame(index=lote_df.index)
    for col in SENSORS:
        mu, sigma = stats.loc["mean", col], stats.loc["std", col]
        low, high = mu - 3 * sigma, mu + 3 * sigma
        flags[col] = (lote_df[col] < low) | (lote_df[col] > high)
    flags["fuera_rango_excel"] = flags[SENSORS].any(axis=1)
    flags["n_sensores_fuera"] = flags[SENSORS].sum(axis=1)
    return flags


def main() -> None:
    train_df = pd.read_csv(DATA / "CasoEstudioValvulas_Entrenamiento.csv")
    lote_df = pd.read_csv(DATA / "CasoEstudioValvulas_LoteAInspeccionar.csv")

    artifact = joblib.load(OUT / "modelo_autoencoder.joblib")
    model = artifact["model"]
    scaler = artifact["scaler"]
    umbral = artifact["umbral"]
    err_val_mean = artifact["err_val_mean"]
    err_val_std = artifact["err_val_std"]

    X_all = scaler.transform(train_df[SENSORS].to_numpy(dtype=float))
    X_lote = scaler.transform(lote_df[SENSORS].to_numpy(dtype=float))
    _, X_val = train_test_split(X_all, test_size=0.2, random_state=42)

    recon_lote = model.predict(X_lote)
    err_lote = np.mean((X_lote - recon_lote) ** 2, axis=1)
    err_sensor = (X_lote - recon_lote) ** 2
    pred_ae = err_lote > umbral

    recon_val = model.predict(X_val)
    err_val = np.mean((X_val - recon_val) ** 2, axis=1)

    flags = filtro_excel(train_df, lote_df)
    pred_excel = flags["fuera_rango_excel"].to_numpy()

    # ---------- Métrica 1: tasa de anomalías ----------
    tasa_ae = pred_ae.mean()
    tasa_excel = pred_excel.mean()
    print("=" * 70)
    print("MÉTRICA 1 — Tasa de anomalías detectadas (operacional)")
    print("=" * 70)
    print(f"Autoencoder: {pred_ae.sum()}/{len(lote_df)} = {100 * tasa_ae:.1f}%")
    print(f"Filtro Excel: {pred_excel.sum()}/{len(lote_df)} = {100 * tasa_excel:.1f}%")
    print(
        "Justificación: en control de calidad importa cuántas piezas se "
        "desvían a revisión. Una tasa extrema (casi 0% o casi 100%) suele "
        "indicar umbral mal calibrado."
    )

    # ---------- Métrica 2: acuerdo entre enfoques ----------
    ambos = pred_ae & pred_excel
    solo_ae = pred_ae & ~pred_excel
    solo_excel = ~pred_ae & pred_excel
    ninguno = ~pred_ae & ~pred_excel
    acuerdo = (pred_ae == pred_excel).mean()

    print("\n" + "=" * 70)
    print("MÉTRICA 2 — Acuerdo Excel vs Autoencoder")
    print("=" * 70)
    print(f"Acuerdo total:           {100 * acuerdo:.1f}%")
    print(f"Detectadas por ambos:    {ambos.sum()}")
    print(f"Solo autoencoder:        {solo_ae.sum()}")
    print(f"Solo filtro Excel:       {solo_excel.sum()}")
    print(f"Ninguno:                 {ninguno.sum()}")
    print(
        "Justificación: el filtro univariado no ve correlaciones. Las "
        "válvulas 'solo autoencoder' son candidatas a defectos multivariados "
        "(varios sensores se mueven juntos sin salirse individualmente de rango)."
    )

    ids = lote_df["ID_Valvula"].astype(str).to_numpy()
    print("\nSolo autoencoder (posible defecto correlacionado):")
    for i in np.where(solo_ae)[0]:
        print(f"  {ids[i]}  error={err_lote[i]:.4f}")
    print("Solo Excel:")
    for i in np.where(solo_excel)[0]:
        print(f"  {ids[i]}  sensores fuera={flags.loc[i, 'n_sensores_fuera']}")

    # ---------- Métrica 3: separación del error ----------
    err_anom = err_lote[pred_ae]
    err_norm = err_lote[~pred_ae]
    ratio = (err_anom.mean() / err_norm.mean()) if len(err_anom) and len(err_norm) else np.nan
    print("\n" + "=" * 70)
    print("MÉTRICA 3 — Separación del error de reconstrucción")
    print("=" * 70)
    print(f"Error medio normales (lote):   {err_norm.mean():.5f}")
    print(f"Error medio anómalas (lote):   {err_anom.mean():.5f}")
    print(f"Ratio anomalía/normal:         {ratio:.2f}x")
    print(f"Error medio validación:        {err_val.mean():.5f}")
    print(
        "Justificación: un buen detector debe separar claramente el error "
        "de las piezas sospechosas respecto al de las normales."
    )

    # ---------- Métrica 4: diagnóstico por sensor ----------
    print("\n" + "=" * 70)
    print("MÉTRICA 4 — Diagnóstico por sensor (contribución al error)")
    print("=" * 70)
    diag_rows = []
    for i in np.where(pred_ae)[0]:
        contrib = 100 * err_sensor[i] / err_sensor[i].sum()
        top = np.argsort(-contrib)[:3]
        diag = ", ".join(f"{SENSORS[j]} ({contrib[j]:.0f}%)" for j in top)
        print(f"  {ids[i]}: {diag}")
        diag_rows.append(
            {
                "ID_Valvula": ids[i],
                "Error": err_lote[i],
                "Top_sensores": diag,
                "Metodo": "ambos" if pred_excel[i] else "solo_AE",
            }
        )

    # ---------- Métrica 5: sensibilidad del umbral ----------
    print("\n" + "=" * 70)
    print("MÉTRICA 5 — Sensibilidad del umbral (estabilidad)")
    print("=" * 70)
    for k in [2, 2.5, 3, 3.5, 4]:
        u = err_val_mean + k * err_val_std
        n = int((err_lote > u).sum())
        print(f"  media + {k}*std = {u:.5f}  ->  {n} anomalas")

    # Guardar comparación completa
    comp = lote_df.copy()
    comp["Error_AE"] = err_lote
    comp["Anomalia_AE"] = np.where(pred_ae, "SI", "NO")
    comp["Anomalia_Excel"] = np.where(pred_excel, "SI", "NO")
    comp["n_sensores_fuera_Excel"] = flags["n_sensores_fuera"]
    for col in SENSORS:
        comp[f"fuera_{col}"] = flags[col].map({True: "SI", False: "NO"})
    contrib_pct = 100 * err_sensor / err_sensor.sum(axis=1, keepdims=True)
    for j, col in enumerate(SENSORS):
        comp[f"contrib_{col}_pct"] = contrib_pct[:, j]
    comp = comp.sort_values("Error_AE", ascending=False)
    comp.to_csv(OUT / "comparacion_excel_vs_autoencoder.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT / "diagnostico_anomalas.csv", index=False)

    metricas = {
        "tasa_anomalias_autoencoder": float(tasa_ae),
        "tasa_anomalias_excel": float(tasa_excel),
        "acuerdo_porcentaje": float(100 * acuerdo),
        "detectadas_ambos": int(ambos.sum()),
        "solo_autoencoder": int(solo_ae.sum()),
        "solo_excel": int(solo_excel.sum()),
        "ratio_error_anomalia_normal": float(ratio) if ratio == ratio else None,
        "ids_solo_autoencoder": ids[solo_ae].tolist(),
        "ids_solo_excel": ids[solo_excel].tolist(),
        "ids_ambos": ids[ambos].tolist(),
        "ids_anomalas_ae": ids[pred_ae].tolist(),
    }
    (OUT / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Gráficos de evidencia
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: histograma lote marcado
    axes[0].hist(err_lote[~pred_ae], bins=20, alpha=0.8, label="Normal AE", color="#0F6E56")
    axes[0].hist(err_lote[pred_ae], bins=20, alpha=0.8, label="Anomalía AE", color="#A32D2D")
    axes[0].axvline(umbral, color="black", linestyle="--", label="Umbral")
    axes[0].set_title("Separación del error en el lote")
    axes[0].set_xlabel("Error MSE")
    axes[0].legend()

    # Panel 2: matriz de acuerdo
    mat = np.array([[ninguno.sum(), solo_excel.sum()], [solo_ae.sum(), ambos.sum()]])
    im = axes[1].imshow(mat, cmap="Blues")
    axes[1].set_xticks([0, 1], ["Excel NO", "Excel SI"])
    axes[1].set_yticks([0, 1], ["AE NO", "AE SI"])
    axes[1].set_title(f"Acuerdo métodos ({100 * acuerdo:.0f}%)")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, mat[i, j], ha="center", va="center", fontsize=14)

    # Panel 3: contribución media de sensores en anómalas AE
    if pred_ae.any():
        mean_contrib = contrib_pct[pred_ae].mean(axis=0)
        axes[2].barh(SENSORS, mean_contrib, color="#A32D2D")
        axes[2].set_xlabel("% contribución media al error")
        axes[2].set_title("Sensores que explican las anomalías AE")
    else:
        axes[2].text(0.5, 0.5, "Sin anomalías AE", ha="center")
        axes[2].set_axis_off()

    fig.tight_layout()
    fig.savefig(OUT / "fig_metricas_comparacion.png", dpi=150)
    plt.close(fig)

    # Dispersión temperatura vs dureza (evidencia de correlación)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        train_df["Temperatura_C"],
        train_df["Dureza_HRC"],
        s=12,
        alpha=0.45,
        color="#0F6E56",
        label="Entrenamiento",
    )
    ax.scatter(
        lote_df.loc[~pred_ae, "Temperatura_C"],
        lote_df.loc[~pred_ae, "Dureza_HRC"],
        s=35,
        marker="s",
        color="#378ADD",
        label="Lote normal AE",
    )
    ax.scatter(
        lote_df.loc[pred_ae, "Temperatura_C"],
        lote_df.loc[pred_ae, "Dureza_HRC"],
        s=55,
        marker="X",
        color="#A32D2D",
        label="Lote anomalía AE",
    )
    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("Dureza (HRC)")
    ax.set_title("Temperatura vs Dureza")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_dispersion_temp_dureza.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        train_df["Diametro_mm"],
        train_df["Peso_g"],
        s=12,
        alpha=0.45,
        color="#0F6E56",
        label="Entrenamiento",
    )
    ax.scatter(
        lote_df.loc[~pred_ae, "Diametro_mm"],
        lote_df.loc[~pred_ae, "Peso_g"],
        s=35,
        marker="s",
        color="#378ADD",
        label="Lote normal AE",
    )
    ax.scatter(
        lote_df.loc[pred_ae, "Diametro_mm"],
        lote_df.loc[pred_ae, "Peso_g"],
        s=55,
        marker="X",
        color="#A32D2D",
        label="Lote anomalía AE",
    )
    ax.set_xlabel("Diámetro (mm)")
    ax.set_ylabel("Peso (g)")
    ax.set_title("Diámetro vs Peso")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_dispersion_diametro_peso.png", dpi=150)
    plt.close(fig)

    print(f"\nSalidas guardadas en: {OUT}")


if __name__ == "__main__":
    main()
