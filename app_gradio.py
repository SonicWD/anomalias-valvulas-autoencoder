"""
Interfaz Gradio con enlace público (share=True) para demostrar el despliegue en la nube.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "salidas"
DATA = BASE / "ACTIVIDAD_2"
artifact = joblib.load(OUT / "modelo_autoencoder.joblib")
SENSORS = artifact["sensors"]


def clasificar(diametro, peso, presion, temperatura, dureza, tiempo):
    values = [diametro, peso, presion, temperatura, dureza, tiempo]
    x = np.array([values], dtype=float)
    xs = artifact["scaler"].transform(x)
    recon = artifact["model"].predict(xs)
    err = float(np.mean((xs - recon) ** 2))
    z = (err - artifact["err_val_mean"]) / artifact["err_val_std"]
    anom = err > artifact["umbral"]
    contrib = ((xs - recon) ** 2).ravel()
    contrib_pct = 100 * contrib / contrib.sum()
    estado = "ANOMALIA" if anom else "normal"
    detalle = (
        f"Clasificacion: {estado}\n"
        f"Error MSE: {err:.5f}\n"
        f"Score z: {z:+.2f}\n"
        f"Umbral: {artifact['umbral']:.5f}\n\n"
        "Contribucion por sensor (%):\n"
        + "\n".join(f"  - {s}: {contrib_pct[i]:.1f}%" for i, s in enumerate(SENSORS))
    )
    return estado, detalle


def lote_resumen():
    path = OUT / "resultados_lote_autoencoder.csv"
    if not path.exists():
        return "Ejecuta primero 01_autoencoder_valvulas.py"
    df = pd.read_csv(path)
    anom = df[df["Clasificacion"] == "ANOMALIA"][
        ["ID_Valvula", "Error_reconstruccion", "Score_z", "Clasificacion"]
    ]
    return anom.to_string(index=False)


with gr.Blocks(title="Anomalias valvulas - Autoencoder") as demo:
    gr.Markdown(
        "# Deteccion de anomalias en valvulas industriales\n"
        "Autoencoder (Deep Learning) · R1-A2-S4 · GR 608145"
    )
    with gr.Tab("Inspeccionar valvula"):
        with gr.Row():
            d = gr.Number(label="Diametro_mm", value=25.0)
            p = gr.Number(label="Peso_g", value=150.0)
            pr = gr.Number(label="Presion_bar", value=40.0)
        with gr.Row():
            t = gr.Number(label="Temperatura_C", value=850.0)
            du = gr.Number(label="Dureza_HRC", value=45.0)
            ti = gr.Number(label="Tiempo_ciclo_s", value=12.0)
        btn = gr.Button("Clasificar", variant="primary")
        out_estado = gr.Textbox(label="Estado")
        out_det = gr.Textbox(label="Detalle", lines=12)
        btn.click(clasificar, [d, p, pr, t, du, ti], [out_estado, out_det])

    with gr.Tab("Lote anomalas"):
        gr.Textbox(value=lote_resumen(), label="Valvulas anomalas del lote", lines=20)

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)
