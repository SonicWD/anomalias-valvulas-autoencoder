# anomalias-valvulas-autoencoder

Detección de anomalías en válvulas industriales con un **Autoencoder** (Deep Learning).

**Curso:** Aplicaciones en Deep Learning · GR 608145 · R1-A2-S4  
**Stack:** Python + Streamlit + scikit-learn  
**Repo:** https://github.com/SonicWD/anomalias-valvulas-autoencoder

> **No uses Vercel.** Vercel es para frontends (Next.js/React). Esta app necesita un runtime Python continuo. Usa **Streamlit Community Cloud**.

## Despliegue en Streamlit Cloud

1. Entra a https://share.streamlit.io e inicia sesión con GitHub.
2. **New app** → repo `SonicWD/anomalias-valvulas-autoencoder`.
3. **Main file path:** `app_streamlit.py`
4. **Python version:** `3.11`
5. Deploy.

URL típica: `https://anomalias-valvulas-autoencoder.streamlit.app`

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Resultados del modelo

- Anómalas Autoencoder: **22 / 100**
- Anómalas filtro Excel (media ± 3σ): **18 / 100**
- Acuerdo entre métodos: **90%**
- Solo Autoencoder (posible defecto multivariado): L086, L087, L091, L092, L093, L099, L100

## Estructura

| Archivo | Rol |
|---|---|
| `app_streamlit.py` | Interfaz en la nube |
| `salidas/modelo_autoencoder.joblib` | Modelo entrenado |
| `ACTIVIDAD_2/*.csv` | Datos del caso de estudio |
| `01_autoencoder_valvulas.py` | Script 1: entrenamiento y clasificación |
| `02_metricas_comparacion.py` | Script 2: métricas y comparación Excel vs AE |

## Regenerar entregables (opcional)

```bash
pip install openpyxl python-docx
python 01_autoencoder_valvulas.py
python 02_metricas_comparacion.py
python 03_generar_excel.py
python 04_generar_documento.py
```
