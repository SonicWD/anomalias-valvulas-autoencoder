# Detección de anomalías en válvulas · Autoencoder (R1-A2-S4)

Aplicación en **Python + Streamlit** para clasificar válvulas industriales con un autoencoder de Deep Learning.

> **No uses Vercel** para este proyecto: Vercel está pensado para frontends (Next.js/React) y funciones serverless cortas. Este modelo usa `scikit-learn` + Streamlit y necesita un runtime Python continuo → **Streamlit Community Cloud** (recomendado) o **Hugging Face Spaces**.

## Despliegue en Streamlit Cloud (recomendado)

1. Crea un repositorio en GitHub con el contenido de esta carpeta.
2. Entra a [https://share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
3. **New app** → selecciona el repo.
4. **Main file path:** `app_streamlit.py`
5. **Python version:** 3.11
6. Deploy.

URL pública típica: `https://<tu-app>.streamlit.app`

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Entrenar / regenerar entregables (opcional)

```bash
pip install openpyxl python-docx
python 01_autoencoder_valvulas.py
python 02_metricas_comparacion.py
python 03_generar_excel.py
python 04_generar_documento.py
```

## Estructura

| Archivo | Rol |
|---|---|
| `app_streamlit.py` | Interfaz en la nube |
| `salidas/modelo_autoencoder.joblib` | Modelo entrenado |
| `ACTIVIDAD_2/*.csv` | Datos del caso |
| `01_*.py` / `02_*.py` | Scripts de la entrega académica |

## Resultados del modelo

- 22 anomalías / 100 en el lote
- Umbral ≈ 0.474 (media + 3·std en validación)
- 7 válvulas solo detectadas por el autoencoder (no por filtro Excel)
