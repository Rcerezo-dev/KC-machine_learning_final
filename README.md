# Práctica de Machine Learning – Predicción de precios en Airbnb

## Autor
Rubén Cerezo

---

## Objetivo

Predecir el precio de un alojamiento de Airbnb en Madrid a partir de datos reales obtenidos mediante scraping. Es un problema de **regresión supervisada**.

El foco de la práctica está en aplicar correctamente la metodología ML (sin data leakage, con evaluación honesta) más que en maximizar métricas.

---

## Estructura del repositorio

```
machine_learning_final/
│
├── data/
│   ├── raw/               ← CSV original de Airbnb
│   ├── processed/         ← datos limpios tras el EDA
│   └── splits/            ← X_train, X_val, X_test, y_train, y_val, y_test
│
├── notebooks/
│   ├── 01_eda.ipynb               ← análisis exploratorio
│   ├── 02_data_preparation.ipynb  ← limpieza, splits, preprocesado
│   ├── 03_modeling.ipynb          ← entrenamiento y evaluación
│   └── 04_conclusions.ipynb       ← comparativa final y conclusiones
│
├── src/
│   └── utils.py           ← funciones reutilizables (métricas, gráficos, I/O)
│
├── models/                ← modelos serializados (.joblib)
│
├── reports/
│   ├── figures/           ← gráficos exportados
│   └── profiling/         ← informes HTML de ydata-profiling
│
├── requirements.txt
└── README.md
```

Los notebooks están pensados para ejecutarse **en orden** (01 → 02 → 03 → 04). Cada uno carga los outputs del anterior desde disco.

---

## Metodología

### 1. Análisis exploratorio (`01_eda.ipynb`)
- Inspección inicial: tipos, nulos, distribuciones
- Detección de outliers por IQR
- Análisis de correlaciones con el precio
- Informe automático con ydata-profiling

### 2. Preparación de datos (`02_data_preparation.ipynb`)
- Filtrado a alojamientos de Madrid y limpieza de columnas
- **Split train / validation / test (70 / 15 / 15)** — antes de cualquier transformación
- Imputación, one-hot encoding y escalado ajustados **solo sobre train**, aplicados a val y test
- Transformación logarítmica del precio (`log1p`) para reducir el sesgo de la distribución

### 3. Modelado (`03_modeling.ipynb`)
Se entrenan y comparan cinco modelos con GridSearchCV:

| Modelo | Notas |
|--------|-------|
| Lasso | regularización L1, selección implícita de features |
| Ridge | regularización L2 |
| ElasticNet | combinación L1 + L2 |
| Random Forest | ensemble de árboles, grid amplio para controlar overfitting |
| Gradient Boosting | boosting secuencial, suele superar a RF en tabular data |

Métricas reportadas: **RMSE, MAE y R²** sobre train, validation y test.

### 4. Conclusiones (`04_conclusions.ipynb`)
- Tabla comparativa de métricas de todos los modelos
- Importancia de features del mejor modelo
- Análisis de residuos
- Reflexión cualitativa sobre los resultados

---

## Resultados

El mejor modelo es **Gradient Boosting**, que obtiene el RMSE y R² más competitivos en el conjunto de test manteniendo un gap train/test razonable.

---

## Tecnologías

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- ydata-profiling
- joblib

Versiones exactas en `requirements.txt`.

---

## Notas

- Los archivos CSV de datos y los HTML de profiling están excluidos del repositorio (`.gitignore`) por su tamaño.
- `random_state=3` fijado en todos los puntos de aleatoriedad para reproducibilidad total.
