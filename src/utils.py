"""
utils.py — Funciones reutilizables para el proyecto de predicción de precios Airbnb.

Uso desde los notebooks:
    import sys
    sys.path.append(str(Path('..').resolve()))
    from src.utils import metricas, plot_resultados, plot_importancia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def metricas(nombre, y_true_train, y_pred_train, y_true_val, y_pred_val):
    """
    Imprime RMSE, MAE y R² para train y validación.

    Parámetros
    ----------
    nombre        : etiqueta del modelo (ej. 'Lasso')
    y_true_train  : precios reales del conjunto de entrenamiento
    y_pred_train  : predicciones sobre entrenamiento
    y_true_val    : precios reales del conjunto de validación
    y_pred_val    : predicciones sobre validación
    """
    print(f'─── {nombre} ───')
    for label, y_true, y_pred in [
        ('Train', y_true_train, y_pred_train),
        ('Val  ', y_true_val,   y_pred_val),
    ]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        print(f'  {label} → RMSE: {rmse:6.2f} €  |  MAE: {mae:6.2f} €  |  R²: {r2:.4f}')
    print()


def metricas_test(nombre, y_true, y_pred):
    """
    Devuelve un diccionario con RMSE, MAE y R² sobre un único conjunto (test).

    Útil para construir la tabla comparativa en 04_conclusions.
    """
    return {
        'Modelo':   nombre,
        'RMSE (€)': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'MAE (€)':  round(mean_absolute_error(y_true, y_pred), 2),
        'R²':       round(r2_score(y_true, y_pred), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def plot_resultados(nombre, y_true, y_pred, figures_dir=None):
    """
    Dos gráficas lado a lado:
      - Precio real vs precio predicho
      - Distribución de residuos

    Parámetros
    ----------
    nombre      : título del modelo
    y_true      : serie con precios reales
    y_pred      : array con predicciones
    figures_dir : Path donde guardar la imagen (None = no guardar)
    """
    residuos = np.array(y_true) - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Real vs predicho
    axes[0].scatter(y_true, y_pred, alpha=0.4)
    lims = [min(np.min(y_true), y_pred.min()), max(np.max(y_true), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', linewidth=1.5)
    axes[0].set_xlabel('Precio real (€)')
    axes[0].set_ylabel('Precio predicho (€)')
    axes[0].set_title(f'{nombre}: Real vs Predicho')
    axes[0].grid(True)

    # Residuos
    axes[1].hist(residuos, bins=40, alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Residuo (real − predicho)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title(f'{nombre}: Distribución de residuos')
    axes[1].grid(True)

    plt.tight_layout()
    if figures_dir is not None:
        fname = nombre.lower().replace(' ', '_') + '_resultados.png'
        plt.savefig(Path(figures_dir) / fname, dpi=120)
    plt.show()


def plot_importancia(nombre, feature_names, importancias, top_n=20, figures_dir=None):
    """
    Gráfico horizontal de barras con las top_n variables más importantes.

    Válido tanto para coeficientes (Lasso/Ridge) como para feature_importances_ (RF).
    """
    series = pd.Series(importancias, index=feature_names)
    top = series.abs().nlargest(top_n).index
    datos = series[top]

    plt.figure(figsize=(10, max(4, top_n * 0.35)))
    plt.barh(datos.index, datos.values)
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Importancia / Coeficiente')
    plt.title(f'{nombre}: top {top_n} variables')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if figures_dir is not None:
        fname = nombre.lower().replace(' ', '_') + '_importancia.png'
        plt.savefig(Path(figures_dir) / fname, dpi=120)
    plt.show()


def plot_comparativa(resultados_df, figures_dir=None):
    """
    Barplot comparativo de RMSE, MAE y R² para todos los modelos.

    Parámetros
    ----------
    resultados_df : DataFrame con columnas 'RMSE (€)', 'MAE (€)', 'R²' e índice = nombre del modelo
    figures_dir   : Path donde guardar la imagen (None = no guardar)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    resultados_df[['RMSE (€)', 'MAE (€)']].plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_title('Error en test por modelo (€)')
    axes[0].set_ylabel('Error (€)')
    axes[0].grid(axis='y')

    resultados_df['R²'].plot(kind='bar', ax=axes[1], color='steelblue', rot=0)
    axes[1].set_title('R² en test por modelo')
    axes[1].set_ylabel('R²')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y')

    plt.tight_layout()
    if figures_dir is not None:
        plt.savefig(Path(figures_dir) / 'comparativa_modelos.png', dpi=120)
    plt.show()


def plot_curva_aprendizaje(nombre, modelo, X, y, figures_dir=None):
    """
    Curva de aprendizaje: RMSE en train y validación cruzada
    según el tamaño del conjunto de entrenamiento.

    Sirve para diagnosticar si el modelo necesita más datos (las dos curvas
    no convergen) o si el problema es sobreajuste (gran separación entre
    train y validación).

    Nota: las métricas están en la escala en que se entrena el modelo
    (si y está en log1p, el RMSE también lo estará).
    """
    train_sizes, train_scores, val_scores = learning_curve(
        modelo, X, y,
        cv=5,
        scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_rmse = np.sqrt(-train_scores).mean(axis=1)
    val_rmse   = np.sqrt(-val_scores).mean(axis=1)
    train_std  = np.sqrt(-train_scores).std(axis=1)
    val_std    = np.sqrt(-val_scores).std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_rmse, 'o-', label='Train')
    plt.fill_between(train_sizes, train_rmse - train_std, train_rmse + train_std, alpha=0.15)
    plt.plot(train_sizes, val_rmse, 'o-', label='Validación (CV)')
    plt.fill_between(train_sizes, val_rmse - val_std, val_rmse + val_std, alpha=0.15)
    plt.xlabel('Tamaño del conjunto de entrenamiento (filas)')
    plt.ylabel('RMSE')
    plt.title(f'Curva de aprendizaje — {nombre}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if figures_dir is not None:
        fname = nombre.lower().replace(' ', '_') + '_learning_curve.png'
        plt.savefig(Path(figures_dir) / fname, dpi=120)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────────────────────

def guardar_modelo(modelo, path):
    """Serializa un modelo entrenado en disco con joblib."""
    joblib.dump(modelo, path)
    print(f'Modelo guardado en {path}')


def cargar_modelo(path):
    """Carga un modelo previamente serializado con joblib."""
    modelo = joblib.load(path)
    print(f'Modelo cargado desde {path}')
    return modelo
