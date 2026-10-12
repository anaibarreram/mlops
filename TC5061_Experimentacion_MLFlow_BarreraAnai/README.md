# Telco Customer Churn | Experimentación con MLflow

Actividad individual TC5061 · Estructura organizada inspirada en CCDS y el proyecto de ejemplo del profesor (`ITESM_MNA_2026_MLFlow_Projects.zip`).

Se utiliza **Regresión Logística**

## Requisitos

- Python **3.14** recomendado (use esa version). Las versiones están en `python_env.yaml`.
- Dataset original `WA_Fn-UseC_-Telco-Customer-Churn.csv` utilizado en la actividad previa. Guardarlo exactamente en `data/raw/`.
- VS Code, terminal zsh/bash/WSL, acceso al navegador.

## Estructura

```text
.
├── data/
│   ├── raw/                # CSV de la práctica anterior
│   ├── interim/
│   └── processed/          # Copias limpias opcionales
├── notebooks/              # Notebook previo: solo análisis exploratorio
├── churn/
│   ├── config.py
│   ├── dataset.py          # Carga/depuración de datos
│   ├── features.py         # ColumnTransformer sin fuga de datos
│   ├── plots.py            # Matriz de confusión
│   └── modeling/
│       ├── train.py        # CLI, entrenamiento y MLflow
│       └── predict.py
├── scripts/
│   ├── prepare_data.py
│   └── run_experiments.py  # Cinco corridas únicas
├── tests/
├── MLproject
├── python_env.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 1. Preparar entorno (ejecutar dentro de la carpeta raíz del proyecto)

En macOS / Linux / WSL Ubuntu, terminal zsh o bash:

```bash
python3.14 -m venv .venv      
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

El archivo `requirements.txt` instala también el paquete local `churn`. El dataset **no** está incluido: debes copiar el CSV del notebook anterior a `data/raw/`.

## 2. Levantar servidor MLflow

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Abre http://127.0.0.1:5000

## 3. Primera corrida (sin argumentos)

```bash
python -m churn.modeling.train
```

El programa mostrará métricas y `RUN_ID`. Por defecto usa C=1, max_iter=1000, class_weight=none, seed=42 y experimento `Telco_Churn_Comparacion`.

Para cambiar parámetros sin editar código:

```bash
python -m churn.modeling.train --c 0.1 --max-iter 1000 --class-weight balanced --seed 42
```

## 4. Ejecutar cinco configuraciones distintas

```bash
python scripts/run_experiments.py
```

En MLflow > experimento `Telco_Churn_Comparacion`: marca las cinco corridas y pulsa **Compare**. La métrica prioritaria es `f1_churn`; también revisar `recall_churn` y `roc_auc`. Compara solo corridas con idéntica partición (`seed=42`, `test_size=0.2`). No presupongas cuál gana.

## 5. MLflow Projects 

Con el servidor del paso 2 activo y el entorno virtual instalado:

```bash
mlflow run . --env-manager local --experiment-name Telco_Churn_Comparacion -P c=0.1 -P max_iter=1000 -P class_weight=balanced -P seed=42
```

**Alternativa:** `mlflow run . -e train --env-manager local --experiment-name Telco_Churn_Comparacion`.


El ejercicio evalúa un conjunto de prueba fijo; el rendimiento puede variar con otra partición. Cambiar `max_iter` no necesariamente afecta resultados si el optimizador ya convergió. `class_weight=balanced` puede mejorar recall y reducir precisión. La selección del mejor modelo debe justificarse mediante evidencia real del registro MLflow, no con métricas previstas.

Hay un PDF en docs para documentacion de la actividad