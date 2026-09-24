# ETML Workbench

A local machine learning workbench for exploring datasets, preparing data, training models, and making predictions—all organized into projects.

The browser UI is the recommended way to use the workbench.

## What it can do

- **Explore data:** automatic EDA reports, column statistics, data previews, and visualizations.
- **Prepare datasets:** review preprocessing recommendations, preview changed rows, and create train/validation/test splits while preserving raw data.
- **Train models:** choose models and hyperparameters from supported libraries including scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, and TensorFlow.
- **Review results:** inspect metrics, model settings, and recorded loss curves; compare models using test metrics, with a labeled training-metrics fallback when no test data exists.
- **Make predictions:** try individual examples or upload a CSV, then select which rows and columns to download.
- **Visualize classifications:** compare true and predicted classes using one or two feature columns and include selected graphs in reports.
- **Export your work:** download model bundles and self-contained HTML summaries containing EDA, model rankings, results, and selected graphs.

The model catalog also includes clustering, dimensionality reduction, image and text models, time-series forecasting, and recommendation models. Available training, prediction, and export features depend on the model and input type. CSV target prediction and column-based classification plots currently support supervised tabular models.

## Installation

You need **Python 3.10 or newer** and Git to clone the repository. A virtual environment is recommended.

```bash
git clone https://github.com/EricTLucas/ETML-Workbench.git
cd ETML-Workbench
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install the workbench with its UI:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[ui]"
```

Missing optional model packages are installed automatically when needed by the UI. These downloads require internet access and may take time for larger libraries. To import Hugging Face datasets, also install:

```bash
python -m pip install -e ".[huggingface]"
```

## Start using the UI

```bash
workbench ui
```

Keep the terminal running while using the browser interface.

1. **Create or open a project.** Upload data or choose a dataset from the library.
2. **Explore and prepare.** Review the EDA, select a target, choose preprocessing steps, and create a split.
3. **Train a model.** Open Models, choose a model, adjust its settings, and start training.
4. **Review and predict.** Inspect Results, compare saved models, or try individual and CSV predictions.
5. **Visualize and export.** Use Visualize to generate classification plots, mark graphs for inclusion, and create an HTML analysis summary. Export model bundles from Model details.

Projects are saved locally in the `projects` directory by default. To use another location:

```bash
workbench ui --projects-dir "path/to/projects"
```

Raw data, processed data, splits, models, and exports stay organized within each project. Dataset downloads and optional package/model downloads may require internet access; exported HTML summaries can be viewed offline.

## CLI

A CLI is also available for terminal workflows:

```bash
workbench          # Interactive project menu
workbench --help   # Available commands
```

For the guided end-to-end workflow, start with `workbench ui`.
