# ETML Workbench

A local machine learning workbench for exploring datasets, preparing data, training models, and making predictions—all organized into projects.

The browser UI is the recommended way to use the workbench.

## What it can do

- **Understand data:** optionally add a dataset explanation and request an OpenAI summary with column meanings and dependency hypotheses. Review suggestions during preprocessing.
- **Explore data:** automatic EDA reports, column statistics, data previews, and visualizations.
- **Prepare datasets:** remove selected columns, find/replace values, and record preprocessing history in reports; review preprocessing recommendations, preview changed rows, and create train/validation/test splits while preserving raw data.
- **Train models:** choose models and hyperparameters from supported libraries including scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, and TensorFlow.
- **Optimize models:** repeat supervised tabular training across split seeds or manually supplied hyperparameter configurations, with saved trial leaderboards and training times.
- **Control training:** cancel at a safe backend boundary, or end a tabular PyTorch/TensorFlow MLP after its current epoch.
- **Review results:** inspect metrics, model settings, and recorded loss curves; compare models using test metrics, with a labeled training-metrics fallback when no test data exists.
- **Make predictions:** reuse saved CSV inputs and export formats, or try individual examples.
- **Competition workflow:** build parameter-trial groups with editable forms, submit generated CSVs through your configured Kaggle CLI, and refresh public scores in Compare.
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

For Kaggle submissions, install `kaggle` with pip, configure your Kaggle credentials, and accept the competition rules on Kaggle. Generate the required CSV columns, then use **Submit this CSV to Kaggle**. Pending scores can be refreshed manually. Public scores are matched to the exact submission and shown separately from local test metrics.

## Optional AI dataset context

Set `OPENAI_API_KEY` in the server environment before launching the UI. The key
stays on the server. `ETML_OPENAI_MODEL` optionally changes the default
`gpt-5-mini` model. The workbench uses OpenAI Responses with structured outputs.

In Add data, enter an optional explanation and enable AI analysis. The workbench
sends that explanation, column names and aggregate statistics to OpenAI; sample
values are a separate opt-in (up to three values per column, 80 characters each).
Up to 100 columns are supported for AI analysis. Local EDA still works without
an API key or when an API call fails. Existing datasets can update context from
Overview & preview > Dataset explanation & AI analysis.

AI dependency hypotheses are saved with the dataset and shown in EDA warnings
and recipe review. Suggested removals start disabled; verify them before enabling.
They are not proven causal relationships. Group/time risks should inform your
split choice. Target columns remain protected. Reports include saved preprocessing
history. Model forms use editable v1 size/width heuristics after splitting;
these are starting points, not tuned or statistically optimal settings.
