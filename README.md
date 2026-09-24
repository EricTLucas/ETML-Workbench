# ETML Workbench

ETML Workbench is a local Python workbench for exploring data, reviewing preprocessing, preparing prediction tasks, comparing models, making predictions, and exporting reusable models. Its project model library also supports image classification, text models, univariate forecasting, recommendations, and tabular unsupervised exploration.

The current workflow is:

```text
Import and preserve raw data
    → inspect it with EDA
    → optionally propose and review a preprocessing recipe
    → define the target and split the data
    → fit preprocessing on training features only
    → train candidates and compare validation results
    → evaluate the chosen model on labeled test data
    → predict, export, or reproduce the experiment
```

The CLI is `workbench`. An optional local browser interface is available with `workbench ui`. Python packages remain independently importable for integrations.

This README describes the implemented 0.18.0 update, including the browser Models section. It assumes those files have been merged into the repository, including its `pyproject.toml`.

Saved-model inspection now includes **Create a new model**, retrying failed configurations, and additional-epoch training for resumable PyTorch/TensorFlow MLPs. Open `workbench PROJECT --predict` and choose a saved entry; failed entries show their error, settings, and retry action. 

**Guided prediction:** splitting now opens model selection automatically; completed training opens model results and prediction tools. Use `workbench PROJECT --predict` to revisit saved models, inspect rows and true values, view training details, evaluate a tabular test split, or export models and their associated data.

**New model workflow:** run `workbench PROJECT --models` to choose, name, configure and train a model.

## Start with a project (new in 0.8)

After installation, enter:

```bash
workbench
```

Choose **New project** or **Existing project**. New projects ask for a name, then offer a local upload or dataset library. Existing projects are listed for selection. Use a portable name such as `customer-churn` (letters, digits, hyphens, underscores; no spaces). Enter `q` at a numbered menu to cancel.

To create a project and import directly:

```bash
workbench flowers --sklearn iris
workbench titanic -upload "uploads/train.csv"
workbench titanic -upload "uploads/train.csv" --test "uploads/test.csv"
```

`-upload` and `--upload` are aliases. Add `--validation FILE` for a supplied validation set. Every import preserves a new dataset snapshot within the named project; previous imports are kept. The most recently imported snapshot becomes the current dataset.

By default, import shows the first five source/training rows, generates summary EDA, saves its HTML report and bounded sample, and opens the report in your default browser. It prints a short status and one next-step command rather than profile JSON or recipe internals. For presplit inputs, this EDA uses training data only.

```bash
workbench flowers --sklearn iris --no-open
workbench large-data --upload data.parquet --no-eda
workbench large-data --eda --no-open
```

`--no-open` still generates the report; `--no-eda` skips it entirely. `--eda` analyzes the current saved dataset without importing it again. If EDA fails, the import remains available and a retry command is printed (exit code 3). A browser-opening failure leaves a usable report path.

Find datasets and return to a project:

```bash
workbench sklearn
workbench huggingface
workbench openml
workbench library
workbench projects
workbench flowers
```

Provider commands show instructions and examples. Import a chosen provider dataset with `workbench PROJECT --sklearn NAME`, `--huggingface OWNER/DATASET`, or `--openml DATA_ID`. Direct file URLs use `--url URL`. Hugging Face also supports `--config`, `--split`, `--revision`, and `--columns`. Import limits default to 200,000 provider rows and 512 MiB; use `--max-rows` and `--max-memory-mb` to adjust applicable limits. EDA has `--batch-size` and `--sample-size` controls.

Project storage is isolated:

```text
projects/
└── flowers/
    ├── project.json
    └── datasets/
        └── data-GENERATED_ID/
            ├── raw/
            ├── profiles/overview-GENERATED_ID/report.html
            ├── recipes/
            ├── processed/
            └── tasks/   (created when you define a task)
```

The default root is `projects` relative to the current directory. Set `ETML_PROJECTS_DIR` or consistently pass `--projects-dir PATH` to use another location. `workbench --no-open` starts the wizard with automatic browser opening disabled. `workbench PROJECT --help` lists project options.

Reopening a project shows its dataset, preview, report location, and saved raw/processed/split artifacts. In an interactive terminal it offers the next unfinished preparation stage. Existing standalone `datasets/` workspaces are unchanged and are not automatically moved into projects. The browser UI uses these same project folders; launch it with `workbench ui`.

## Guided preprocessing and splitting (new in 0.9)

The project flow now continues through preparation, stopping before model training:

```text
Project → import → preview + EDA → target/features → review recipe
        → five-row before/after preview → save processed copy
        → choose split → save prepared train/validation/test files
```

Start fresh or continue an existing project:

```bash
workbench
workbench flowers --sklearn iris
workbench flowers --continue
```

In a terminal, imports automatically offer **Review preprocessing now** or **Save for later**. For redirected input/scripts, `--continue` explicitly enables the questions. Use `--no-continue` to stop after importing and EDA; `workbench flowers --status` shows saved artifacts without asking questions. `--no-open` and `--no-eda` retain their EDA-specific meanings and do not disable preparation questions.

### Review the recipe

1. Select the target column. Its values are protected from feature preprocessing.
2. Choose classification or regression. This configures validation and the usual split strategy; no model is trained.
3. Optionally exclude ID or unwanted columns using their displayed column numbers. Excluded features are omitted from prepared data. If you intend to split by a group/time column, exclude that control from feature preprocessing here.
4. Choose whether missing targets should stop preparation or be excluded during splitting.
5. Choose **Review suggested changes**, **Load custom recipe JSON**, **Keep features unchanged**, or **Save for later**.
6. For each suggested/custom recipe step, choose **Keep**, **Change settings**, or **Skip**. Missing-value changes offer mean, median, mode, or a constant value. Other transformations accept a parameter object.
7. Optionally add a custom transformation using the implemented transform catalog, selected columns, and JSON parameters.

Suggestions reuse the existing EDA-based recipe rules. The wizard saves your reviewed recipe and its approval fingerprints. Targets and excluded features cannot be transformed. Invalid recipes fail validation rather than silently changing the target.

A custom recipe file may contain a raw recipe such as this (replace `Age` with an actual feature):

```json
{
  "format_version": 1,
  "name": "Median age cleanup",
  "steps": [
    {"operation": "fill_missing", "columns": ["Age"], "params": {"strategy": "median"}}
  ]
}
```

The wizard also accepts a saved preprocessing-proposal wrapper. It asks you to review imported steps; the JSON file does not bypass approval.

### Preview and save preprocessing

The wizard shows up to five original feature rows before and after the recipe. Row indexes stay visible, so rows removed by a transformation do not appear to shift into another input row's place. The target is preserved in saved data but omitted from the feature-change preview.

Choose **Save processed data**, **Edit recipe**, or **Save recipe and finish later**. On save, the complete processed inspection copy is written as Parquet under the dataset's `processed/vN/data/` folder, alongside the reviewed recipe, learned values, and manifest. For presplit uploads this inspection copy contains the training upload only.

**The pre-split copy is for inspection.** Its learned statistics use the available source/training upload. The later split stage always starts from preserved raw rows and refits the same reviewed recipe on the final training split only. Validation and test use that fitted state. This prevents whole-source imputation from leaking into model evaluation; inspection values can therefore differ from final split values. The artifact metadata records this distinction.

### Choose and save a split

Unsplit data offers:

- 70% train / 15% validation / 15% test.
- 80% / 10% / 10%.
- 60% / 20% / 20%.
- Custom percentages totaling 100, with positive training size.

Choose the split strategy next: stratified (usual classification choice), random (usual regression choice), group, or chronological. Group/time strategies ask for their control column; chronological splits also ask for the date format. The seed defaults to 42. Invalid percentages can be corrected in the same prompt; incompatible split settings leave the saved processed copy intact.

For presplit uploads, supplied test/validation membership is preserved. If validation was not uploaded, choose 20%, 15%, 10%, or a custom percentage taken only from the training upload. The final preparation stores original split rows, assignment IDs, fitted preprocessing, and prepared split rows. Missing unlabeled-test targets remain supported.

Successful preparation prints the stored paths and row counts and automatically opens model selection. You can finish for now there, or choose and train a model. Completed training opens results and prediction tools. Returning to a project offers preprocessing/splits, models, and prediction/details/export; previous artifacts are preserved.

### Reopen and inspect saved work

```bash
workbench flowers
workbench flowers --status
workbench flowers --continue
workbench flowers --dataset-id data-PREVIOUS_ID --continue
```

The project status lists earlier dataset imports with copyable selection commands. Preparation state belongs to each dataset, so selecting an older import restores its own recipe/processed/split stage.

```text
projects/flowers/datasets/data-ID/
├── raw/                             # Preserved original files
├── profiles/overview-ID/
│   ├── report.html                  # Includes row selector
│   └── row_views.json               # Exact windows; integrity checked
├── recipes/project-ID.json          # Reviewed recipe
├── project-preparation.json         # Current stage + artifact history
├── processed/v1/
│   ├── data/part-00000.parquet       # Inspection copy
│   ├── recipe.json                  # Recipe, fit state, and purpose
│   └── manifest.json
└── tasks/task-ID/runs/run-ID/
    ├── assignments/part-00000.parquet
    ├── splits/{train,validation,test}/part-00000.parquet
    ├── prepared/{train,validation,test}/part-00000.parquet
    ├── preprocessing/{recipe.json,fitted.json}
    ├── split_config.json
    └── manifest.json
```

### HTML data viewer

New HTML EDA exports include **Explore the data** below the charts. Select **First 10 rows**, **Middle 10 rows**, or **Last 10 rows**. These are exact contiguous windows in source order, with one-based row numbers; small datasets show fewer rows and overlapping windows. The middle window is centered on the dataset, independently of the random chart sample. All columns are available through horizontal scrolling.

The viewer is offline, keyboard-accessible, and script-free. Collecting the exact middle window adds a bounded-memory pass over the source. HTML exports save up to 30 source rows in `row_views.json` and embed them in the HTML, even without `--save-sample`; the separate random chart sample still requires `--save-sample`. A saved run can re-export the viewer without reopening the original dataset. Older reports need regenerating with `workbench PROJECT --eda` to acquire the windows.

The following walkthroughs document the advanced command interface. Their standalone `datasets` workspace is separate from the new project flow.

## Contents

- [Installation](#installation)
- [Start here: a complete first experiment](#start-here-a-complete-first-experiment)
- [How suggested commands and IDs work](#how-suggested-commands-and-ids-work)
- [Importing your own data](#importing-your-own-data)
- [EDA and visualizations](#eda-and-visualizations)
- [Reviewing preprocessing recipes](#reviewing-preprocessing-recipes)
- [Tasks and train/validation/test preparation](#tasks-and-trainvalidationtest-preparation)
- [Training, tuning, and resuming](#training-tuning-and-resuming)
- [Reading results](#reading-results)
- [Prediction](#prediction)
- [Export and reproduction](#export-and-reproduction)
- [Local browser interface](#local-browser-interface)
- [Storage and repository layout](#storage-and-repository-layout)
- [Python integration](#python-integration)
- [Large datasets and current limits](#large-datasets-and-current-limits)
- [Command reference](#command-reference)
- [Testing and troubleshooting](#testing-and-troubleshooting)

## Installation

Run commands from the repository root: the directory containing `pyproject.toml`, `src/`, and `tests/`.

The project declares Python 3.10 or newer. Optional backend packages have their own Python/platform requirements. A virtual environment keeps this project's dependencies separate.

### Windows PowerShell

```powershell
git clone https://github.com/EricTLucas/ETML-Workbench.git
cd ETML-Workbench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
workbench --version
workbench --help
```

If you already have the repository, start at environment creation or installation. Apply the latest update files before installing. If activation is unavailable, use `.\.venv\Scripts\python.exe` in place of `python`, and `.\.venv\Scripts\workbench.exe` in place of `workbench`.

### macOS/Linux

```bash
git clone https://github.com/EricTLucas/ETML-Workbench.git
cd ETML-Workbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
workbench --version
```

The distribution is currently named `eda_tool`; the application is named ETML Workbench. Use the repository's `pyproject.toml` and editable installation for this version. The older `requirements.txt` is not the authoritative list of all workbench dependencies.

### Optional features

Install only the extras you need:

| Feature | Installation |
| --- | --- |
| Browser interface | `python -m pip install -e ".[ui]"` |
| Hugging Face datasets | `python -m pip install -e ".[huggingface]"` |
| XGBoost | `python -m pip install -e ".[xgboost]"` |
| PyTorch MLP | `python -m pip install -e ".[pytorch]"` |
| TensorFlow MLP | `python -m pip install -e ".[tensorflow]"` |
| Supported ONNX exports | `python -m pip install -e ".[onnx]"` |

For the interface and Hugging Face together:

```bash
python -m pip install -e ".[ui,huggingface]"
```

The base install already includes sklearn models, EDA, preprocessing, prediction, and bundle export. Optional dependencies are not all required to start.

## Start here: a complete first experiment

This example uses sklearn's bundled Iris dataset, so no CSV preparation or dataset download is necessary. Run these commands in order from the repository root.

```bash
workbench datasets from-sklearn iris --id iris
workbench datasets show iris --verify
workbench tasks create iris species --target target --type classification
workbench tasks prepare iris species
workbench models train iris species --metric f1_macro
workbench models test iris species
workbench models input-template --dataset iris --task species
workbench models export --dataset iris --task species --output exports/iris-model
```

What each command does:

1. Saves a local dataset snapshot and source metadata under `datasets/iris/`.
2. Shows the dataset manifest and checks its preserved file hashes.
3. Defines `species` as a classification task predicting the `target` column.
4. Creates a preparation run with stratified 70%/15%/15% train/validation/test splits. No explicit recipe is used in this quick start.
5. Trains the default sklearn linear and random-forest candidates, without adding a dummy baseline. Selects the winner using validation macro F1.
6. Evaluates that run's winner on the held-out test split. In a real experiment, tune using validation before this step.
7. Shows the raw feature names and types needed for prediction.
8. Exports the latest training run's winning model and the preprocessing needed for raw-input prediction.

The training encoder can handle ordinary missing numeric/categorical features even without an explicit recipe. The recipe workflow below provides deliberate, reviewable cleanup before that encoder.

### Predict one Iris example

Create `iris-example.json` in the repository root:

```json
{
  "sepal length (cm)": 5.1,
  "sepal width (cm)": 3.5,
  "petal length (cm)": 1.4,
  "petal width (cm)": 0.2
}
```

Then run:

```bash
workbench models predict-one --bundle exports/iris-model --example-file iris-example.json
```

The result includes the prediction, class probabilities, raw input, and the features after the reviewed recipe. Iris target values are numeric class codes; source provenance records their names.

### Optional EDA before training

After importing Iris, you can insert this before task creation:

```bash
workbench preprocess prepare iris --protect target --summary --html
```

This saves a profile and proposed recipe, and prints their locations. It does not modify the dataset or automatically approve its suggestions. Use the detailed recipe workflow below if you want the task to apply a recipe.

### Regression quick start

```bash
workbench datasets from-sklearn diabetes --id diabetes
workbench tasks create diabetes progression --target target --type regression
workbench tasks prepare diabetes progression
workbench models train diabetes progression --metric rmse
workbench models test diabetes progression
workbench models export --dataset diabetes --task progression --output exports/diabetes-model
```

Regression preparation defaults to a random split; lower RMSE is better.

## How suggested commands and IDs work

Most workflow-changing commands print **Suggested next steps** containing concrete dataset, recipe, preparation, training, or checkpoint identifiers. Copy the relevant command after inspecting the current result. These suggestions are guidance; they are not automatically executed.

| Identifier | Meaning | Example |
| --- | --- | --- |
| Dataset ID | One imported dataset and its preserved source | `titanic` |
| Task ID | A target, task type, and feature exclusions | `survival` |
| Recipe filename | One saved proposal/review revision | `recipe-<generated-id>.json` |
| Preparation run ID | One split plus fitted recipe and prepared data | Printed by `tasks prepare` |
| Training run ID | One candidate comparison or search | Printed by `models train` or `search` |
| Candidate ID | One model inside a training run | Printed in the leaderboard |
| Checkpoint path | A completed neural epoch with resume state | Printed during training or by `models checkpoints` |

Examples containing `RECIPE_FILE`, `REVIEWED_RECIPE_FILE`, `PREPARATION_RUN_ID`, `TRAINING_RUN_ID`, `CANDIDATE_ID`, or `CHECKPOINT_PATH` require substitution. Do not enter those placeholders literally.

`models train` and `models search` default to the latest preparation when its positional run ID is omitted. `models test`, `history`, and dataset-based prediction/export default to the latest training run. **Latest is not necessarily the best run across your experiments.** Use `models compare`, then pass explicit run/candidate IDs when selecting a final model.

Most workspace commands accept:

- `--workspace PATH`: dataset storage root, default `datasets` relative to the current directory.
- `--json`: machine-readable JSON on standard output, including `next_steps` when available.
- `--quiet`: suppress progress and human-readable next-step suggestions; it does not suppress the result.

For example:

```bash
workbench tasks prepare iris species --workspace datasets --json --quiet
```

Keep using the same workspace path on later commands. If you launch from another directory, provide an absolute workspace path.

## Importing your own data

### One unsplit file

```bash
workbench datasets import uploads/customers.csv --id customers --name "Customer churn"
workbench datasets list
workbench datasets show customers --verify
```

Import preserves original bytes in `raw/`; it does not run EDA or create processed data. Use a new dataset ID for a new import. Treat preserved raw files as immutable.

Multiple file arguments represent parts of one logical dataset, with matching columns:

```bash
workbench datasets import uploads/january.csv uploads/february.csv --id customers-2026
```

Use `import-split` to distinguish training and testing roles; ordinary multi-file import does not assign them.

### Existing train/test/validation files

```bash
workbench datasets import-split --train uploads/train.csv --test uploads/test.csv --validation uploads/validation.csv --id titanic
workbench preprocess prepare titanic --protect Survived PassengerId --summary --html
workbench tasks create titanic survival --target Survived --type classification --exclude PassengerId Name Ticket Cabin
```

All columns named above must actually exist. Adjust the exclusions to your data. Review and approve the proposed recipe as described below, then prepare with its reviewed filename.

If you omit the validation upload, preparation takes a validation portion only from the supplied training file:

```bash
workbench datasets import-split --train uploads/train.csv --test uploads/test.csv --id titanic-presplit
workbench tasks create titanic-presplit survival --target Survived --type classification --exclude PassengerId Name Ticket Cabin
workbench tasks prepare titanic-presplit survival --validation-fraction 0.2
workbench models train titanic-presplit survival
```

For presplit inputs:

- Supplied validation and test memberships are preserved; supplied test rows are not moved into training.
- Without supplied validation, `--validation-fraction` defaults to `0.2` of the supplied training set.
- `--train`, `--validation`, and `--test` are the ordinary unsplit-data fractions; use `--validation-fraction` for the presplit validation carve-out.
- EDA used for recipe suggestions reads the training source only.
- Recipe fitting uses training features only; learned values are reused for validation and test.
- Train/validation need the target. A test file may omit it completely.
- An unlabeled test file supports predictions but cannot produce F1, accuracy, or regression error metrics. Skip `models test` for that file.
- Compatible feature names are required across the supplied files.

### Dataset providers

```bash
workbench datasets from-sklearn wine --id wine
workbench datasets from-huggingface scikit-learn/iris --id hf-iris
workbench datasets from-openml 61 --id openml-iris
```

Built-in sklearn names: `iris`, `wine`, `breast_cancer`, `digits`, and `diabetes`.

For a direct file URL, replace the example address with an actual downloadable dataset file:

```bash
workbench datasets from-url "https://YOUR_HOST/path/data.csv" --id remote-data
```

This expects a direct CSV/TSV/Parquet/JSON/JSONL resource, not an arbitrary website page. `--filename data.csv` can supply the intended filename when the URL does not have a useful extension.

Hugging Face options include configuration, revision, selected columns, a single split, or explicit split roles:

```bash
workbench datasets from-huggingface OWNER/DATASET --config CONFIG_NAME --train-split train --validation-split validation --test-split test --revision REVISION --id hub-data
```

Substitute names that exist in the selected Hub dataset. For one split, use `--split train`. `--columns COLUMN_A COLUMN_B TARGET` restricts the imported fields. The importer currently supports scalar tabular columns, not nested records, images, audio, or other media fields. It resolves the requested revision to a commit and saves a local Parquet snapshot.

Source commands default to `--max-rows 200000` and `--max-memory-mb 512`, but these controls differ by source: URL/presplit imports use byte limits; row limits apply to materialized sklearn/OpenML/Hugging Face imports. OpenML fetches in memory before checking the resulting data. These options are guardrails, not a hard cap on process RAM. Exceeding an applicable limit fails rather than silently importing a truncated training dataset.

## EDA and visualizations

### Standalone analysis and HTML gallery

Standalone EDA accepts a source path without first registering it as a workspace dataset:

```bash
workbench eda analyze uploads/train.csv --output reports/titanic-eda --summary --html --save-sample
```

The output directory must be new. The result contains `profile.json`, a manifest, requested charts, optional `sample.parquet`, and optional self-contained `report.html`. Open `reports/titanic-eda/report.html` in a browser after this example. The CLI/workflow prints the run location; it does not automatically open the gallery.

- `--summary`: generate a bounded automatic chart collection.
- `--html`: include a gallery; also enables summary charts for this CLI command.
- `--save-sample`: persist the bounded sample as Parquet for later row-based visualizations.
- `--max-charts 10`: default summary chart limit.
- `--batch-size 50000`: rows per input batch.
- `--sample-size 10000`: retained sample size.
- `--max-pairs 100`: bound pair analysis.
- `--no-correlations`: disable profiler correlation calculations.
- `--seed 42`: sample reproducibility.

Charts use the dark Orchid style. The HTML report is a gallery of rendered charts that can be embedded in a UI; it is not an interactive model dashboard. Sample-based plots describe the retained observations, not necessarily every row.

### Column roles

By default, numeric or text columns with at most 10 distinct nonmissing values can be categorized after scanning the stream. Automatic category summary charts use pie charts. Override a role when the domain meaning differs:

```bash
workbench eda analyze uploads/train.csv --output reports/titanic-roles --html --role Age=numeric --role Pclass=category
```

Valid roles: `numeric`, `category`, `text`, `datetime`, `timedelta`. `--categorical-threshold 0` disables automatic low-cardinality category detection. Use `--dtype COLUMN=DTYPE` when the loader needs a stable physical type; a visualization role does not by itself convert the source values.

**EDA role overrides do not automatically configure model encoding.** During training, use `--categorical Pclass` to treat a numerically stored category as categorical.

### Request particular charts

```bash
workbench eda charts
workbench eda plot --run reports/titanic-eda --kind histogram --x Age --output plots/age.png
workbench eda plot --run reports/titanic-eda --kind scatter --x Age --y Fare --group Survived --output plots/age-fare.png
workbench eda plot --run reports/titanic-eda --kind pie --x Embarked --output plots/embarked.svg
workbench eda plot uploads/train.csv --kind raincloud --x Age --group Survived --output plots/age-raincloud.png
```

Charts may be saved as PNG, SVG, or PDF. Outputs must not already exist. `eda charts` lists accepted options per chart; `--option NAME=JSON_VALUE` passes additional options, such as `--option bins=20`.

Implemented chart kinds:

| Purpose | Kinds |
| --- | --- |
| Numeric distributions | `histogram`, `density`, `ecdf`, `box`, `violin`, `raincloud`, `ridgeline`, `strip`, `qq`, `outliers` |
| Numeric relationships | `scatter`, `bubble`, `density2d`, `hexbin`, `contour`, `joint` |
| Categories | `bar`, `lollipop`, `pie`, `donut`, `category_heatmap`, `grouped_bar`, `stacked_bar` |
| Missing data | `missing_bar`, `missing_matrix`, `missing_patterns` |
| Associations | `correlation`, `association` |
| Text | `word_frequency`, `wordcloud`, `text_length` |
| Ordered/time data | `line`, `time_series`, `date_counts` |
| Multiple features | `scatter_matrix`, `parallel` |

A chart needs compatible columns and enough usable observations. Loading a saved EDA run never silently reopens its original source. Row-based charts need a saved sample; otherwise generate from the source or rerun EDA with `--save-sample`.

Export the charts already stored in a run to another gallery:

```bash
workbench eda report --run reports/titanic-eda --output reports/titanic-gallery.html --title "Titanic exploration"
```

This does not automatically collect standalone PNGs generated later with `eda plot`.

## Reviewing preprocessing recipes

A recipe is an ordered list of transformations. EDA supplies evidence for suggestions; it does not automatically approve them or learn imputation values for the final model.

### Full review path

Assuming `titanic` has been imported:

```bash
workbench preprocess prepare titanic --protect Survived PassengerId Name Ticket Cabin --summary --html
workbench preprocess show titanic --recipe RECIPE_FILE --output recipe-edit.json
```

Copy `RECIPE_FILE` from the preparation output. Open `recipe-edit.json` in an editor. Preserve the complete exported recipe structure, including step IDs. Adjust operations, columns, parameters, or enabled state as needed.

Protect the target and columns you plan to exclude. Recipes used in task preparation operate only on the feature columns that remain, so remove steps referring to excluded columns before using that recipe.

After reviewing the file:

```bash
workbench preprocess review titanic --recipe RECIPE_FILE --from-file recipe-edit.json --approve-all
workbench preprocess preview titanic --recipe REVIEWED_RECIPE_FILE --max-rows 20
workbench tasks create titanic survival --target Survived --type classification --exclude PassengerId Name Ticket Cabin
workbench tasks prepare titanic survival --recipe REVIEWED_RECIPE_FILE
workbench models train titanic survival --categorical Pclass
```

`review` creates a new recipe revision. Use **the new filename it prints** in subsequent commands. `preview` displays a small before/after example; fitting done for a preview is not the final training fit. Task preparation learns the actual recipe state from its training split.

Instead of approving all enabled steps, use `--approve STEP_ID` or `--reject STEP_ID`, repeated as necessary. Editing a previously approved step invalidates its approval fingerprint; review the edited revision again.

### Implemented transformations

```bash
workbench preprocess transforms
```

| Operation | Purpose and selected parameters |
| --- | --- |
| `fill_missing` | `strategy`: `constant`, `mean`, `median`, or `mode`; `value` for constant; optional `fallback` and `max_unique` |
| `drop_missing` | Remove rows missing selected values; `how`: `any` or `all` |
| `convert_type` | `dtype`: `Int64`, `Float64`, `string`, `boolean`, or `datetime`; `errors`: `raise`/`coerce`; datetime requires explicit `format`, with optional `utc` |
| `normalize_categories` | Strip whitespace, select case, collapse whitespace, optionally treat empty strings as missing |
| `map_categories` | Explicit `{from, to}` mappings and an unknown-value policy |
| `select_columns` | Keep specified columns |
| `drop_columns` | Remove specified columns |
| `rename_columns` | `names`: a list of new names in the same order as the step's selected columns |

Current automatic suggestions include mean imputation for numeric missingness, mode imputation for categorical missingness, reviewable removal of fully observed constant features, and trimming observed whitespace. All-missing columns are not assigned guessed values. These are deterministic rules; there is no LLM integration yet.

To change a numeric missing-value step to median, change its `params` to:

```json
{"strategy": "median"}
```

Do not replace the entire recipe file with that fragment. KNN and iterative/MICE imputation are not implemented recipe options. Exact median/mode fitting retains distinct-value counts and fails if the configured `max_unique` cap is exceeded.

### Standalone processed versions

For cleanup outside a model task, execute an approved recipe into a new dataset version:

```bash
workbench preprocess execute titanic --recipe REVIEWED_RECIPE_FILE --fit-data uploads/training-only.csv --summary --html
```

For an intentionally whole-source fit, `--fit-on-source` is explicit. Do not use that approach to learn preprocessing from an unsplit dataset and then claim held-out model evaluation: it would let evaluation data influence learned transformations. The normal ML route is `tasks prepare --recipe ...` from raw data.

Export and reuse fitted transformations:

```bash
workbench preprocess export-fitted titanic VERSION_ID --output fitted-recipe.json
workbench preprocess execute titanic --recipe REVIEWED_RECIPE_FILE --fitted fitted-recipe.json
```

Substitute the processed version ID printed by execution. Source/recipe compatibility is checked. `--no-profile` skips profiling after execution. An exit code of 3 means processing succeeded but the subsequent profile failed; inspect the reported processed version before retrying.

## Tasks and train/validation/test preparation

A task records a target column, classification/regression type, excluded features, and a policy for missing targets. One dataset can have multiple tasks.

```bash
workbench tasks create customers churn --target Churn --type classification --exclude CustomerId
workbench tasks show customers churn
workbench tasks prepare customers churn --train 0.7 --validation 0.15 --test 0.15 --seed 42
```

Adjust column names to match your data. Missing target values fail by default; use `--missing-target drop` at task creation to deliberately exclude those rows. The target is never passed into feature-recipe fitting.

Available split strategies:

| Strategy | Use | Additional option |
| --- | --- | --- |
| `stratified` | Classification with class proportions represented | Default for classification |
| `random` | Independent observations | Default for regression |
| `group` | Keep a customer/patient/entity together | `--group-column CustomerId` |
| `chronological` | Train on earlier observations, evaluate later ones | `--time-column Timestamp`, optionally `--time-format` |

```bash
workbench tasks prepare customers churn --strategy group --group-column CustomerId
workbench tasks prepare customers churn --strategy chronological --time-column Timestamp
```

These are alternative examples requiring those columns. Choose the strategy appropriate to the data; do not randomly split repeated entities or temporal records merely because it is the default. Split-control columns are excluded from model features. Small datasets or scarce classes may not support the requested split.

Preparation writes split assignments, original split data, fitted recipe state, and prepared splits. Inspect the printed run ID:

```bash
workbench tasks inspect-run customers churn PREPARATION_RUN_ID --verify
```

`split_counts` describes membership before recipe filtering; `prepared_counts` describes rows remaining afterward. A row-dropping operation may change them.

Raw data is the default source. `--source-version VERSION_ID --allow-processed-source` is available only when you deliberately confirm the processed source used fixed cleanup rather than learned preprocessing that would leak information across splits.

## Training, tuning, and resuming

### Available model adapters

```bash
workbench models list
```

| Model key | Classification | Regression | Epochs/resume |
| --- | --- | --- | --- |
| `sklearn:dummy` | Prior baseline | Mean baseline | No |
| `sklearn:linear` | Logistic regression | Ridge regression | No workbench epoch loop |
| `sklearn:random_forest` | Random forest classifier | Random forest regressor | No |
| `xgboost:boosted_trees` | Boosted trees | Boosted trees | Boosting history/early stopping; no neural-style resume |
| `pytorch:mlp` | Tabular MLP | Tabular MLP | Yes |
| `tensorflow:mlp` | Tabular MLP | Tabular MLP | Yes |

The table above lists the original adapters. The expanded catalog includes additional sklearn estimators, LightGBM, CatBoost, CNNs, transformers, clustering, projections, forecasting and recommendation models. Catalog entries describe supported adapters, not whether optional libraries are installed. A custom NumPy backend remains a future extension.

### Train candidates

```bash
workbench models train titanic survival --model sklearn:linear --model sklearn:random_forest --categorical Pclass --metric f1_macro --label initial-comparison
```

Training uses the latest preparation unless you provide it explicitly:

```bash
workbench models train titanic survival PREPARATION_RUN_ID --model sklearn:random_forest
```

Only requested models are trained. A dummy baseline is available if explicitly selected; training and search never add one automatically. The winner is selected on validation only. Default selection metrics are `balanced_accuracy` for classification and `rmse` for regression.

The feature encoder fits on training data: numeric missing values use median imputation, categorical values use one-hot encoding, and unknown categories are ignored by the fitted one-hot encoder. Numeric scaling defaults on for linear/MLP models, SVM, KNN and the linear-containing voting/stacking ensembles. Numeric columns are numeric unless listed in `--categorical`; EDA's low-cardinality category inference is separate.

For multiple model configurations, create `models.json`:

```json
[
  {
    "backend": "sklearn",
    "algorithm": "linear",
    "seed": 42,
    "categorical_columns": ["Pclass"],
    "params": {"C": 1.0}
  },
  {
    "backend": "sklearn",
    "algorithm": "random_forest",
    "seed": 42,
    "categorical_columns": ["Pclass"],
    "params": {"n_estimators": 200, "max_depth": 8}
  }
]
```

```bash
workbench models train titanic survival --config models.json --metric f1_macro
```

That linear configuration is for classification; regression Ridge has different parameters. Configuration files may also set `scale_numeric`. Put per-model settings inside the file rather than combining it with the CLI's per-model flags. For one model, `--params` accepts a JSON object, but a file avoids shell-specific quoting problems for larger configurations.

### Neural training

```bash
workbench models train titanic survival --model pytorch:mlp --epochs 50 --batch-size 64 --hidden-sizes 64 32 --learning-rate 0.001 --patience 8 --categorical Pclass
```

Use `--model tensorflow:mlp` for the TensorFlow adapter. An epoch is a pass through the training set; batch size controls minibatches within that pass. Validation loss drives neural early stopping. The candidate-selection metric still determines the winner among models.

```bash
workbench models history titanic survival TRAINING_RUN_ID --candidate CANDIDATE_ID --plot plots/mlp-history.png
workbench models checkpoints titanic survival
workbench models resume "CHECKPOINT_PATH" --epochs 100
```

`--epochs 100` means continue to a **total of 100 epochs**, not add 100. Resume uses a completed checkpoint, including optimizer/training state, and creates a new training run. If the checkpoint had already stopped for patience, use `--reset-patience` deliberately. Copy the concrete resume command printed by training/checkpoint listing.

### XGBoost

```bash
workbench models train titanic survival --model xgboost:boosted_trees --learning-rate 0.05 --early-stopping-rounds 20 --categorical Pclass
```

Boosting rounds are distinct from neural epochs. Use `n_estimators` in a configuration file or parameter object to set the maximum number of trees/rounds. History reports boosting progress where available.

### Hyperparameter search and comparison

```bash
workbench models search titanic survival --model sklearn:random_forest --method random --trials 10 --metric f1_macro --categorical Pclass
workbench models runs titanic survival
workbench models compare titanic survival
```

To define a search space, save `search-space.json`:

```json
{
  "n_estimators": [100, 200, 400],
  "max_depth": [null, 6, 12],
  "min_samples_leaf": [1, 3]
}
```

```bash
workbench models search titanic survival --model sklearn:random_forest --space search-space.json --method random --trials 6 --max-trials 30 --metric f1_macro
```

Grid search uses `--method grid`; `--max-trials` caps the allowed search budget. Default spaces exist for random forests, XGBoost, and the MLP adapters. Supply a compatible custom space for other models. Search uses a fixed validation split, not cross-validation.

Compare runs using the same preparation and metric for an interpretable ranking:

```bash
workbench models compare titanic survival --preparation-run PREPARATION_RUN_ID --metric f1_macro
workbench models test titanic survival TRAINING_RUN_ID --candidate CANDIDATE_ID
```

Run the final test evaluation after deciding which candidate to keep. The CLI does not prevent you from repeatedly looking at test scores; preserving a useful held-out estimate is part of the experiment workflow.

## Reading results

The training table shows candidate, backend/algorithm, baseline flag, validation selection score, macro F1 (classification) or MAE (regression), fit seconds, and improvement over the run's baseline.

Classification results include accuracy, balanced accuracy, macro/weighted F1, macro precision/recall, binary F1 for binary tasks, and log loss when probabilities are available. ROC AUC is currently provided for eligible binary evaluations; it may be `null` when undefined. Per-class results and confusion matrices are stored with candidate metrics.

Regression results include MAE, RMSE, and R². Undefined metrics are represented as `null`. Higher is better for accuracy/F1/R²; lower is better for MAE/RMSE/log loss.

Details worth checking:

- Confusion-matrix rows are actual classes; columns are predicted classes.
- The binary positive class is explicitly recorded; do not assume an arbitrary label is positive.
- Improvement is direction-aware: a positive value means better than baseline for the selected metric.
- `fit_seconds` measures model fitting; neural fitting includes checkpoint-writing overhead. Encoding and total timing are recorded separately.
- Validation scores select the model. Test results are produced separately by `models test`.
- Baseline improvement is not proof of usefulness; check errors and class-specific behavior for your use case.

## Prediction

### One example

```bash
workbench models input-template --dataset titanic --task survival
workbench models predict-one --dataset titanic --task survival --input uploads/test.csv --row 0
```

`--row` is zero-based. Alternatively use `--example-file example.json`, containing a single object of raw column values, or `--values` with a JSON object. For `input-template`, copy the nested `values` object into your example file and fill it in; do not pass the whole template wrapper as the example.

Predictions apply the saved recipe and encoder. You provide raw inputs, not manually encoded features. `prepared_features` in the result shows the recipe output, not the final one-hot/scaled matrix. If you also provide the actual target, the response includes correctness for classification or residual for regression.

### A complete file

```bash
workbench models predict uploads/test.csv --dataset titanic --task survival --output predictions/titanic.csv
```

Or use an exported bundle independently of dataset/task lookup:

```bash
workbench models predict uploads/test.csv --bundle exports/titanic-model --output predictions/titanic-from-bundle.csv --batch-size 10000
```

The output must be a new CSV file. It contains input-row references, status, predictions, and class-probability columns where requested. The command result supplies the mapping from probability column names to actual classes. Rows excluded by recipe filtering remain identifiable rather than shifting the remaining row correspondence. `--strict` rejects unexpected input columns; required feature columns are checked in either mode. `--no-probabilities` omits probability columns.

To select a particular experiment rather than the latest winner, add `--run TRAINING_RUN_ID --candidate CANDIDATE_ID` to dataset/task selection. Direct `--bundle` selection is mutually exclusive with those run/candidate selectors.

## Export and reproduction

### Full prediction bundle: usual export

```bash
workbench models export --dataset titanic --task survival --run TRAINING_RUN_ID --candidate CANDIDATE_ID --format bundle --output exports/titanic-model
```

The bundle includes trained model parameters, fitted preprocessing/encoding, schemas, labels, configuration, metrics, environment/setup information, and checksums. Use it for raw-data inference through ETML. Export destinations must be new and outside the source bundle.

### Native or ONNX model

```bash
workbench models export --bundle exports/titanic-model --format native --output exports/titanic-native
workbench models export --bundle exports/titanic-model --format onnx --sample-data uploads/train.csv --output exports/titanic-onnx
```

Native export contains the backend model and input-schema information. Native and ONNX exports expect the **encoded numeric feature matrix**; they do not include the raw-input preprocessing pipeline. Classification outputs need the saved label mapping.

ONNX is currently supported for the sklearn linear and random-forest adapters, not every backend. It requires the ONNX extra and raw sample data for conversion parity checking. A successful export does not deploy a service or upload the model to an external host.

### Reproduce a training result

Each new bundle has `SETUP.md`, pinned `requirements.txt`, and `reproducibility.json`. To include the code and data needed to retrain:

```bash
workbench models setup --dataset titanic --task survival --run TRAINING_RUN_ID --candidate CANDIDATE_ID --output exports/titanic-replay
```

The original preparation and matching source code must still be available. A resumed model also needs its original checkpoint when creating the export. The package includes training/validation rows, the original model, source snapshot, fitted preprocessing, and resume state when applicable. Held-out test rows are omitted. Because it contains data, this is a different deliverable from an ordinary inference bundle.

Follow the package's generated `SETUP.md`, using its exact recorded Python version. Windows example, from inside the replay directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python reproduce.py
```

Replay validates hashes and environment versions, retrains with the preserved preparation, and writes `retrained/` containing a model bundle and `replay_parity.json`. It compares validation predictions/probabilities with the original. `matches` uses `rtol=1e-5`, `atol=1e-6`; `exact_predictions` reports exact prediction-array equality. A mismatch exits with code 1. The replay output must not already exist.

Saved bundles retain the original trained parameters. Retraining can still differ across hardware/library builds; the parity report makes that distinction explicit. Replay retrains the model using saved preprocessing rather than rerunning the entire raw-import/recipe-selection process.

## Local browser interface

The project UI uses FastAPI and a custom HTML/CSS/JavaScript frontend. It replaces Streamlit and requires no Node.js build.

From your repository root:

```bash
python -m pip install -e ".[ui]"
workbench ui
```

To use a different projects directory, pass the same root used by the CLI:

```bash
workbench ui --projects-dir "D:/ETML/projects"
```

The default is `projects/` relative to your current directory, or `ETML_PROJECTS_DIR` when set. Existing standalone `datasets/` folders are not project roots and are not automatically moved. The old UI's `--workspace` option is replaced by `--projects-dir`.

The server binds to `127.0.0.1:8501` and opens your browser. Use `--port 8502` for another port, `--headless` to open the browser yourself, or `--max-upload-mb 1024` to raise the default 512 MB import limit. Stop with Ctrl+C.

### Browser walkthrough

1. **Choose a project.** Create a named project or select an existing one. The brief loading screen uses the ETML identity and the rest of the interface follows the EDA's dark palette.
2. **Add data.** Drag a file onto the upload area, select a file, enter a local path, or choose a library provider. Formats: CSV, TSV, Parquet, JSON and JSONL. The local path is interpreted on the computer running the server. Raw files are copied into the project.
3. **Choose a library dataset.** Scikit-learn includes Iris, Wine, Breast Cancer, Digits and Diabetes. OpenML accepts a numeric data ID; Hugging Face accepts a repository, optional configuration/revision, and split; direct download accepts a data-file URL. Install `".[ui,huggingface]"` for Hugging Face. Remote providers need network access and appropriate dataset permissions. Provider imports are capped at 200,000 rows where supported by the existing source adapter.
4. **Explore the data.** EDA generation is on by default and can be unchecked during import. The Preview tab embeds the report; click it or **Open report** to open the full report in a new tab. The full EDA retains its first/middle/last row controls. The separate raw table shows the first 10 source rows. If EDA fails, the raw import stays saved and **Generate EDA** lets you retry.
5. **Choose a target.** In **Preprocess & split**, pick the target and classification/regression task. Optional controls exclude columns or drop rows with missing targets. **Get recommendations** uses the existing profiler and proposal engine.
6. **Review the recipe.** Enable or disable suggestions, choose columns, adjust parameters, use quick imputation strategies, remove steps or add a custom transform. Parameters use the existing recipe JSON format. **Approve & preview changes** approves the enabled steps and shows up to five source rows that actually changed, before and after (or first rows for reference with an explicit no-change message). Highlighted columns are affected by the recipe. Row indexes preserve alignment when steps drop rows.
7. **Save processed data.** Saving writes a new version into the project's dataset. This is an exploration copy. Final training preparation splits raw data and fits the reviewed recipe on training rows only.
8. **Create splits.** Choose 70/15/15, 80/10/10, 60/20/20, 80/0/20, or custom percentages totaling 100. Methods include random, stratified (classification), group and chronological. Group/time methods require a column; chronological dates must be ISO 8601. Existing presplit datasets imported through the CLI retain their uploaded test/validation partitions; if needed, choose a validation holdout from their training data.
9. **Return later.** The dataset selector can reopen older imports. **Saved project files** lists raw data, processed versions and split locations. Reopening a project restores its saved target, recipe and split state. An unsaved preview must be generated again after restarting the server.
10. **Train a model.** Choose **Continue to models** after splitting, or open **Models** in the left navigation. Select a category and suggested model, edit default hyperparameters, train with live backend progress, inspect Results and Model details, export a ZIP, and try dataset-row or custom Predictions.

Import/profiling/preparation jobs run in the background and display their current phase. The browser blocks overlapping actions for the same project. Keep the server running until the operation finishes. Reloading can reconnect to an active job; job history itself is not persisted across server restarts. Avoid editing the same project from the CLI while a browser operation is running.

### UI source and tests

- `src/etml/ui.py`: launch command and local server configuration.
- `src/etml/web/server.py`: API, local session checks, static resources and scoped report serving.
- `src/etml/web/service.py`: project operations, background jobs and preview tokens.
- `src/etml/web/static/`: HTML, CSS and JavaScript, included in installed wheels.
- `tests/test_web_ui.py`: API tests covering the preparation flow, persistence, reports, invalid input, stale previews and failures.

```bash
python -m pip install -e ".[ui,test-ui]"
python -m unittest discover -s tests -p "test_web_ui.py" -v
```

The UI is for a single user's local machine. It is not a hosted multi-user service. Raw data, reports and prepared copies stay under the project root; selecting a remote dataset downloads from that provider.

## Storage and repository layout

Source code lives in `src/`; imported and generated dataset artifacts live outside it. The `src/data/` package is code, while root-level `datasets/` is runtime storage.

```text
ETML-Workbench/
├── pyproject.toml
├── README.md
├── src/
│   ├── etml/             # CLI dispatch, display helpers, local UI
│   ├── eda_tool/         # Loader, profiler, visualizer, HTML and EDA artifacts
│   ├── data/             # Workspace, manifests, provider imports
│   ├── preprocessing/    # Recipe validation, fitting, execution, transforms
│   ├── tasks/            # Prediction-task definitions and validation
│   ├── splitting/        # Split strategies and presplit handling
│   ├── models/           # Registry, configuration, backend adapters
│   ├── training/         # Encoding, experiments, metrics, checkpoints, replay
│   ├── prediction/       # Bundle loading and inference
│   ├── exporting/        # Bundle/native/ONNX export
│   └── workflows/        # Operations combining the packages
├── tests/
├── datasets/             # Default workspace, generated as needed
├── reports/              # Example standalone EDA output location
├── predictions/          # Example prediction output location
└── exports/              # Example model/replay output location
```

A dataset's logical storage includes:

```text
datasets/DATASET_ID/
├── manifest.json
├── raw/                              # Preserved source files
├── profiles/                         # EDA runs
├── recipes/                          # Proposed and reviewed revisions
├── processed/VERSION_ID/             # Optional standalone preprocessing output
└── tasks/TASK_ID/
    ├── task.json
    ├── runs/PREPARATION_RUN_ID/
    │   ├── manifest.json
    │   ├── split_config.json
    │   ├── splits/{train,validation,test}/
    │   ├── preprocessing/{recipe.json,fitted.json}
    │   └── prepared/{train,validation,test}/
    ├── training/TRAINING_RUN_ID/      # Candidate bundles and comparison results
    └── checkpoints/                  # Completed neural epoch state
```

The tree highlights major artifacts rather than every file. Read the printed `directory` and manifests for actual generated paths. Task-prepared files live under the preparation run, not the standalone `processed/` directory.

Keep raw data, generated models, and sensitive reports out of version control unless you intentionally choose otherwise. Use Git for code and configuration; keep dataset/model storage policies explicit.

## Python integration

The CLI and UI call reusable workflows. For example, run EDA directly:

```python
from eda_tool.profiler import ProfileConfig
from eda_tool.workflows import run_eda

result = run_eda(
    "uploads/train.csv",
    profile_config=ProfileConfig(batch_size=50_000, sample_size=10_000),
    generate_summary=True,
    export_html=True,
    save_sample=True,
    output_dir="reports/python-eda",
)
try:
    print(result.profile["summary"].data)
    print(result.html_path)
finally:
    result.close()
```

Or create and train a task:

```python
from data import DatasetWorkspace
from data.sources import import_sklearn
from tasks import TaskConfig
from workflows.preparation import prepare_task
from models import ModelConfig
from training import train_models

workspace = DatasetWorkspace("datasets")
import_sklearn(workspace, "iris", dataset_id="iris-python")
task = TaskConfig("species", "target", "classification")
prepared = prepare_task(workspace, "iris-python", task)
trained = train_models(
    workspace, "iris-python", "species", prepared.run_id,
    configs=[ModelConfig(backend="sklearn", algorithm="random_forest")],
    metric="f1_macro",
)
print(trained.bundle)
```

Predict from a saved bundle:

```python
from prediction.example import predict_example

result = predict_example("exports/iris-model", {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
})
print(result)
```

These examples create artifacts and require new output paths/dataset IDs. The model registry and adapter contract provide an extension point for a future custom implementation; registering an adapter is separate from merely installing another library.

## Large datasets and current limits

The pipeline is batch-aware, but the entire workbench is not an out-of-core/distributed training system.

| Stage | Current behavior |
| --- | --- |
| CSV/TSV/delimited and Parquet loading | Lazy, replayable batches without concatenating the whole source |
| Other loader formats | In-memory fallback; may need additional format dependencies |
| EDA | Streaming aggregates plus bounded samples/cardinality/pair analysis |
| Recipe fitting | Repeated batch passes; exact median/mode may retain many distinct values |
| Task preparation | Batch-oriented split and Parquet writing; split bookkeeping also consumes resources |
| Training/search/test evaluation | Loads bounded prepared data into memory |
| File prediction | Reads and predicts in batches |
| Hugging Face import | Streams scalar rows into a local snapshot |

Training defaults are `--max-rows 200000`, `--max-memory-mb 512`, and `--max-features 50000` (the feature cap applies to train/search). Raise them deliberately for a machine that can support the workload:

```bash
workbench models train customers churn --max-rows 500000 --max-memory-mb 2048 --max-features 20000
```

These checks do not guarantee a process-level memory ceiling. Raw DataFrames, encoders, intermediate arrays, backend allocations, and model state add overhead; neural dense matrices can be especially large. Batch size limits rows, not bytes. Lowering the loader batch size does not turn sklearn/MLP fitting into streaming training.

For larger data, prefer Parquet, stable dtypes, relevant columns, bounded chart samples, and manageable categorical cardinality. Avoid one-hot encoding arbitrary IDs or unrestricted free text. Date/time features currently need explicit derivation or exclusion before model training.

Profiler results distinguish full-stream statistics from sampled analyses. Read section/chart metadata when interpreting estimates. Exact duplicate tracking is disabled by default, and high-cardinality or pairwise analyses have limits; the profiler does not promise every statistic is exact over unlimited data.

## Command reference

Use `--help` on a leaf command for its complete parser options:

```bash
workbench --help
workbench datasets import-split --help
workbench preprocess prepare --help
workbench tasks prepare --help
workbench models train --help
workbench models predict-one --help
workbench eda charts
```

| Group | Commands |
| --- | --- |
| `datasets` | `import`, `import-split`, `from-sklearn`, `from-huggingface`, `from-url`, `from-openml`, `list`, `show` |
| `eda` | `analyze`, `plot`, `charts`, `report` |
| `preprocess` | `prepare`, `show`, `review`, `preview`, `execute`, `export-fitted`, `transforms` |
| `tasks` | `create`, `show`, `prepare`, `inspect-run` |
| `models` | `list`, `train`, `search`, `resume`, `runs`, `compare`, `history`, `checkpoints`, `test`, `input-template`, `predict-one`, `predict`, `export`, `setup` |
| `ui` | Launch project interface; options include `--projects-dir`, `--port`, `--headless`, `--max-upload-mb` |

`python -m etml` is an alternative entry point if the `workbench` executable is not on your path. The legacy `eda INPUT --out DIRECTORY` command remains a compatibility alias for EDA with an HTML gallery.

## Testing and troubleshooting

Run unittest discovery from the repository root after installing the package:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run the latest data/import/prediction/replay tests only:

```bash
python -m unittest discover -s tests -p "test_data_experience.py" -v
```

Install optional backend/UI/provider extras to exercise their integration tests; a base-only environment cannot exercise every optional feature. The update was validated with 137 existing tests and 12 added tests, plus a live Hugging Face import and package installation checks.

| Symptom | What to check |
| --- | --- |
| `workbench` is not found | Activate the environment, reinstall with `python -m pip install -e .`, or use `python -m etml` |
| `Start directory is not importable: 'outputs'` | Use the repository's `tests` directory in unittest discovery, not an artifact/output folder |
| Dataset or task not found | Check the dataset/task ID, current directory, and `--workspace` |
| Dataset ID or output already exists | Reuse the existing dataset, or choose a new ID/output path; commands preserve earlier artifacts |
| Recipe is not approved or fingerprint changed | Run `preprocess review` and use the newly printed revision filename |
| Recipe references an excluded/target column | Protect those columns while proposing, or remove their steps before task preparation |
| Numeric categories are encoded as numbers | Add `--categorical COLUMN`; an EDA role is not a model encoding setting |
| CSV role/type changes between batches | Supply stable `--dtype` for standalone EDA or dtype mappings in workflow `--loader-options` |
| Saved-run plot lacks data | Analyze with `--save-sample`, or plot directly from the source |
| Model training exceeds a limit | Reduce rows/features/cardinality, or explicitly raise the applicable budget |
| Missing optional library | Install the matching project extra into the same Python environment |
| No test labels | Use prediction; metrics require ground truth |
| No epoch history for a forest/linear model | These adapters do not expose a neural epoch loop |
| Replay reports changed source/environment | Use the matching code and exact recorded Python/dependencies; read the generated setup file |
| Shell rejects inline JSON | Prefer configuration/example files; quoting rules differ between shells |

For a new dataset, the usual entry point is `workbench datasets import ...` or a provider import. Inspect the output, follow its suggested preparation command, review any recipe, and continue through task preparation and training. Keep the chosen preparation/training IDs when an experiment needs to be repeatable.



## Results, comparison and CSV prediction update (0.16)

The browser installs missing optional packages for the selected model in the
Python environment running the workbench. Training shows the installation stage
and continues automatically. Internet access and a supported Python/platform are
required. Existing incompatible versions produce an actionable restart/update
message rather than silently replacing an already loaded library. Large neural
packages can take several minutes to install.

Results starts with a teal Test metrics panel. The Compare tab groups completed,
evaluated supervised tabular models by their saved dataset and split. Classification
defaults to accuracy; regression defaults to RMSE. Select another available metric
to change the ranking. Loss/error metrics rank lowest first. Models without test
metrics remain unranked. Continue using validation for repeated tuning; repeated
selection using test results compromises their independence.

In Predictions, supervised tabular models offer Predict a CSV. Upload raw feature
columns, then download the generated CSV. The model's target column contains the
predictions. If that column was supplied, its original values are preserved in a
separate `_actual` column (with a suffix to avoid overwriting existing columns).
Every original row is retained. Rows dropped by the fitted recipe have a blank
prediction and are counted as excluded in the UI. Files are processed in batches
of 10,000 rows, subject to the configured upload size limit; outputs are saved in
project exports. Image, language-generation, forecasting and recommendation inputs
continue to use their specialized prediction forms rather than this tabular CSV flow.

Newly generated EDA reports omit missing-value plots when population missingness is
zero. Nonzero missing percentages and exact distinct counts equal to the row count
are red. Summary charts include up to three highest-magnitude computed association
pairs: scatterplots for numeric pairs, heatmaps for categorical pairs, and grouped
boxplots for mixed pairs. Titles identify the association method and value; mixed
methods are not interchangeable statistical tests. Row plots retain the existing
sampling labels and chart limits. Regenerate an existing saved report to see these
changes; saved HTML snapshots are not rewritten automatically.


## Filtered CSV exports and analysis summaries (0.17)

After **Predictions → Generate predictions CSV**, use the column checkboxes to
choose which columns to export. **Include rows** accepts source row numbers and
ranges such as `1, 3-10`; blank includes every row. **Exclude rows** removes selected
rows and takes priority. Row numbers start at 1, excluding the CSV header. Click
**Create filtered CSV** and download the result. Each filter operation starts from
the full generated CSV, so changing filters never requires another prediction.
The original and filtered exports remain in the project exports directory.
Prediction status is no longer appended to either CSV. Excluded-by-preprocessing
rows keep a blank prediction; their count is shown in the UI. Original source
cell text is preserved in exported feature columns, including leading zeros.

**Models → Results** shows loss curves when the saved model has recorded loss
history, including train-only runs and supported boosting-round histories. No
validation curve or loss values are fabricated when unavailable.

**Models → Visualize → Generate HTML summary** downloads one self-contained HTML
report. Select the dataset/split cohort first. The report contains fresh EDA,
an accuracy-ranked test leaderboard, and details/results/recorded loss curves of
the best model in that cohort. Regression uses lowest test RMSE. Ties use model
name for deterministic selection. Rankings do not mix different test splits, and
changing the interactive leaderboard metric does not change the report's default
accuracy/RMSE criterion. With no evaluated models, export contains EDA and an
explicit no-model-results note. Images are embedded; viewing requires no server
or internet connection. Generating fresh EDA can take time on large datasets.


## Classification visualizations (0.18)

Select a completed tabular classification model, then open **Models → Visualize**.
Choose a first feature column and optionally a different second column. Choose
train, validation or test from the model's saved splits and click **Generate
classification graph**. One feature gives a 1D strip plot with display-only vertical
jitter; two features give a 2D scatterplot. Categorical axes are supported up to 30
sampled categories. True and predicted class panels share axes, rows and class colors.
These are observed-row plots, not decision-boundary projections.

Each plot uses a uniform, reproducible sample of at most 2,000 split rows. Source,
sampled and plotted counts are saved; missing/nonfinite axes and rows excluded by
preprocessing are omitted from both panels. Unknown true labels are gray. Generated
PNGs and their metadata are saved under the project's visualizations folder.

Below each graph, **Add this graph to the HTML summary** saves your inclusion choice.
The summary export controls now live below the graph gallery in Visualize, rather
than Compare. Choose the summary dataset/split and generate the HTML. Checked graphs
from models sharing that exact dataset/split are embedded, with model names, feature
names, split and sample notes. Other cohorts' graphs remain saved but are excluded.
Regression and specialized models can still export summaries here, but these column
classification plots require a supervised tabular classification model.


Leaderboard fallback: when a completed tabular model has no test split/data,
Compare uses its recorded training metrics and labels the cohort **Training
fallback**. The HTML summary applies the same rule and explicitly identifies
in-sample scores. Models with test data but no successful test evaluation stay
unranked until evaluation succeeds; training scores do not silently replace a
failed test evaluation.
