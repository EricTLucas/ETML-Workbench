# ETML Workbench browser UI

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
10. **Train a model.** Choose **Continue to models** after splitting, or open **Models** in the left navigation. Select a category and suggested model, edit default hyperparameters, train with live backend progress, inspect Results and Model details, export a ZIP, and try dataset-row or custom Predictions. See [MODELS_UI_GUIDE.md](MODELS_UI_GUIDE.md).

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
