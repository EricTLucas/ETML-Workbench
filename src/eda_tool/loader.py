"""Dataset access for EDA.

open_dataset() is lazy for CSV/TSV/delimited text and Parquet. iter_batches()
never concatenates the dataset. load_dataset() explicitly materializes it for
the old pipeline. Batch size limits rows, NOT bytes; parser buffers, Parquet
metadata and very large individual values still consume memory.

CSV schema types are inferred from a preview and may change in later batches.
Supply dtype overrides when stable types are required. Files/uploads must stay
available and unchanged throughout use. Upload streams must be seekable and
must not be used concurrently. Other supported formats are in-memory fallbacks.
"""
from __future__ import annotations

from contextlib import contextmanager
from glob import glob, has_magic
from pathlib import Path
import json
import numpy as np
import pandas as pd

__all__ = ["DatasetSource", "open_dataset", "load_dataset",
           "load_from_upload", "load_from_path_file", "load_from_path_folder",
           "convert_to_dataframe"]

_TEXT = {".csv", ".tsv", ".txt", ".dat", ".tab"}
_SUPPORTED = _TEXT | {".parquet", ".xls", ".xlsx", ".xlsm", ".xlsb", ".ods",
    ".json", ".jsonl", ".ndjson", ".feather", ".orc", ".arrow", ".dta",
    ".sas7bdat", ".sav", ".zsav", ".pkl", ".pickle", ".h5", ".hdf5"}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _extension(item):
    return Path(item.filename if hasattr(item, "filename") else item).suffix.lower()


@contextmanager
def _input(item):
    """Borrow upload streams without copying, closing, or changing their position."""
    if not hasattr(item, "filename"):
        yield item
        return
    stream = getattr(item, "file", None)
    if stream is None or not stream.seekable():
        raise ValueError("Uploads need a seekable .file stream; spool streams to disk first")
    position = stream.tell()
    try:
        stream.seek(0)
        yield stream
    finally:
        stream.seek(position)


class DatasetSource:
    """Replayable collection of frames/files, with strict column-name matching.

    Multiple files must have the same column names; order may differ. Types are
    not silently forced from a CSV preview. dtype overrides are enforced while
    reading. schema() validates names but its inferred CSV types are provisional.
    """

    def __init__(self, items, *, dtype=None, csv_options=None, allow_pickle=False):
        self._items = tuple(items)
        self.dtype = dtype
        self.csv_options = dict(csv_options or {})
        self.allow_pickle = allow_pickle
        forbidden = {"chunksize", "iterator", "nrows", "usecols", "dtype", "index_col"}
        overlap = forbidden.intersection(self.csv_options)
        if overlap:
            raise ValueError(f"Use the source API instead of CSV options: {sorted(overlap)}")
        if self.csv_options.get("engine") == "pyarrow":
            raise ValueError("CSV batching requires the pandas C or Python engine")
        for item in self._items:
            if not isinstance(item, pd.DataFrame):
                ext = _extension(item)
                if ext not in _SUPPORTED:
                    raise ValueError(f"Unsupported dataset file type: {ext}")
                if ext in {".pkl", ".pickle"} and not allow_pickle:
                    raise ValueError("Pickle can execute code; set allow_pickle=True only for trusted files")

    @property
    def supports_streaming(self):
        """False if any file requires full materialization before slicing."""
        return all(isinstance(x, pd.DataFrame) or _extension(x) in _TEXT | {".parquet"}
                   for x in self._items)

    def _csv_kwargs(self, ext):
        options = dict(self.csv_options)
        if "sep" not in options and "delimiter" not in options:
            options["sep"] = "\t" if ext == ".tsv" else ("," if ext == ".csv" else None)
        if options.get("sep", ",") is None:
            options.setdefault("engine", "python")
        if self.dtype is not None:
            options["dtype"] = self.dtype
        return options

    def _cast(self, frame):
        if self.dtype is None:
            return frame
        dtype = ({k: v for k, v in self.dtype.items() if k in frame.columns}
                 if isinstance(self.dtype, dict) else self.dtype)
        return frame.astype(dtype)

    def _schema_item(self, item, preview_rows):
        if isinstance(item, pd.DataFrame):
            return self._cast(item.iloc[:0]).dtypes
        ext = _extension(item)
        with _input(item) as value:
            if ext in _TEXT:
                return pd.read_csv(value, nrows=preview_rows, **self._csv_kwargs(ext)).dtypes
            if ext == ".parquet":
                import pyarrow.parquet as pq
                with pq.ParquetFile(value) as reader:
                    schema = reader.schema_arrow
                    result = pd.Series({field.name: str(field.type) for field in schema})
                    if len(schema.names) != len(set(schema.names)):
                        raise ValueError("Duplicate column names are not supported")
                    if self.dtype is not None:
                        for col in result.index:
                            if not isinstance(self.dtype, dict) or col in self.dtype:
                                result[col] = str(self.dtype[col] if isinstance(self.dtype, dict) else self.dtype)
                    return result
            return self._cast(_read_eager(value, ext)).dtypes

    def schema(self, preview_rows=1000):
        """Return {column: type}; CSV types use a bounded preview, Parquet metadata.

        Non-streaming formats require a full read. Empty physical CSVs raise
        pandas EmptyDataError; header-only CSVs retain their column names.
        """
        _positive(preview_rows, "preview_rows")
        first = None
        for item in self._items:
            types = self._schema_item(item, preview_rows)
            if not types.index.is_unique:
                raise ValueError("Duplicate column names are not supported")
            if first is None:
                first = types
            elif set(types.index) != set(first.index):
                raise ValueError("Dataset files must have matching column names")
        return {} if first is None else {col: str(dtype) for col, dtype in first.items()}

    def _batches(self, item, columns, batch_size):
        if isinstance(item, pd.DataFrame):
            for start in range(0, len(item), batch_size):
                yield self._cast(item.iloc[start:start + batch_size].loc[:, columns].copy())
            return
        ext = _extension(item)
        with _input(item) as value:
            if ext in _TEXT:
                with pd.read_csv(value, usecols=columns, chunksize=batch_size,
                                 **self._csv_kwargs(ext)) as reader:
                    for frame in reader:
                        yield frame.loc[:, columns]
            elif ext == ".parquet":
                import pyarrow.parquet as pq
                with pq.ParquetFile(value) as reader:
                    for batch in reader.iter_batches(batch_size=batch_size, columns=columns):
                        yield self._cast(batch.to_pandas().loc[:, columns])
            else:
                frame = _read_eager(value, ext)
                for start in range(0, len(frame), batch_size):
                    yield self._cast(frame.iloc[start:start + batch_size].loc[:, columns].copy())

    def iter_batches(self, columns=None, batch_size=50_000):
        """Yield DataFrames with at most batch_size rows, in source order.

        Close the returned generator if stopping early to release active readers.
        No automatic concatenation occurs. Non-streaming formats read one entire
        file at a time. Dtype inference may vary by batch unless overridden.
        """
        _positive(batch_size, "batch_size")
        names = list(self.schema())
        if isinstance(columns, str):
            raise TypeError("columns must be a sequence of column names, not a string")
        selected = names if columns is None else list(columns)
        if columns is not None and not selected:
            raise ValueError("Select at least one column")
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate selected columns")
        missing = set(selected) - set(names)
        if missing:
            raise ValueError(f"Unknown columns: {sorted(missing, key=str)}")
        for item in self._items:
            yield from self._batches(item, selected, batch_size)

    def sample(self, n=10_000, seed=42, columns=None, batch_size=50_000):
        """Uniform sample without replacement via random priorities.

        Scans ALL rows, retaining at most n rows plus one batch and temporary
        selection buffers. Same seed/source order gives repeatable selections.
        Output is in random-priority order. Sampling does not stratify classes.
        """
        _positive(n, "n")
        rng = np.random.default_rng(seed)
        kept = None
        keys = np.empty(0)
        iterator = self.iter_batches(columns=columns, batch_size=batch_size)
        try:
            for frame in iterator:
                frame = frame.reset_index(drop=True)
                new_keys = rng.random(len(frame))
                candidates = frame if kept is None else pd.concat([kept, frame], ignore_index=True)
                combined = np.concatenate([keys, new_keys])
                size = min(n, len(combined))
                indices = np.argsort(combined, kind="stable")[:size]
                keys = combined[indices]
                kept = candidates.iloc[indices].copy()
        finally:
            iterator.close()
        if kept is None:
            return pd.DataFrame(columns=list(self.schema()) if columns is None else list(columns))
        return kept.reset_index(drop=True)


def open_dataset(source, *, dtype=None, csv_options=None, allow_pickle=False):
    """Open a frame, file, folder, glob, file list, records, or upload list.

    Folders select supported files at the top level in sorted order; globs can
    explicitly select nested files. Matching column names are required across
    files. Opening does not read dataset contents. Partition keys encoded only
    in Parquet folder names are not synthesized by this file-based reader.
    """
    if isinstance(source, DatasetSource):
        if dtype is not None or csv_options is not None or allow_pickle:
            raise ValueError("Configure an existing DatasetSource when creating it")
        return source
    if isinstance(source, pd.DataFrame) or hasattr(source, "filename"):
        items = [source]
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            items = [path]
        elif path.is_dir():
            items = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED)
        elif has_magic(str(source)):
            items = sorted(Path(p) for p in glob(str(source), recursive=True) if Path(p).is_file())
        else:
            raise FileNotFoundError(source)
        if not items:
            raise ValueError(f"No dataset files found: {source}")
    elif isinstance(source, list):
        if not source:
            items = [pd.DataFrame()]
        elif all(isinstance(x, dict) for x in source) or all(isinstance(x, (list, tuple)) for x in source):
            items = [pd.DataFrame(source)]
        elif all(isinstance(x, pd.DataFrame) or hasattr(x, "filename") or isinstance(x, (str, Path)) for x in source):
            items = []
            for x in source:
                if isinstance(x, (str, Path)) and not Path(x).is_file():
                    raise FileNotFoundError(x)
                items.append(x)
        else:
            raise TypeError("List input must contain records, rows, frames, file paths, or uploads")
    else:
        raise TypeError(f"Unsupported dataset source: {type(source).__name__}")
    return DatasetSource(items, dtype=dtype, csv_options=csv_options, allow_pickle=allow_pickle)


def load_dataset(source, *, columns=None, batch_size=50_000, **options):
    """Compatibility API: returns a FULL in-memory DataFrame, even for large files.

    Use open_dataset(...).iter_batches() for bounded loading. The old profiler
    still requires a DataFrame; replacing this module alone does not stream EDA.
    """
    if isinstance(source, pd.DataFrame) and columns is None and not options:
        return source
    dataset = open_dataset(source, **options)
    frames = list(dataset.iter_batches(columns=columns, batch_size=batch_size))
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=list(dataset.schema()) if columns is None else columns))


def load_from_upload(files, **options):
    return load_dataset(files, **options)


def load_from_path_file(path, **options):
    return load_dataset(path, **options)


def load_from_path_folder(folder, **options):
    return load_dataset(folder, **options)


def convert_to_dataframe(file, **options):
    return load_dataset(file, **options)


def _read_eager(value, ext):
    """Legacy formats: full reads, with their usual optional engine dependencies."""
    if ext in {".xls", ".xlsx", ".xlsm", ".xlsb"}:
        return pd.read_excel(value)
    if ext == ".ods":
        return pd.read_excel(value, engine="odf")
    if ext == ".json":
        if isinstance(value, (str, Path)):
            with open(value, encoding="utf-8") as stream:
                obj = json.load(stream)
        else:
            position = value.tell()
            obj = json.load(value)
            value.seek(position)
        try:
            return pd.DataFrame(obj)
        except ValueError:
            return pd.json_normalize(obj)
    if ext in {".jsonl", ".ndjson"}:
        return pd.read_json(value, lines=True)
    if ext == ".feather":
        return pd.read_feather(value)
    if ext == ".orc":
        import pyarrow.orc as orc
        return orc.ORCFile(value).read().to_pandas()
    if ext == ".arrow":
        import pyarrow as pa
        import pyarrow.ipc as ipc
        if isinstance(value, (str, Path)):
            with pa.memory_map(str(value), "r") as stream:
                return ipc.RecordBatchFileReader(stream).read_all().to_pandas()
        return ipc.RecordBatchFileReader(value).read_all().to_pandas()
    if ext == ".dta":
        return pd.read_stata(value)
    if ext == ".sas7bdat":
        return pd.read_sas(value, format="sas7bdat")
    if ext in {".sav", ".zsav"}:
        return pd.read_spss(value)
    if ext in {".pkl", ".pickle"}:
        return pd.read_pickle(value)
    if ext in {".h5", ".hdf5"}:
        return pd.read_hdf(value)
    raise ValueError(f"Unsupported dataset file type: {ext}")
