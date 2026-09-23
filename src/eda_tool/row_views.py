"""Bounded, exact windows in source order, independent of the random chart sample."""
import pandas as pd


def collect_row_views(dataset, total_rows, *, batch_size=50_000, width=10):
    if total_rows < 0 or width < 1:
        raise ValueError('Invalid row window size')
    count = min(width, total_rows)
    starts = {'first':0, 'middle':max(0, (total_rows-count)//2), 'last':max(0,total_rows-count)}
    parts = {key:[] for key in starts}
    offset = 0
    iterator = iter(dataset.iter_batches(batch_size=batch_size))
    try:
        for frame in iterator:
            for key, start in starts.items():
                left, right = max(start, offset), min(start+count, offset+len(frame))
                if right > left:
                    selected = frame.iloc[left-offset:right-offset].copy()
                    selected.index = range(left+1, right+1)
                    parts[key].append(selected)
            offset += len(frame)
    finally:
        close = getattr(iterator, 'close', None)
        if close:
            close()
    if offset != total_rows:
        raise ValueError('Source row count changed while collecting report previews')
    return {key:pd.concat(values) if values else pd.DataFrame(columns=list(dataset.schema()))
            for key, values in parts.items()}
