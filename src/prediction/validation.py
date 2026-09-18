from preprocessing.transforms.base import check_frame


def validate_input(frame, columns, *, strict=False):
    check_frame(frame)
    missing = set(columns)-set(frame.columns)
    if missing:
        raise ValueError(f'Missing required input columns: {sorted(missing)}')
    if strict and set(frame.columns)-set(columns):
        raise ValueError('Unexpected input columns')
    return frame.loc[:,list(columns)].copy().reset_index(drop=True)
