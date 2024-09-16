from typing import Any, Iterable, Union


try:
    import polars as pl

    ENABLED = True
except ImportError:
    ENABLED = False

from airtrain.core import CreationArgs, DatasetMetadata, Unpack, upload_from_arrow_tables


# In case polars is not installed
DataFrame = Any


def upload_from_polars(
    data: Union[Iterable[DataFrame], DataFrame],
    **kwargs: Unpack[CreationArgs],
) -> DatasetMetadata:
    """Upload an Airtrain dataset from the provided polars DataFrame(s).

    Parameters
    ----------
    data:
        Either an individual polars DataFrame or an iterable of DataFrames.
        Data will be intermediately represented as pyarrow tables.
    kwargs:
        See `upload_from_arrow_tables` for other arguments.

    Returns
    -------
    A DatasetMetadata object summarizing the created dataset.
    """
    if not ENABLED:
        raise ImportError(
            "Polars integration not enabled. Please install Airtrain package as "
            "`airtrain-py[polars]`"
        )
    if isinstance(data, pl.DataFrame):
        data = [data]

    data = (df.to_arrow() for df in data)  # type: ignore

    return upload_from_arrow_tables(data, **kwargs)
