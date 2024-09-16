from typing import Any, Iterable, Union


try:
    import pandas as pd

    ENABLED = True
except ImportError:
    ENABLED = False
import pyarrow as pa

from airtrain.core import CreationArgs, DatasetMetadata, Unpack, upload_from_arrow_tables


# In case pandas is not installed
DataFrame = Any


def upload_from_pandas(
    data: Union[Iterable[DataFrame], DataFrame],
    **kwargs: Unpack[CreationArgs],
) -> DatasetMetadata:
    """Upload an Airtrain dataset from the provided pandas DataFrame(s).

    Parameters
    ----------
    data:
        Either an individual pandas DataFrame or an iterable of DataFrames.
        Data will be intermediately represented as pyarrow tables.
    kwargs:
        See `upload_from_arrow_tables` for other arguments.

    Returns
    -------
    A DatasetMetadata object summarizing the created dataset.
    """
    if not ENABLED:
        raise ImportError(
            "Pandas integration not enabled. Please install Airtrain package as "
            "`airtrain-py[pandas]`"
        )
    if isinstance(data, pd.DataFrame):
        data = [data]
    data = (pa.Table.from_pandas(df) for df in data)  # type: ignore

    return upload_from_arrow_tables(data, **kwargs)
