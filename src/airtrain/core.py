import io
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, fields
from datetime import datetime
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.compute import count as count_arrow

from airtrain.client import client


if sys.version_info > (3, 11):
    from typing import TypedDict, Unpack

    class CreationArgs(TypedDict):
        name: Optional[str]
        embedding_column: Optional[str]
else:
    # Unpack is only >3.11 . We'll just rely on type
    # checking in those versions to catch mistakes.
    # This will make Unpack[CreationArgs] into
    # Optional[Any] for lower versions, which should
    # pass checks.
    from typing import Optional as Unpack  # noqa
    from typing import Any as CreationArgs  # noqa


logger = logging.getLogger(__name__)


_MAX_BATCH_SIZE: int = 2000


@dataclass
class DatasetMetadata:
    name: str
    id: str
    url: str
    size: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, field.type):
                raise ValueError(
                    f"Field '{field.name}' must be {field.type}. Got: '{value}'"
                )


def upload_from_dicts(
    data: Iterable[Dict[str, Any]],
    schema: Optional[pa.Schema] = None,
    **kwargs: Unpack[CreationArgs],
) -> DatasetMetadata:
    """Upload an Airtrain dataset from the provided dictionaries.

    Parameters
    ----------
    data:
        An iterable of dictionary data to construct an Airtrain dataset out of.
        Each row in the data must be a python dictionary, and use only types
        that can be converted into pyarrow types. Data will be intermediately
        represented as pyarrow tables.
    schema:
        Optionally, the Arrow schema the data conforms to. If not provided, the
        schema will be inferred from a sample of the data.
    kwargs:
        See `upload_from_arrow_tables` for other arguments.

    Returns
    -------
    A DatasetMetadata object summarizing the created dataset.
    """
    data = iter(data)  # to ensure itertools works even if it was a list, etc.
    batches = _batched(data, _MAX_BATCH_SIZE)
    return upload_from_arrow_tables(
        data=_dict_batches_to_tables(batches, schema),
        **kwargs,
    )


def _is_arrow_number(type_: pa.DataType) -> bool:
    checks = [
        pa.types.is_floating,
        pa.types.is_integer,
        pa.types.is_decimal,
    ]
    return any(check(type_) for check in checks)


def _validate_embedding_field(
    table: pa.Table, embedding_column: str, expected_dim: Optional[int] = None
) -> int:
    if embedding_column not in table.column_names:
        raise ValueError(f"No column named '{embedding_column}' containing embeddings.")
    column_type = table.schema.field(embedding_column).type
    if (
        pa.types.is_list(column_type)
        or pa.types.is_large_list(column_type)
        or pa.types.is_fixed_size_list(column_type)
    ):
        val_type = column_type.value_type
        if not _is_arrow_number(val_type):
            raise TypeError(
                f"Embedding column must contain lists of numbers, not list of {val_type}"
            )
        first_vec_dimensions = len(table[embedding_column][0])
        if expected_dim is None:
            expected_dim = first_vec_dimensions
        if expected_dim != first_vec_dimensions:
            raise ValueError(
                f"Expected embeddings to have {expected_dim} "
                f"dimensions, got: {first_vec_dimensions}"
            )
        as_fixed_dim = pa.list_(val_type, expected_dim)
        column = table[embedding_column]
        try:
            column.cast(as_fixed_dim)
        except pa.lib.ArrowInvalid:
            raise ValueError(
                f"Not all embeddings in '{embedding_column}' were "
                f"{expected_dim} dimensions."
            )

        n_nulls = count_arrow(column, mode="only_null").as_py()
        if n_nulls > 0:
            raise ValueError(f"Found {n_nulls} null values in '{embedding_column}'")

    else:
        raise TypeError(
            f"Embedding column must contain lists of numbers. Got: {column_type}"
        )
    return expected_dim


def upload_from_arrow_tables(
    data: Iterable[pa.Table],
    name: Optional[str] = None,
    embedding_column: Optional[str] = None,
) -> DatasetMetadata:
    """Upload an Airtrain dataset from the provided dictionaries.

    Parameters
    ----------
    data:
        An iterable of arrow tables to construct an Airtrain dataset out of.
        Each row in the data must be an arrow table, and all tables must have
        the same schema.
    name:
        The name of the dataset you are creating, which will be shown in the
        Airtrain dashboard.
    embedding_column:
        The name of a column containing pre-computed embeddings for the data.
        The column must have non-null values for every row. Every row must be
        a list of numewric values, representing the embedding vector. All
        vectors must have the same dimensionality (length).

    Returns
    -------
    A DatasetMetadata object summarizing the created dataset.
    """
    name = name or f"My Dataset {datetime.now()}"
    c = client()
    creation_call_result = c.create_dataset(
        name=name, embedding_column_name=embedding_column
    )
    limit = creation_call_result.row_limit
    dataset_id = creation_call_result.dataset_id
    size = 0
    embedding_dim: Optional[int] = None
    schema: Optional[pa.Schema] = None

    for table in data:
        if schema is None:
            schema = table.schema
        if schema != table.schema:
            logger.error("Mismatched schemas:\n%s\n\n%s", schema, table.schema)
            raise ValueError("All uploaded tables must have the same schema.")
        if embedding_column is not None:
            embedding_dim = _validate_embedding_field(
                table, embedding_column, embedding_dim
            )
        table = table[: limit - size]
        table = _remove_illegal_parquet_types(table)

        upload_buffer = io.BytesIO()
        pq.write_table(table, upload_buffer)
        upload_buffer.seek(0)
        c.upload_dataset_data(dataset_id, upload_buffer)
        size += table.shape[0]

        if size >= limit:
            break

    if size == 0:
        raise ValueError("Cannot ingest empty dataset.")
    c.trigger_dataset_ingest(dataset_id)
    return DatasetMetadata(
        name=name,
        id=dataset_id,
        url=c.dataset_dashboard_url(dataset_id),
        size=size,
    )


T = TypeVar("T")


def _dict_batches_to_tables(
    batches: Iterable[Tuple[Dict[str, Any], ...]], schema: Optional[pa.Schema] = None
) -> Iterable[pa.Table]:
    for batch in batches:
        table = _dicts_to_table(batch, schema)
        if schema is None:
            # ensure later batches use the same schema.
            schema = table.schema
        yield table


def _dicts_to_table(
    dicts: Tuple[Dict[str, Any], ...], schema: Optional[pa.Schema]
) -> pa.Table:
    columns: Set[str] = set()
    for row in dicts:
        if not isinstance(row, dict):
            logger.error("Unexpected row: %s", row)
            raise ValueError("All data rows must be python dicts.")
        columns.update(row.keys())

    table_dict: Dict[str, List[Any]] = defaultdict(list)
    for row in dicts:
        for column in columns:
            table_dict[column].append(row.get(column))
    return pa.table(table_dict, schema=schema)


# This is in the standard lib in itertools as of 3.12; this code
# is adapted from documentation there.
def _batched(iterable: Iterable[T], n: int) -> Iterable[Tuple[T, ...]]:
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    batch: Tuple[T, ...] = ()
    while True:
        batch = tuple(islice(iterator, n))
        if len(batch) == 0:
            break
        yield batch


def _remove_illegal_parquet_types(table: pa.Table) -> pa.Table:
    schema = table.schema
    for name in schema.names:
        try:
            _assert_can_be_written_to_parquet(schema.field(name).type, [name])
        except TypeError as e:
            logger.warning(
                "Column '%s' cannot be written to parquet; skipping: %s",
                name,
                e,
            )
            table = table.drop_columns([name])
    return table


def _assert_can_be_written_to_parquet(
    arrow_type: pa.DataType, field_path: List[str]
) -> None:
    field_path_str = " -> ".join(field_path)
    if pa.types.is_union(arrow_type):
        # Arrow does indeed forbid writing unions to parquet,
        # but FWIW they aren't logically equivalent to python's
        # Union[T1, T2, ...]
        raise TypeError(
            f"Cannot serialize arrow union to Parquet. "
            f"Consider removing these values. "
            f"Offending union at: {field_path_str}"
        )
    if pa.types.is_struct(arrow_type):
        if arrow_type.num_fields == 0:
            raise TypeError(
                f"Cannot write struct with no fields to Parquet. "
                f"Consider removing these values or passing an explicit schema "
                f"with the struct as a map type. "
                f"Offending struct at: {field_path_str}"
            )
        for i_field in range(0, arrow_type.num_fields):
            field = arrow_type.field(i_field)
            _assert_can_be_written_to_parquet(field.type, field_path + [field.name])
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        _assert_can_be_written_to_parquet(arrow_type.value_type, field_path + ["[...]"])
    if pa.types.is_map(arrow_type):
        _assert_can_be_written_to_parquet(
            arrow_type.key_type, field_path + ["keys()[...]"]
        )
        _assert_can_be_written_to_parquet(arrow_type.item_type, field_path + ["[...]"])
