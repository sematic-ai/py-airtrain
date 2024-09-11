from itertools import count

import pyarrow as pa
import pytest

from airtrain.core import DatasetMetadata, upload_from_arrow_tables, upload_from_dicts
from tests.fixtures import MockAirtrainClient, mock_client  # noqa: F401


def test_upload_from_dicts(mock_client: MockAirtrainClient):  # noqa: F811
    data = [{"foo": 42}, {"foo": 43}, {"foo": 44}, {"foo": 45, "bar": "hi"}]
    name = "Foo dataset"
    result = upload_from_dicts(data, name=name)
    assert isinstance(result, DatasetMetadata)
    assert result.size == len(data)
    assert result.name == name
    fake_dataset = mock_client.get_fake_dataset(result.id)
    assert fake_dataset.name == name
    table = fake_dataset.ingested
    assert table is not None
    assert table.shape[0] == len(data)
    assert table["foo"].to_pylist() == [42, 43, 44, 45]
    assert table["bar"].to_pylist() == [None, None, None, "hi"]

    result = upload_from_dicts(
        data,
        name=name,
        schema=pa.schema(
            [
                ("foo", pa.float32()),
                ("bar", pa.string()),
            ]
        ),
    )
    fake_dataset = mock_client.get_fake_dataset(result.id)
    table = fake_dataset.ingested

    # make sure the schema was respected
    assert all(isinstance(v, float) for v in table["foo"].to_pylist())


def test_upload_from_dicts_limits(mock_client: MockAirtrainClient):  # noqa: F811
    # This would go forever if the code didn't stop early.
    data = ({"foo": i} for i in count())

    row_limit = mock_client.dataset_row_limit
    result = upload_from_dicts(data)
    assert isinstance(result, DatasetMetadata)
    assert result.size == row_limit


def test_upload_from_dicts_invalid(mock_client: MockAirtrainClient):  # noqa: F811
    data = [{"foo": 42}, {"foo": 43}, {"foo": 44}, {"foo": 45, "bar": "hi"}]
    with pytest.raises(pa.lib.ArrowInvalid):
        upload_from_dicts(
            data,
            schema=pa.schema(
                [
                    ("foo", pa.float32()),
                    ("bar", pa.float32()),
                ]
            ),
        )

    with pytest.raises(ValueError):
        upload_from_dicts([])

    with pytest.raises(ValueError):
        upload_from_dicts(["foo" for _ in range(0, 10)])


def test_upload_from_dicts_embedded(mock_client: MockAirtrainClient):  # noqa: F811
    data = [
        {"foo": 42, "bar": [1.0, 2.0]},
        {"foo": 43, "bar": [1.1, 2.1]},
        {"foo": 44, "bar": [1.2, 2.2]},
        {"foo": 45, "bar": [1.3, 2.3]},
    ]

    result = upload_from_dicts(data, embedding_column="bar")
    assert isinstance(result, DatasetMetadata)

    schema = pa.schema(
        [
            ("foo", pa.float32()),
            ("bar", pa.list_(pa.float32())),
        ]
    )
    result = upload_from_dicts(data, embedding_column="bar", schema=schema)
    assert isinstance(result, DatasetMetadata)

    schema = pa.schema(
        [
            ("foo", pa.float32()),
            ("bar", pa.list_(pa.float32(), 2)),
        ]
    )
    result = upload_from_dicts(data, embedding_column="bar", schema=schema)
    assert isinstance(result, DatasetMetadata)

    # integers should work too
    data = [
        {"foo": 42, "bar": [1, 2]},
        {"foo": 43, "bar": [3, 4]},
    ]
    result = upload_from_dicts(data, embedding_column="bar", schema=schema)
    assert isinstance(result, DatasetMetadata)


def test_bad_embeds(mock_client: MockAirtrainClient):  # noqa: F811
    data = [
        {"foo": 42, "bar": [1.0, 2.0]},
        {"foo": 43, "bar": [1.1, 2.1]},
    ]

    with pytest.raises(TypeError):
        upload_from_dicts(data, embedding_column="foo")

    with pytest.raises(ValueError):
        upload_from_dicts(data, embedding_column="baz")

    data = [
        {"foo": 42, "bar": ["hi", "hi"]},
        {"foo": 43, "bar": ["there", "there"]},
    ]
    with pytest.raises(TypeError):
        upload_from_dicts(data, embedding_column="bar")

    data = [
        {"foo": 42, "bar": [1.0, 2.0]},
        {"foo": 43, "bar": None},
    ]
    with pytest.raises(ValueError):
        upload_from_dicts(data, embedding_column="bar")

    data = [
        {"foo": 42, "bar": [1.0, 2.0]},
        {"foo": 43, "bar": [1.1, 2.1, 3.1]},
    ]
    with pytest.raises(ValueError):
        upload_from_dicts(data, embedding_column="bar")


def test_upload_from_arrow_tables(mock_client: MockAirtrainClient):  # noqa: F811
    table_1 = pa.table({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})
    table_2 = pa.table({"foo": [4, 5, 6], "bar": ["d", "e", "f"]})
    uploaded = upload_from_arrow_tables([table_1, table_2], name="My Arrow")
    fake_dataset = mock_client.get_fake_dataset(uploaded.id)
    table = fake_dataset.ingested
    assert table is not None
    assert table.shape[0] == table_1.shape[0] + table_2.shape[0]
    assert table["foo"].to_pylist() == [1, 2, 3, 4, 5, 6]
    assert table["bar"].to_pylist() == ["a", "b", "c", "d", "e", "f"]


def test_upload_from_mismatched_tables(mock_client: MockAirtrainClient):  # noqa: F811
    table_1 = pa.table({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})
    table_2 = pa.table({"foo": ["d", "e", "f"], "bar": [4, 5, 6]})

    with pytest.raises(ValueError):
        upload_from_arrow_tables([table_1, table_2], name="My Arrow")
