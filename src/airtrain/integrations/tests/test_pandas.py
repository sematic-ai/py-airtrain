import numpy as np
import pandas as pd
import pytest

from airtrain.core import DatasetMetadata
from airtrain.integrations.pandas import upload_from_pandas
from tests.fixtures import MockAirtrainClient, mock_client  # noqa: F401


def test_upload_from_pandas(mock_client: MockAirtrainClient):  # noqa: F811
    df = pd.DataFrame(
        [
            {"foo": 42, "bar": "a"},
            {"foo": 43, "bar": "b"},
            {"foo": 44, "bar": "c"},
            {"foo": 45, "bar": "d"},
        ]
    )
    name = "Foo dataset"
    result = upload_from_pandas(df, name=name)
    assert isinstance(result, DatasetMetadata)
    assert result.size == df.shape[0]
    assert result.name == name
    fake_dataset = mock_client.get_fake_dataset(result.id)
    assert fake_dataset.name == name
    table = fake_dataset.ingested
    assert table is not None
    assert table.shape[0] == df.shape[0]
    assert table["foo"].to_pylist() == [42, 43, 44, 45]
    assert table["bar"].to_pylist() == ["a", "b", "c", "d"]


def test_upload_from_pandas_multiple(mock_client: MockAirtrainClient):  # noqa: F811
    df_1 = pd.DataFrame(
        [
            {"foo": 42, "bar": "a"},
            {"foo": 43, "bar": "b"},
            {"foo": 44, "bar": "c"},
            {"foo": 45, "bar": "d"},
        ]
    )
    df_2 = pd.DataFrame(
        [
            {"foo": 46, "bar": "e"},
            {"foo": 47, "bar": "f"},
            {"foo": 48, "bar": "g"},
            {"foo": 49, "bar": "h"},
        ]
    )
    result = upload_from_pandas((df_1, df_2))
    assert isinstance(result, DatasetMetadata)
    assert result.size == df_1.shape[0] + df_2.shape[0]
    fake_dataset = mock_client.get_fake_dataset(result.id)
    table = fake_dataset.ingested
    assert table is not None
    assert table.shape[0] == result.size
    assert table["foo"].to_pylist() == [42, 43, 44, 45, 46, 47, 48, 49]
    assert table["bar"].to_pylist() == ["a", "b", "c", "d", "e", "f", "g", "h"]


def test_upload_from_pandas_embeddings(mock_client: MockAirtrainClient):  # noqa: F811
    df = pd.DataFrame(
        [
            {"foo": 42, "bar": np.array([1.0, 0.0, 0.0, 0.0])},
            {"foo": 43, "bar": np.array([0.0, 1.0, 0.0, 0.0])},
            {"foo": 44, "bar": np.array([0.0, 0.0, 1.0, 0.0])},
            {"foo": 45, "bar": np.array([0.0, 0.0, 0.0, 1.0])},
        ]
    )
    result = upload_from_pandas(df, embedding_column="bar")
    assert isinstance(result, DatasetMetadata)
    assert result.size == df.shape[0]
    fake_dataset = mock_client.get_fake_dataset(result.id)
    table = fake_dataset.ingested
    assert table is not None
    assert table.shape[0] == df.shape[0]
    assert table["foo"].to_pylist() == [42, 43, 44, 45]
    assert table["bar"].to_pylist()[1] == [0.0, 1.0, 0.0, 0.0]

    df_bad = pd.DataFrame(
        [
            {"foo": 42, "bar": np.array([1.0, 0.0, 0.0, 0.0])},
            {"foo": 43, "bar": np.array([0.0, 1.0, 0.0, 0.0])},
            {"foo": 44, "bar": np.array([0.0, 0.0, 1.0])},
            {"foo": 45, "bar": np.array([0.0, 0.0, 0.0, 1.0])},
        ]
    )
    with pytest.raises(ValueError):
        # one row has a different number of embedding dimensions.
        upload_from_pandas(df_bad, embedding_column="bar")
