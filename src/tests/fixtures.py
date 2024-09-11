import io
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import patch, MagicMock

import pyarrow as pa
import pytest
from pyarrow import parquet as pq

from airtrain.client import (
    AirtrainClient,
    CreateDatasetResponse,
    NotFoundError,
    TriggerIngestResponse,
    client,
)


@pytest.fixture
def mock_client():
    client.cache_clear()
    mock = MagicMock()
    mock.return_value = MockAirtrainClient()
    with patch(f"{AirtrainClient.__module__}.{AirtrainClient.__name__}", new=mock):
        yield mock.return_value


@dataclass
class FakeDataset:
    id: str
    name: str
    ingestion_job_id: Optional[str] = None
    source_data: List[io.BufferedIOBase] = field(default_factory=list)
    ingested: Optional[pa.Table] = None


class MockAirtrainClient(AirtrainClient):
    def __init__(self) -> None:
        super().__init__(api_key="53c237", base_url="https://fake.local")
        self._fake_datasets: Dict[str, FakeDataset] = {}
        self.dataset_row_limit = 100

    def get_fake_dataset(self, dataset_id: str) -> FakeDataset:
        return self._fake_datasets[dataset_id]

    def trigger_dataset_ingest(self, dataset_id: str) -> TriggerIngestResponse:
        job_id = uuid.uuid4().hex
        if dataset_id not in self._fake_datasets:
            raise NotFoundError("Dataset not uploaded first")

        table = None
        for source in self._fake_datasets[dataset_id].source_data:
            table_part = pq.read_table(source)
            if table is None:
                table = table_part
            else:
                table = pa.concat_tables([table, table_part])

        self._fake_datasets[dataset_id].ingested = table
        return TriggerIngestResponse(ingest_job_id=job_id)

    def create_dataset(
        self, name: str, embedding_column_name: Optional[str]
    ) -> CreateDatasetResponse:
        dataset_id = uuid.uuid4().hex
        self._fake_datasets[dataset_id] = FakeDataset(id=dataset_id, name=name)
        return CreateDatasetResponse(
            dataset_id=dataset_id, row_limit=self.dataset_row_limit
        )

    def upload_dataset_data(self, dataset_id: str, data: io.BufferedIOBase) -> None:
        if dataset_id not in self._fake_datasets:
            raise NotFoundError("Dataset not uploaded first")
        self._fake_datasets[dataset_id].source_data.append(data)
