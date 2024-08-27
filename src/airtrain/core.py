from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, Union


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
    name: Union[str, None] = None,
    embedding_column: Union[str, None] = None,
) -> DatasetMetadata:
    return DatasetMetadata(
        name=name or "My Dataset",
        id="abc123",
        url="https://example.com",
        size=0,
    )
