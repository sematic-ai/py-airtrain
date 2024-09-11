from itertools import chain
from typing import Any, Callable, Dict, Iterable

from llama_index.core.schema import BaseNode

from airtrain.core import CreationArgs, DatasetMetadata, Unpack, upload_from_dicts


_TRANSFORM_COLUMNS: Dict[str, Callable[[Any], str]] = {
    "relationships": lambda r: str(r),
}


def upload_from_llama_nodes(
    data: Iterable[BaseNode], **kwargs: Unpack[CreationArgs]
) -> DatasetMetadata:
    data_as_iter = iter(data)
    try:
        first_node = next(data_as_iter)
    except StopIteration:
        raise ValueError("No LlamaIndex nodes to upload.")

    has_embedding = first_node.embedding is not None
    if has_embedding and "embedding_column" not in kwargs:
        kwargs["embedding_column"] = "embedding"

    # Now that we peeked at the first node to see if it had an
    # embedding, replace it back at the start of the iterable.
    data = chain([first_node], data_as_iter)

    return upload_from_dicts(
        (_sanitize_dict(node.to_dict()) for node in data),
        **kwargs,
    )


def _sanitize_dict(node_dict: Dict[str, Any]) -> Dict[str, Any]:
    if "embedding" in node_dict and node_dict["embedding"] is None:
        del node_dict["embedding"]
    for transform_column, transform in _TRANSFORM_COLUMNS.items():
        if transform_column in node_dict:
            node_dict[transform_column] = transform(node_dict[transform_column])
    return node_dict
