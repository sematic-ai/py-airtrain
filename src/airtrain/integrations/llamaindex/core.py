from itertools import chain
from typing import Iterable

from llama_index.core.schema import BaseNode

from airtrain.core import CreationArgs, DatasetMetadata, Unpack, upload_from_dicts


def upload_from_llama_nodes(
    data: Iterable[BaseNode], **kwargs: Unpack[CreationArgs]
) -> DatasetMetadata:
    data_as_iter = iter(data)
    try:
        first_node = next(data_as_iter)
    except StopIteration:
        raise ValueError("Cannot create data from empty iterable")

    has_embedding = first_node.embedding is not None
    if has_embedding and "embedding_column" not in kwargs:
        kwargs["embedding_column"] = "embedding"

    # Now that we peeked at the first node to see if it had an
    # embedding, replace it back at the start of the iterable.
    data = chain([first_node], data_as_iter)

    return upload_from_dicts(
        (node.to_dict() for node in data),
        **kwargs,
    )
