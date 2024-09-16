import logging
import sys
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

from llama_index.core.schema import BaseNode

from airtrain.core import CreationArgs, DatasetMetadata, Unpack, upload_from_dicts


logger = logging.getLogger(__name__)


if sys.version_info < (3, 11):
    # My theory is that this fails below 3.11 because of some change in what
    # pydantic can do to serialize to dicts below that version. This theory is
    # as-yet unverified, however.
    logger.warning(
        "Airtrain can extract more information from LlamaIndex data when "
        "using python versions >=3.11. Consider upgrading if possible."
    )

_TRANSFORM_COLUMNS: Dict[str, Callable[[Any], Union[str, Dict[str, str]]]] = {
    "relationships": lambda r: _flatten(r, "relationships"),
    "metadata": lambda r: _flatten(r, "metadata"),
}


def upload_from_llama_nodes(
    data: Iterable[BaseNode], **kwargs: Unpack[CreationArgs]
) -> DatasetMetadata:
    """Upload an Airtrain dataset from the provided LlamaIndex nodes.

    Parameters
    ----------
    data:
        Nodes may be from documents, chunks, or anything else that results
        in instances of `BaseNode`.  Attributes of the nodes will be treated
        as columns in the resulting Airtrain dataset. If present (and python
        version is >= 3.11), the relationships and metadata for the node
        will be flattened into multiple columns of the resulting dataset.
    kwargs:
        See `upload_from_arrow_tables` for other arguments.

    Returns
    -------
    A DatasetMetadata object summarizing the created dataset.
    """
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
            result = transform(node_dict[transform_column])
            if isinstance(result, str):
                node_dict[transform_column] = result
            else:
                # column is being flattened
                del node_dict[transform_column]
                node_dict.update(result)
    return node_dict


def _flatten(value: Dict[str, Any], key_prefix: str) -> Dict[str, str]:
    flattened: Dict[str, str] = {}
    to_flatten: List[Tuple[str, Any]] = [
        (f"{key_prefix}.{k}", v) for k, v in value.items()
    ]
    while len(to_flatten) > 0:
        key, val = to_flatten.pop(0)
        if isinstance(val, dict):
            to_flatten.extend([(f"{key}.{k}", v) for k, v in val.items()])
        elif val is not None:
            # we only flatten dicts, not lists. This is intentional to keep the column
            # count from getting too high.
            flattened[key] = str(val)
    return flattened
