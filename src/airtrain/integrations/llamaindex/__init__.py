from typing import Any


try:
    from airtrain.integrations.llamaindex.core import (
        upload_from_llama_nodes,  # noqa: F401
    )
except ImportError:

    def upload_from_llama_nodes(*args, **kwargs) -> Any:
        raise NotImplementedError(
            "Llama Index integration not installed. Install with airtrain-py[llama-index]"
        )
