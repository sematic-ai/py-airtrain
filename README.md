<div align="center">
  <img src="https://raw.githubusercontent.com/sematic-ai/py-airtrain/refs/heads/main/images/airtrain-logo.png" alt="Airtrain Ai Logo" style="vertical-align: middle; display: inline-block;" width="100px">
</div>

<p align="center">
  <a href="https://github.com/sematic-ai/py-airtrain/actions/workflows/ci.yaml?query=branch%3Amain+" target="_blank">
    <img height="30px" src="https://github.com/sematic-ai/py-airtrain/actions/workflows/ci.yaml/badge.svg?branch=main" alt="CI status">
  </a>
  <a href="./LICENSE" target="_blank">
    <img height="30px" src="https://img.shields.io/pypi/l/sematic?style=for-the-badge" alt="License">
  </a>
  <a href="https://airtrain.ai" target="_blank">
    <img height="30px" src="https://img.shields.io/badge/Made_by-Airtrain_🚀-blue?style=for-the-badge&logo=none" alt="Made by Airtrain">
  </a>
  <a href="https://docs.python.org/3.8/" target="_blank">
    <img height="30px" src="https://img.shields.io/badge/Python-3.8-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
  <a href="https://docs.python.org/3.9/" target="_blank">
    <img height="30px" src="https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
  <a href="https://docs.python.org/3.10/" target="_blank">
    <img height="30px" src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
  <a href="https://docs.python.org/3.11/" target="_blank">
    <img height="30px" src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
  <a href="https://docs.python.org/3.12/" target="_blank">
    <img height="30px" src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
</p>


# Airtrain SDK

This repository holds the SDK for interacting with
[Airtrain](https://www.airtrain.ai/),
the tool for improving your AI apps, RAG pipelines, and models by curating
high-quality training and eval datasets.

## Installation

To install the core package without any integrations, simply

`pip install airtrain-py`

You may install integrations by using pip extras. As an example, to
install the pandas integration:

`pip install airtrain-py[pandas]`

If you want to install all integrations, you may do the following:

`pip install airtrain-py[all]`

The following are available extras:

- `pandas`
- `polars`
- `llama-index`

## Usage

Obtain your API key by going to your user settings on
https://app.airtrain.ai .

Then you may upload a new dataset as follows:

```python
import airtrain as at

# Can also be set with the environment variable AIRTRAIN_API_KEY
at.set_api_key("sUpErSeCr3t")

url = at.upload_from_dicts(
    [
        {"foo": "some text", "bar": "more text"},
        {"foo": "even more text", "bar": "so much text"},
    ],
    name="My Dataset name",  # name is Optional
).url

# You may view your dataset in the Airtrain dashboard at this URL
# It may take some time to complete ingestion and generation of
# automated insights. You will receive an email when it is complete.
print(f"Dataset URL: {url}")
```

The data may be any iterable of dictionaries that can be represented using
automatically inferred [Apache Arrow](https://arrow.apache.org/docs/python/index.html)
types. If you would like to give a hint as to the Arrow schema of the data being
uploaded, you may provide one using the `schema` parameter to `upload_from_dicts`.

### Custom Embeddings

Airtrain produces a variety of insights into your data automatically. Some of
these insights (ex: automatic clustering) relies on embeddings of the data. Airtrain
will also embed your data automatically, but if you wish to provide your own embeddings
you may do so by adding the `embedding_column` parameter when you upload:

```python
url = at.upload_from_dicts(
    [
        {"foo": "some text", "bar": [0.0, 0.707, 0.707, 0.0]},
        {"foo": "even more text", "bar": [0.577, 0.577, 0.0, 0.577]},
    ],
    embedding_column="bar",
).url
```

If you provide this argument, the embeddings must all be lists of floating point
numbers with the same length.

### Integrations

Airtrain provides integrations to allow for uploading data from a variety of
sources. In general most integrations take the form of an `upload_from_x(...)`
function with a signature matching that of `upload_from_dicts` except for
the first parameter specifying the data to be uploaded. Integrations may require
installing the Airtrain SDK [with extras](#installation).

#### Pandas

```python
import pandas as pd

# ...

df = pd.DataFrame(
    {
        "foo": ["some text", "more text", "even more"],
        "bar": [1, 2, 3],
    }
)


url = at.upload_from_pandas(df, name="My Pandas Dataset").url
```

You may also provide an iterable of dataframes instead of a single one.

#### Polars

```python
import polars as pl

# ...

df = pl.DataFrame(
    {
        "foo": ["some text", "more text", "even more"],
        "bar": [1, 2, 3],
    }
)


url = at.upload_from_polars(df, name="My Polars Dataset").url
```

You may also provide an iterable of dataframes instead of a single one.


#### Arrow

```python
import pyarrow as pa

# ...

table = pa.table({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})


url = at.upload_from_arrow_tables([table], name="My Arrow Dataset").url
```


#### LlamaIndex

Note that these examples also involve installing additional Llama Index
integrations. A more detailed example of using Airtrain + Llama Index
can be found in the
[Llama Index docs](https://docs.llamaindex.ai/en/stable/examples/cookbooks/airtrain/)

```python
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding

# Data does not have to come from GitHub; this is for illustrative purposes.
github_client = GithubClient(...)
documents = GithubRepositoryReader(...).load_data(branch=branch)

# You can upload documents directly. In this case Airtrain will generate embeddings
result = at.upload_from_llama_nodes(
    nodes,
    name="My Document Dataset",
)
print(f"Uploaded {result.size} rows to {result.name}. View at: {result.url}")

# Or you can chunk and/or embed it first. Airtrain will use the embeddings
# you created via LlamaIndex.
embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(...)
nodes = splitter.get_nodes_from_documents(documents)
result = upload_from_llama_nodes(
    nodes,
    name="My embedded RAG Dataset",
)
print(f"Uploaded {result.size} rows to {result.name}. View at: {result.url}")
```

Alternatively, using the
["Workflows"](https://docs.llamaindex.ai/en/stable/module_guides/workflow/)
API:

```python
import asyncio

from llama_index.core.schema import Node
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.readers.web import AsyncWebPageReader

from airtrain import DatasetMetadata, upload_from_llama_nodes


URLS = [
    "https://news.ycombinator.com/item?id=41694044",
    "https://news.ycombinator.com/item?id=41696046",
    "https://news.ycombinator.com/item?id=41693087",
    "https://news.ycombinator.com/item?id=41695756",
    "https://news.ycombinator.com/item?id=41666269",
    "https://news.ycombinator.com/item?id=41697137",
    "https://news.ycombinator.com/item?id=41695840",
    "https://news.ycombinator.com/item?id=41694712",
    "https://news.ycombinator.com/item?id=41690302",
    "https://news.ycombinator.com/item?id=41695076",
    "https://news.ycombinator.com/item?id=41669747",
    "https://news.ycombinator.com/item?id=41694504",
    "https://news.ycombinator.com/item?id=41697032",
    "https://news.ycombinator.com/item?id=41694025",
    "https://news.ycombinator.com/item?id=41652935",
    "https://news.ycombinator.com/item?id=41693979",
    "https://news.ycombinator.com/item?id=41696236",
    "https://news.ycombinator.com/item?id=41696434",
    "https://news.ycombinator.com/item?id=41688469",
    "https://news.ycombinator.com/item?id=41646782",
    "https://news.ycombinator.com/item?id=41689332",
    "https://news.ycombinator.com/item?id=41688018",
    "https://news.ycombinator.com/item?id=41668896",
    "https://news.ycombinator.com/item?id=41690087",
    "https://news.ycombinator.com/item?id=41679497",
    "https://news.ycombinator.com/item?id=41687739",
    "https://news.ycombinator.com/item?id=41686722",
    "https://news.ycombinator.com/item?id=41689138",
    "https://news.ycombinator.com/item?id=41691530"
]


class CompletedDocumentRetrievalEvent(Event):
    name: str
    documents: list[Node]

class AirtrainDocumentDatasetEvent(Event):
    metadata: DatasetMetadata


class IngestToAirtrainWorkflow(Workflow):
    @step
    async def ingest_documents(
        self, ctx: Context, ev: StartEvent
    ) -> CompletedDocumentRetrievalEvent | None:
        if not ev.get("urls"):
            return None
        reader = AsyncWebPageReader(html_to_text=True)
        documents = await reader.aload_data(urls=ev.get("urls"))
        return CompletedDocumentRetrievalEvent(name=ev.get("name"), documents=documents)

    @step
    async def ingest_documents_to_airtrain(
        self, ctx: Context, ev: CompletedDocumentRetrievalEvent
    ) -> AirtrainDocumentDatasetEvent | None:
        if not isinstance(ev, CompletedDocumentRetrievalEvent):
            return None

        dataset_meta = upload_from_llama_nodes(ev.documents, name=ev.name)
        return AirtrainDocumentDatasetEvent(metadata=dataset_meta)

    @step
    async def complete_workflow(
        self, ctx: Context, ev: AirtrainDocumentDatasetEvent
    ) -> None | StopEvent:
        if not isinstance(ev, AirtrainDocumentDatasetEvent):
            return None
        return StopEvent(result=ev.metadata)


async def main() -> None:
    workflow = IngestToAirtrainWorkflow()
    result = await workflow.run(
        name="My HN Discussions Dataset", urls=URLS,
    )
    print(f"Uploaded {result.size} rows to {result.name}. View at: {result.url}")


if __name__ == "__main__":
    asyncio.run(main())
```
