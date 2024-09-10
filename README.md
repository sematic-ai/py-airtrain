<div align="center">
  <img src="images/airtrain-logo.png" alt="Airtrain Ai Logo" style="vertical-align: middle; display: inline-block;" width="100px">
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
