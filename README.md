<div align="center">
  <img src="images/airtrain-logo.png" alt="Airtrain Ai Logo" style="vertical-align: middle; display: inline-block;" width="150px">
</div>

# Airtrain SDK

This repository holds the SDK for interacting with
[Airtrain](https://www.airtrain.ai/),
the tool for improving your AI apps, RAG pipelines, and models by curating
high-quality training and eval datasets.

## Usage

Obtain your API key by going to your user settings on
https://app.airtrain.ai .

Then you may upload a new dataset as follows:

```python
import airtrain as at

# Can also be set with the environment variable AIRTRAIN_API_KEY
at.api_key = "sUpErSeCr3t"

url = at.upload_from_dicts(
    [
        {"foo": "some text", "bar": "more text"},
        {"foo": "even more text", "bar": "so much text"},
    ]
).url

# You may view your dataset at this URL
print(f"Dataset URL: {url}")
```
