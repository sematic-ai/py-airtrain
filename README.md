<div align="center">
  <img src="images/airtrain-logo.png" alt="Airtrain Ai Logo" style="vertical-align: middle; display: inline-block;" width="150px">
</div>

<p align="center">
  <a href="https://airtrain.ai" target="_blank">
    <img src="https://img.shields.io/badge/Made_by-Airtrain_🚀-E19632?style=for-the-badge&logo=none" alt="Made by Airtrain">
  </a>
  <a href="https://docs.python.org/3.8/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.8-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
  <a href="https://docs.python.org/3.9/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
  <a href="https://docs.python.org/3.10/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
  <a href="https://docs.python.org/3.11/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
  <a href="https://docs.python.org/3.12/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
</p>


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
