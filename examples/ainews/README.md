# AI News Ingestion

[AI News](https://buttondown.com/ainews/) is a newsletter that summarizes
activity from around the web on large language/multimodal models. This
example demonstrates using the Airtrain SDK + Llama Index to ingest the
newsletter's archive into Airtrain for analysis.

## Setup

To run the sample yourself, you'll need to sign up for an Airtrain account
[here](https://app.airtrain.ai/select-task), then activate a Pro
plan or Pro trial. Note that the full example will only run with a Pro
account; the trial is only sufficient for a dataset consisting of the
Newsletters themselves, without looking at a dataset of chunks of the
Newsletter.

Once you have an account, install the requirements with
`pip install requirements.txt` and then run the example as:

`AIRTRAIN_API_KEY="<Your Airtrain API Key>" python3 ingest_ainews.py`
