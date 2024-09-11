import io
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional

import httpx


logger = logging.getLogger(__name__)


RequestJson = Dict[str, Any]
ResponseJson = Dict[str, Any]


API_KEY_ENV_VAR: str = "AIRTRAIN_API_KEY"
_DEFAULT_API_KEY: Optional[str] = None
_DEFAULT_BASE_URL: str = "https://api.airtrain.ai"
_BUFFER_CHUNK_SIZE = 8192


class BadRequestError(Exception):
    """The request was bad for some reason."""

    pass


class AuthenticationError(BadRequestError):
    """The caller does not have permission to complete the request."""

    pass


class NotFoundError(BadRequestError):
    """The caller requested something that doesn't exist."""

    pass


class ServerError(Exception):
    """There was some problem with the server."""

    pass


@dataclass
class CreateDatasetResponse:
    dataset_id: str
    row_limit: int


@dataclass
class TriggerIngestResponse:
    ingest_job_id: str


class AirtrainClient:
    """A direct wrapper around Airtrain's  HTTP API. Intended for internal package use.

    SDK users should NOT use this class directly and should not assume it will have a
    stable API. It makes no attempt to make sure calls are sequenced in an appropriate
    order or that information is passed between calls in a logical way. There should be
    a direct 1:1 correspondence between methods here and API endpoints that the SDK
    needs to interact with.
    """

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> None:
        self._api_key: str = api_key or _find_api_key()  # type: ignore
        self._base_url: str = base_url or os.environ.get(  # type: ignore
            "AIRTRAIN_API_URL", _DEFAULT_BASE_URL
        )
        if self._base_url.endswith("/"):
            self._base_url = self._base_url[:-1]
        self._http_client = httpx.Client()

        if self._api_key is None:
            raise AuthenticationError(
                "No Api key found. "
                "Set one with the environment variable 'AIRTRAIN_API_KEY' or the "
                "function airtrain.set_api_key"
            )

    def dataset_dashboard_url(self, dataset_id: str) -> str:
        """Get the webapp URL for a dataset, given its id."""
        api_url = self._base_url
        app_url = api_url.replace("://api.dev", "://airtrain.dev").replace(
            "://api.", "://app."
        )
        return f"{app_url}/dataset/{dataset_id}"

    def trigger_dataset_ingest(self, dataset_id: str) -> TriggerIngestResponse:
        """Wraps: POST /dataset/[id]/ingest"""
        response = self._post_json(url_path=f"dataset/{dataset_id}/ingest", content={})
        job_id = response.get("ingestionJobId")
        if not isinstance(job_id, str):
            raise ServerError(f"Malformed response: {response}")
        return TriggerIngestResponse(ingest_job_id=job_id)

    def create_dataset(
        self, name: str, embedding_column_name: Optional[str]
    ) -> CreateDatasetResponse:
        """Wraps: POST /dataset"""
        response = self._post_json(
            "dataset", dict(name=name, embeddingColumn=embedding_column_name)
        )
        dataset_id = response.get("datasetId")
        row_limit = response.get("rowLimit")

        if not (isinstance(dataset_id, str) and isinstance(row_limit, int)):
            raise ServerError(f"Malformed response: {response}")
        return CreateDatasetResponse(dataset_id=dataset_id, row_limit=row_limit)

    def upload_dataset_data(self, dataset_id: str, data: io.BufferedIOBase) -> None:
        """Wraps: PUT /dataset/[id]/source"""
        self._put_bytes(
            url_path=f"dataset/{dataset_id}/source",
            content=data,
            params={"format": "parquet"},
        )

    def _post_json(
        self, url_path: str, content: RequestJson, params: Optional[Dict[str, str]] = None
    ) -> ResponseJson:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        url = self._full_url(url_path)
        response = self._http_client.post(
            url, headers=headers, json=content, params=params
        )
        response_json = self._handle_response(response, expect_json=True)
        assert response_json is not None  # please mypy
        return response_json

    def _put_bytes(
        self,
        url_path: str,
        content: io.BufferedIOBase,
        params: Optional[Dict[str, str]] = None,
    ) -> None:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/octet-stream",
        }
        url = self._full_url(url_path)

        response = self._http_client.put(
            url,
            headers=headers,
            content=iter([b""]),  # send some dummy data to not consume the stream
            params=params,
            follow_redirects=False,
        )
        if response.next_request is None:
            logger.error("Response text:\n%s", response.text)
            raise ServerError(f"Expected redirect but got: {response.status_code}")

        response = self._http_client.put(
            response.next_request.url,
            headers=response.next_request.headers,
            content=_buffer_to_byte_iterable(content),
            follow_redirects=False,
        )
        self._handle_response(response, expect_json=False)

    def _handle_response(
        self, response: httpx.Response, expect_json: bool
    ) -> Optional[ResponseJson]:
        response_json: Optional[ResponseJson] = None
        error_message: Optional[str] = None
        request_kind = response.request.method
        url_path = response.request.url.path

        try:
            response_json = response.json()
        except Exception as e:
            # raise more appropriate error below, but log content here
            # to help debug.
            logger.debug("Response did not contain json. %s", e)

        if isinstance(response_json, dict):
            error_message = response_json.get("errorMessage")
            error_message = response_json.get("errorMessageDisplay") or error_message

        base_message = (
            error_message
            or f"Got '{response.status_code}' from {request_kind} to {url_path}"
        )
        status_code = response.status_code
        if status_code in (401, 403):
            logger.error("Authentication error response text:\n%s", response.text)
            raise AuthenticationError(f"You may not have access. {base_message}")
        if status_code == 404:
            raise NotFoundError(f"The resource may not exist. {base_message}")
        if 400 <= status_code < 500:
            raise BadRequestError(f"Bad Request. {base_message}")
        if status_code // 100 != 2:
            logger.error("Server error response text:\n%s", response.text)
            # Consider 100s, 300s, 500s to all be server errors because they are not
            # expected from the API.
            raise ServerError(f"Server error. {base_message}")

        if expect_json and not (
            isinstance(response_json, dict) and "data" in response_json
        ):
            # All our json APIs return dicts.
            logger.error("Malformed response text:\n%s", response.text)
            raise ServerError("Malformed server response.")

        if expect_json:
            return response_json["data"]  # type: ignore
        return None

    def _full_url(self, url_path: str) -> str:
        return f"{self._base_url}/{url_path}"


@lru_cache(maxsize=1)
def client() -> AirtrainClient:
    """Get the default Airtrain client. This is an internal API for advanced usage."""
    # Since we have used the lru_cache this will always return the same instance.
    return AirtrainClient()


def _find_api_key() -> Optional[str]:
    global _DEFAULT_API_KEY
    if _DEFAULT_API_KEY is not None:
        return _DEFAULT_API_KEY
    return os.environ.get(API_KEY_ENV_VAR)


def set_api_key(api_key: str) -> None:
    """Explicitly set the API key for the default client."""
    if api_key is None:
        raise AuthenticationError("Invalid API key; must not be None")
    global _DEFAULT_API_KEY
    _DEFAULT_API_KEY = api_key

    # In case the default client already exists.
    client()._api_key = api_key


def _buffer_to_byte_iterable(buffer: io.BufferedIOBase) -> Iterable[bytes]:
    while True:
        chunk = buffer.read(_BUFFER_CHUNK_SIZE)
        if not chunk:
            break
        yield chunk
