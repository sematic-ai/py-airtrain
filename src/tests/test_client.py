import io
import itertools
import os
from unittest.mock import MagicMock

import pytest

from airtrain.client import (
    API_KEY_ENV_VAR,
    AirtrainClient,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    ServerError,
    client,
    set_api_key,
    _buffer_to_byte_iterable,
)
from tests.utils import environment_variables


def test_set_api_key():
    client.cache_clear()

    # make sure no existing env var is clouding the test.
    assert os.environ.get(API_KEY_ENV_VAR) is None
    with pytest.raises(AuthenticationError):
        client()

    with environment_variables({API_KEY_ENV_VAR: "foo"}):
        c = client()
    assert isinstance(c, AirtrainClient)
    assert c._api_key == "foo"

    client.cache_clear()
    with pytest.raises(AuthenticationError):
        client()

    set_api_key("bar")
    c = client()
    assert isinstance(c, AirtrainClient)
    assert c._api_key == "bar"

    set_api_key("baz")
    assert c._api_key == "baz"


def test_default_client():
    client.cache_clear()
    set_api_key("secret")

    assert client() is client()


def test_buffer_to_byte_iterable():
    hex_digits = "0123456789abcdef"

    # some non-repeating byte data that should be larger than the byte buffer
    original_bytes = bytes.fromhex(
        "".join(
            "".join(p)
            for p in itertools.islice(itertools.permutations(hex_digits, 16), 2**12)
        )
    )

    buffer_iter = _buffer_to_byte_iterable(io.BytesIO(original_bytes))

    counter = itertools.count()
    read_bytes = b"".join(buffer for buffer, _ in zip(buffer_iter, counter))
    assert read_bytes == original_bytes

    # ensure we had to use more than 1 buffer.
    assert next(counter) > 2


def test_handle_response():
    c = AirtrainClient(api_key="secret")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": 42}
    assert c._handle_response(mock_response, expect_json=True) == 42

    mock_response.reset_mock()
    mock_response.status_code = 200
    mock_response.json.return_value = 42
    with pytest.raises(ServerError):
        c._handle_response(mock_response, expect_json=True)
    assert c._handle_response(mock_response, expect_json=False) is None

    mock_response.status_code = 404
    mock_response.json.return_value = {
        "data": 42,
        "errorMessage": "These are not the droids you're looking for",
    }
    with pytest.raises(NotFoundError, match=r".*not the droids.*"):
        c._handle_response(mock_response, expect_json=True)

    mock_response.status_code = 401
    with pytest.raises(AuthenticationError):
        c._handle_response(mock_response, expect_json=True)

    mock_response.status_code = 400
    with pytest.raises(BadRequestError):
        c._handle_response(mock_response, expect_json=True)

    mock_response.status_code = 500
    with pytest.raises(ServerError):
        c._handle_response(mock_response, expect_json=True)

    mock_response.reset_mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("A problem has be to your computer")
    with pytest.raises(ServerError):
        c._handle_response(mock_response, expect_json=True)
    c._handle_response(mock_response, expect_json=False)
