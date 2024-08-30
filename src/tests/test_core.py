from airtrain.core import DatasetMetadata, upload_from_dicts


def test_upload_from_dicts():
    result = upload_from_dicts([{"foo": 42}, {"foo": 43}], name="Foo dataset")
    assert isinstance(result, DatasetMetadata)
