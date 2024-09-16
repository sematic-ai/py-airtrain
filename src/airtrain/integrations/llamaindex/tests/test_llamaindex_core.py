import sys

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from airtrain.core import DatasetMetadata
from airtrain.integrations.llamaindex.core import upload_from_llama_nodes
from tests.fixtures import MockAirtrainClient, mock_client  # noqa: F401


def test_upload_from_nodes(mock_client: MockAirtrainClient):  # noqa: F811
    nodes = [TextNode(text="hello"), TextNode(text="world")]
    nodes[0].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
        node_id=nodes[1].node_id
    )
    nodes[1].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
        node_id=nodes[0].node_id
    )
    name = "Foo dataset"
    result = upload_from_llama_nodes(nodes, name=name)
    assert isinstance(result, DatasetMetadata)
    assert result.size == len(nodes)
    assert result.name == name
    fake_dataset = mock_client.get_fake_dataset(result.id)
    assert fake_dataset.name == name
    table = fake_dataset.ingested
    assert table is not None
    assert table.shape[0] == len(nodes)
    assert all(isinstance(id_, str) for id_ in table["id_"].to_pylist())
    assert table["text"].to_pylist() == ["hello", "world"]

    if sys.version_info >= (3, 11):
        if "mimetype" in table.column_names:
            assert table["mimetype"].to_pylist() == ["text/plain", "text/plain"]
        assert table["relationships.NodeRelationship.NEXT.class_name"].to_pylist() == [
            "RelatedNodeInfo",
            None,
        ]
        assert table["relationships.NodeRelationship.PREVIOUS.node_id"].to_pylist() == [
            None,
            nodes[0].node_id,
        ]
        assert table["relationships.NodeRelationship.NEXT.node_id"].to_pylist() == [
            nodes[1].node_id,
            None,
        ]
