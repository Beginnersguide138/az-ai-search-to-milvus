"""設定ファイル読み込みのテスト。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from az_search_to_milvus.config import MigrationConfig


SAMPLE_YAML = """\
azure_search:
  endpoint: "https://test.search.windows.net"
  index_name: "my-index"
  api_key: "test-key"

milvus:
  uri: "http://milvus:19530"
  collection_name: "my_collection"

options:
  batch_size: 250
  dry_run: true
  enable_dynamic_field: true
  partition_key_field: "tenant_id"
  exclude_fields:
    - "internal"
  field_overrides:
    title:
      milvus_name: "doc_title"
      max_length: 8192
"""


class TestConfigFromYaml:
    def test_loads_all_sections(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(SAMPLE_YAML)
            f.flush()

            config = MigrationConfig.from_yaml(f.name)

        assert config.azure_search.endpoint == "https://test.search.windows.net"
        assert config.azure_search.index_name == "my-index"
        assert config.azure_search.api_key == "test-key"

        assert config.milvus.uri == "http://milvus:19530"
        assert config.milvus.collection_name == "my_collection"

        assert config.options.batch_size == 250
        assert config.options.dry_run is True
        assert config.options.enable_dynamic_field is True
        assert config.options.partition_key_field == "tenant_id"
        assert "internal" in config.options.exclude_fields
        assert "title" in config.options.field_overrides
        assert config.options.field_overrides["title"]["milvus_name"] == "doc_title"

    def test_defaults(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("azure_search:\n  endpoint: https://x.search.windows.net\n")
            f.flush()
            config = MigrationConfig.from_yaml(f.name)

        assert config.options.batch_size == 500
        assert config.options.dry_run is False
        assert config.milvus.uri == "http://localhost:19530"

    def test_to_dict_masks_secrets(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(SAMPLE_YAML)
            f.flush()
            config = MigrationConfig.from_yaml(f.name)

        d = config.to_dict()
        assert d["azure_search"]["api_key"] == "***"


class TestZillizConfig:
    def test_effective_uri_for_zilliz(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""\
azure_search:
  endpoint: https://x.search.windows.net
milvus:
  use_zilliz: true
  zilliz_endpoint: "https://xxx.zillizcloud.com"
  zilliz_api_key: "key123"
""")
            f.flush()
            config = MigrationConfig.from_yaml(f.name)

        assert config.milvus.effective_uri == "https://xxx.zillizcloud.com"
        assert config.milvus.effective_token == "key123"

    def test_effective_uri_for_self_hosted(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""\
azure_search:
  endpoint: https://x.search.windows.net
milvus:
  uri: "http://10.0.0.5:19530"
  token: "root:Milvus"
""")
            f.flush()
            config = MigrationConfig.from_yaml(f.name)

        assert config.milvus.effective_uri == "http://10.0.0.5:19530"
        assert config.milvus.effective_token == "root:Milvus"
