import pytest

from primus.core.config.yaml_loader import _resolve_env_in_string, parse_yaml


class TestYamlLoader:

    def test_parse_yaml_basic(self, tmp_path):
        config_content = """
        a: 1
        b: test
        """
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(config_content)

        result = parse_yaml(str(cfg_file))
        assert result == {"a": 1, "b": "test"}

    def test_parse_yaml_empty(self, tmp_path):
        """Test parsing an empty YAML file raises ValueError."""
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")

        with pytest.raises(ValueError, match="is empty or invalid"):
            parse_yaml(str(cfg_file))

    def test_parse_yaml_extends(self, tmp_path):
        """Test recursive extends merging."""
        base_content = """
        base_val: 1
        override_val: 1
        """
        base_file = tmp_path / "base.yaml"
        base_file.write_text(base_content)

        # Child extends Base
        child_content = """
        extends:
          - base.yaml
        override_val: 2
        child_val: 3
        """
        child_file = tmp_path / "child.yaml"
        child_file.write_text(child_content)

        result = parse_yaml(str(child_file))
        assert result["base_val"] == 1
        assert result["override_val"] == 2
        assert result["child_val"] == 3
        assert "extends" not in result  # Should be removed

    def test_parse_yaml_multi_extends(self, tmp_path):
        """Test extending multiple files."""
        base1 = tmp_path / "base1.yaml"
        base1.write_text("a: 1\nb: 1")

        base2 = tmp_path / "base2.yaml"
        base2.write_text("b: 2\nc: 2")

        main = tmp_path / "main.yaml"
        main.write_text(
            """
        extends:
          - base1.yaml
          - base2.yaml
        c: 3
        """
        )

        result = parse_yaml(str(main))
        assert result["a"] == 1
        assert result["b"] == 2  # base2 overrides base1
        assert result["c"] == 3  # main overrides base2

    def test_env_resolution(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEST_VAR", "123")
        monkeypatch.setenv("TEST_STR", "hello")

        content = """
        val_int: ${TEST_VAR}
        val_str: ${TEST_STR}
        val_default: ${MISSING:default}
        val_nested:
           inner: ${TEST_VAR}
        """
        cfg_file = tmp_path / "env.yaml"
        cfg_file.write_text(content)

        result = parse_yaml(str(cfg_file))
        assert result["val_int"] == 123
        assert result["val_str"] == "hello"
        assert result["val_default"] == "default"
        assert result["val_nested"]["inner"] == 123

    def test_env_resolution_missing_error(self, tmp_path):
        content = "val: ${NON_EXISTENT_VAR}"
        cfg_file = tmp_path / "error.yaml"
        cfg_file.write_text(content)

        # Use exact string match (pytest handles regex escaping automatically for simple strings if needed,
        # but explicit check is better)
        expected_msg = "Environment variable 'NON_EXISTENT_VAR' is required but not set."
        with pytest.raises(ValueError, match=expected_msg):
            parse_yaml(str(cfg_file))

    def test_resolve_env_in_string_helpers(self, monkeypatch):
        monkeypatch.setenv("NUM", "42")
        monkeypatch.setenv("FLOAT", "3.14")

        assert _resolve_env_in_string("${NUM}") == 42
        assert _resolve_env_in_string("${FLOAT}") == 3.14
        assert _resolve_env_in_string("prefix_${NUM}") == "prefix_42"

    def test_env_resolution_bool_strings(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FLAG_TRUE", "true")
        monkeypatch.setenv("FLAG_FALSE", "false")
        monkeypatch.setenv("FLAG_UPPER", "False")
        monkeypatch.delenv("UNSET_BOOL", raising=False)

        assert _resolve_env_in_string("${FLAG_TRUE}") is True
        assert _resolve_env_in_string("${FLAG_FALSE}") is False
        assert _resolve_env_in_string("${FLAG_UPPER}") is False
        assert _resolve_env_in_string("${UNSET_BOOL:false}") is False
        assert _resolve_env_in_string("${UNSET_BOOL:true}") is True

        content = """
        native_false: false
        env_false_default: ${UNSET_BOOL:false}
        env_false_set: ${FLAG_FALSE}
        env_true_set: ${FLAG_TRUE}
        env_zero: ${UNSET_ZERO:0}
        """
        monkeypatch.delenv("UNSET_ZERO", raising=False)
        cfg_file = tmp_path / "bool_env.yaml"
        cfg_file.write_text(content)

        result = parse_yaml(str(cfg_file))
        assert result["native_false"] is False
        assert result["env_false_default"] is False
        assert result["env_false_set"] is False
        assert result["env_true_set"] is True
        assert result["env_zero"] == 0
        assert isinstance(result["env_zero"], int)
