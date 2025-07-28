import pytest
from src.model_parser import ModelicaParameterParser


def test_get_available_parameters():
    """Test parameter parsing for a valid model."""
    package_path = "./example/package.mo"
    model_name = "example.Cycle"
    parser = ModelicaParameterParser(
        package_path, model_name, default_params=['i_iss.T'])

    params = parser.get_available_parameters()
    assert isinstance(params, list), "Expected a list of parameters"
    assert 'i_iss.T' in params, "Expected default parameter in list"
    print(f"Parsed parameters: {params}")


def test_invalid_model():
    """Test parameter parsing for an invalid model."""
    package_path = "./example/package.mo"
    model_name = "Invalid.Model"
    parser = ModelicaParameterParser(
        package_path, model_name, default_params=['i_iss.T'])

    params = parser.get_available_parameters()
    assert params == [
        'i_iss.T'], "Expected default parameters for invalid model"


def test_invalid_package():
    """Test parameter parsing for an invalid package path."""
    package_path = "invalid/path/package.mo"
    model_name = "example.Cycle"
    parser = ModelicaParameterParser(
        package_path, model_name, default_params=['i_iss.T'])

    params = parser.get_available_parameters()
    assert params == [
        'i_iss.T'], "Expected default parameters for invalid package"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
