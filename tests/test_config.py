from core.config import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.algorithm == "dqn"
    assert settings.state_dim == 26
    assert settings.action_dim == 4


def test_project_root():
    settings = Settings()
    assert settings.project_root.exists()
