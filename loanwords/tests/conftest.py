def pytest_configure(config):
    for m in (
        "slow: downloads a model or dataset",
        "network: needs network access",
        "gpu: needs a GPU",
    ):
        config.addinivalue_line("markers", m)
