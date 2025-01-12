def verify_cnn_config(config):

    assert config is not None, \
        "Full config dictionary must be provided to run CNN"
    assert 'cnn_path' in config, \
        "CNN path must be provided to run CNN"
    assert 'norm_path' in config, \
        "Normalization path must be provided to run CNN"
    assert 'library' in config, \
        "Library must be provided to run CNN"
    assert 'cnn_config' in config, \
        "CNN config must be provided to run CNN"
    assert 'input_stepnorm' in config, \
        "Input stepnorm must be provided to run CNN"
    if 'resid' in config['cnn_config']:
        assert config['cnn_config']['resid'] == 'gm4', \
            "Only GM4 residual is supported for CNN"