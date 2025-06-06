def verify_full_config_cnn(config):

    assert config is not None, \
        "Full config dictionary must be provided to run CNN"
    assert 'library' in config, \
        "Library must be provided to run CNN"
    assert 'cnn_config' in config, \
        "CNN config must be provided to run CNN"
    assert 'input_stepnorm' in config, \
        "Input stepnorm must be provided to run CNN"
    if isinstance(config['cnn_config'], list):
        if len(config['cnn_config']) != 2:
            raise NotImplementedError("Only 2 CNN models are supported for multi-CNN model")
        
        # check that parameters for mixing CNN outputs are provided
        assert 'mixing_var' in config
        assert config['mixing_var'] == 'omega'
        assert 'mixing_quantile' in config
        assert 'mixing_quantile_mode' in config
        assert config['mixing_quantile_mode'] in ['firstcnnbelow', 'firstcnnabove']

        for cnn_config in config['cnn_config']:
            verify_cnn_config(cnn_config)
    else:
        verify_cnn_config(config['cnn_config'])

def verify_cnn_config(cnn_config):
    assert 'cnn_path' in cnn_config, \
        "CNN path must be provided to run CNN"
    assert 'norm_path' in cnn_config, \
        "Normalization path must be provided to run CNN"
    if 'residual' in cnn_config:
        if cnn_config['residual'] != 'gm4':
            raise NotImplementedError("Only GM4 residual is supported for CNN")