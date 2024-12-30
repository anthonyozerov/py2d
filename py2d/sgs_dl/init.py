from scipy.io import loadmat

def initialize_model(filename, whichlib):
    if whichlib == 'pytorch':
        # assume it is a pytorch model
        from py2d.sgs_dl.cnn import CNN
        import torch


        # force the model to not use TensorFloat32 cores,
        # which are fast but reduce precision.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnnallow_tf32 = False

        cnn = CNN.load_from_checkpoint(filename)
        cnn.eval()
        print(f'DL SGS model is on: {cnn.device}')

        return cnn.cnn
    elif whichlib == 'tensorflow':
        # assume it is a tensorflow model
        from tensorflow.keras.models import load_model
        model = load_model(filename)
        model.summary()
        return model
    elif whichlib == 'onnx':
        import onnxruntime as rt
        print(rt.get_available_providers())
        if 'CUDAExecutionProvider' in rt.get_available_providers():
            providers = [("CUDAExecutionProvider", {"use_tf32": 0})]
        else:
            providers = ["CPUExecutionProvider"]
        sess = rt.InferenceSession(filename, providers=providers)
        print(sess.get_providers())

        return ort_session
    else:
        raise ValueError(f'Unsupported library {whichlib}')

def initialize_model_norm(filename):
    """Initialize the normalization dictionary from a file."""
    norm_data = loadmat(filename)
    norm = {}

    norm['mean_psi'] = norm_data['MEAN_IP']
    norm['sdev_psi'] = norm_data['SDEV_IP']

    norm['mean_omega'] = norm_data['MEAN_IW']
    norm['sdev_omega'] = norm_data['SDEV_IW']

    norm['mean_pi'] = norm_data['MEAN_IPI']
    norm['sdev_pi'] = norm_data['SDEV_IPI']

    return norm
