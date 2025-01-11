from scipy.io import loadmat
import os


def initialize_pytorch_model(filename):
    from py2d.sgs_dl.cnn import CNN
    import torch

    # force the model to not use TensorFloat32 cores,
    # which are fast but reduce precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnnallow_tf32 = False

    cnn = CNN.load_from_checkpoint(filename)
    cnn.eval()
    print(f"DL SGS model is on: {cnn.device}")

    return cnn.cnn


def initialize_tensorflow_model(filename):
    from tensorflow.keras.models import load_model

    model = load_model(filename)
    print(model.summary())
    return model


def initialize_onnx_model(filename):
    import onnxruntime as rt

    print(rt.get_available_providers())
    if "CUDAExecutionProvider" in rt.get_available_providers():
        # force the model to not use TensorFloat32 cores,
        # which are fast but reduce precision.
        providers = [("CUDAExecutionProvider", {"use_tf32": 0})]
    else:
        providers = ["CPUExecutionProvider"]
    sess = rt.InferenceSession(filename, providers=providers)
    print(sess.get_providers())


def initialize_model(filename, whichlib):
    initializers = {
        "pytorch": initialize_pytorch_model,
        "tensorflow": initialize_tensorflow_model,
        "onnx": initialize_onnx_model,
    }

    if whichlib not in initializers.keys():
        raise ValueError(f"Unsupported library {whichlib}")

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    return initializers[whichlib](filename)


def initialize_model_norm(filename):
    """Initialize the normalization dictionary from a file."""
    norm_data = loadmat(filename)
    norm = {}

    norm["mean_psi"] = norm_data["MEAN_IP"]
    norm["sdev_psi"] = norm_data["SDEV_IP"]

    norm["mean_omega"] = norm_data["MEAN_IW"]
    norm["sdev_omega"] = norm_data["SDEV_IW"]

    norm["mean_pi"] = norm_data["MEAN_IPI"]
    norm["sdev_pi"] = norm_data["SDEV_IPI"]

    return norm
