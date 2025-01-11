import numpy as np
import jax.numpy as jnp


def get_converter(whichlib):
    if whichlib == "pytorch":
        import torch
        converter = torch.Tensor

    elif whichlib == "tensorflow":
        import tensorflow as tf
        converter = tf.convert_to_tensor

    elif whichlib == "onnx":
        converter = np.array

    else:
        raise ValueError(f"Unsupported library {whichlib}")

    return converter


def inference_pytorch(model, input_data):
    import torch

    # disable gradient calculations
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.inference_mode():
            input_data = torch.stack(input_data).to(torch.float32).to(device)
            output = jnp.array(model(input_data)[0].cpu().numpy())

    return output


def inference_tensorflow(model, input_data):
    import tensorflow as tf

    input_data = tf.cast(tf.stack(input_data), tf.float32)
    output = model(input_data)[0].numpy()

    return output


def inference_onnx(model, input_data):
    input_data = np.stack(input_data).astype(np.float32)[np.newaxis, :, :, :]
    # currently it is [1, CHANNELS, 128, 128]
    # need to convert to [1, 128, 128, CHANNELS]
    input_data = np.transpose(input_data, (0, 2, 3, 1))

    output = model.run(None, {"conv2d_input": input_data})[0]
    output = np.squeeze(output)

    return output


def evaluate_model(model, model_norm, input_data_dict, whichlib, input_stepnorm=False):
    input_data = []

    converter = get_converter(whichlib)

    for name, data in input_data_dict.items():
        data = np.array(data)

        if input_stepnorm:
            data_norm = (data - data.mean()) / data.std()
        else:
            data_norm = (data - model_norm[f"mean_{name}"]) / model_norm[f"sdev_{name}"]

        input_data.append(converter(data_norm))

    inferencers = {
        "pytorch": inference_pytorch,
        "tensorflow": inference_tensorflow,
        "onnx": inference_onnx,
    }

    if whichlib not in inferencers.keys():
        raise ValueError(f"Unsupported library {whichlib}")

    output = inferencers[whichlib](model, input_data)

    output = output * model_norm["sdev_pi"] + model_norm["mean_pi"]

    return output
