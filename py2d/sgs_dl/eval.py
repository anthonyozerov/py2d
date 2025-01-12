import numpy as np
import jax.numpy as jnp


def get_converter(whichlib):
    """
    Given a library name, get a function or class which will convert
    numpy arrays to the appropriate library's tensor type.
    """
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


def inference_pytorch(model, input_data, reorder=None):
    import torch

    assert reorder is None, "Axis reordering not implemented for PyTorch"

    # disable gradient calculations
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.inference_mode():
            input_data = torch.stack(input_data).to(torch.float32).to(device)
            output = jnp.array(model(input_data)[0].cpu().numpy())

    return output


def inference_tensorflow(model, input_data, reorder=None):
    import tensorflow as tf

    assert reorder is None, "Axis reordering not implemented for Tensorflow"

    input_data = tf.cast(tf.stack(input_data), tf.float32)
    output = model(input_data)[0].numpy()

    return output


def inference_onnx(model, input_data, reorder=None):
    input_data = np.stack(input_data).astype(np.float32)[np.newaxis, :, :, :]
    # currently it is [1, CHANNELS, NX, NX]

    # reorder the axes of the input data if needed
    if reorder is not None:
        input_data = np.moveaxis(input_data, [0, 1, 2, 3], reorder)

    output = model.run(None, {model.get_inputs()[0].name: input_data})[0]

    # reorder the axes of the output data if needed
    if reorder is not None:
        output = np.moveaxis(output, reorder, [0, 1, 2, 3])

    output = np.squeeze(output)

    return output


def evaluate_model(model, model_norm, input_data_dict, whichlib, input_stepnorm=False, reorder=None):
    input_data = []

    # get the appropriate np.array -> tensor converter
    converter = get_converter(whichlib)

    # construct a list of input tensors, with one tensor for each input channel.
    # input channels are normalized.
    for name, data in input_data_dict.items():
        data = np.array(data)

        if input_stepnorm:
            data_norm = (data - data.mean()) / data.std()
        else:
            data_norm = (data - model_norm[f"mean_{name}"]) / model_norm[f"sdev_{name}"]

        input_data.append(converter(data_norm))

    # dict of inference functions
    inferencers = {
        "pytorch": inference_pytorch,
        "tensorflow": inference_tensorflow,
        "onnx": inference_onnx,
    }

    if whichlib not in inferencers.keys():
        raise ValueError(f"Unsupported library {whichlib}")

    # apply the inference function for the appropriate library
    output = inferencers[whichlib](model, input_data, reorder)
    # denormalize the output
    output = output * model_norm["sdev_pi"] + model_norm["mean_pi"]

    # ensure the output shape is the same as the input shape (i.e. [NX, NX])
    assert output.shape == data.shape

    return output
