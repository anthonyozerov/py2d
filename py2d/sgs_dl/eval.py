import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

def evaluate_model(model, model_norm, input_data_dict, whichlib, input_stepnorm=False):
    input_data = []

    if whichlib == 'pytorch':
        import torch
        converter = torch.Tensor
    elif whichlib == 'tensorflow':
        import tensorflow as tf
        converter = tf.convert_to_tensor
    elif whichlib == 'onnx':
        converter = np.array

    for name, data in input_data_dict.items():
        data = np.array(data)

        if input_stepnorm:
            input_data.append(converter((data - data.mean())/data.std()))
        else:
            input_data.append(converter((data - model_norm[f'mean_{name}'])/model_norm[f'sdev_{name}']))
    if whichlib == 'pytorch':
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with torch.inference_mode():
                input_data = torch.stack(input_data).to(torch.float32).to(device)
                output = jnp.array(model(input_data)[0].cpu().numpy())
    elif whichlib == 'tensorflow':
        input_data = tf.cast(tf.stack(input_data), tf.float32)
        output = model(input_data)[0].numpy()

    # PREFERRED METHOD (more portable)
    elif whichlib == 'onnx':
        input_data = np.stack(input_data).astype(np.float32)[np.newaxis, :, :, :]

        # currently it is [1, CHANNELS, 128, 128]
        # need to convert to [1, 128, 128, CHANNELS]

        input_data = np.transpose(input_data, (0, 2, 3, 1))
        output = model.run(None, {'conv2d_input': input_data})[0]
        output = np.squeeze(output)

    output = output * model_norm['sdev_pi'] + model_norm['mean_pi']

    return output
