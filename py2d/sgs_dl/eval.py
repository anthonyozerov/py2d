import torch
import numpy as np
import jax.numpy as jnp

def evaluate_model(model, model_norm, input_data_dict):
    input_data = []
    for name, data in input_data_dict.items():
        input_data.append((data - model_norm[f'mean_{name}'])/model_norm[f'sdev_{name}'])
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.inference_mode():
            input_data = torch.Tensor(np.array(input_data)).to(device)
        # input_data = input_data.unsqueeze(0)

        output = jnp.array(model(input_data)[0][0])

        output = output * model_norm['sdev_pi'] + model_norm['mean_pi']

    return output
