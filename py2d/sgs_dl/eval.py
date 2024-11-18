import torch
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

def evaluate_model(model, model_norm, input_data_dict, input_stepnorm=False):
    input_data = []
    for name, data in input_data_dict.items():
        data = np.array(data)
        if input_stepnorm:
            #input_data.append(torch.from_dlpack(asdlpack((data - data.mean())/data.std())))
            input_data.append(torch.Tensor((data - data.mean())/data.std()))
        else:
            #input_data.append(torch.from_dlpack(asdlpack((data - model_norm[f'mean_{name}'])/model_norm[f'sdev_{name}'])))
            input_data.append(torch.Tensor((data - model_norm[f'mean_{name}'])/model_norm[f'sdev_{name}']))
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.inference_mode():
            input_data = torch.stack(input_data).to(torch.float32).to(device)
            output = jnp.array(model(input_data)[0].cpu().numpy())

        output = output * model_norm['sdev_pi'] + model_norm['mean_pi']

    return output
