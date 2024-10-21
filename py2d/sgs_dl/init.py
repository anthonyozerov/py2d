from scipy.io import loadmat
from py2d.sgs_dl.cnn import CNN
# import ivy
# from torch2jax import t2j
from torch import nn

def initialize_model(filename):
    cnn = CNN.load_from_checkpoint(filename)
    cnn.eval()
    print(f'DL SGS model is on: {cnn.device}')
    # print(isinstance(cnn.cnn, nn.Module))
    # print(type(cnn.cnn))
    # model = nn.Sequential(nn.ModuleList([cnn.cnn.eval()]))
    # print(dir(cnn.cnn))
    # jaxmodel = t2j(model)
    # jaxmodel = ivy.transpile(model, source='torch', target='jax')
    return cnn.cnn

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
