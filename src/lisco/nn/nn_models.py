import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeedforwardNNConfig:
    """Hyperparameter config for FeedforwardNN. Defaults match FeedforwardNN.__init__.

    Note: n_in and n_out are excluded — they are always derived from NLP
    dimensions at runtime and are not static hyperparameters.
    """

    n_hidden_layers: int = 2
    n_neurons: int = 500
    act_fn: str = "relu"
    output_act_fn: str = "linear"
    nn_dtype: torch.dtype = torch.float32


def get_activation_layer(act_fn):
    if act_fn == 'relu':
        return torch.nn.ReLU()
    elif act_fn == 'tanh':
        return torch.nn.Tanh()
    elif act_fn == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif act_fn == 'sigmoid':
        return torch.nn.Sigmoid()
    elif act_fn == "gelu":
        return torch.nn.GELU(approximate='tanh')
    else:
        raise ValueError("Activation function not implemented.")


class FeedforwardNN(torch.nn.Module):
    """Feedforward Neural Network model.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden_layers (int): Number of hidden layers.
        n_neurons (int): Number of neurons in each hidden layer.
        act_fn (str): Activation function.
        output_act_fn (str): Output activation function.
    """
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons=500, act_fn='relu', output_act_fn='linear',nn_dtype=torch.float32):
        super().__init__()
        if n_hidden_layers <= 0:
            raise ValueError("Number of hidden layers must be > 0.")
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden_layers = n_hidden_layers
        self.n_layers = n_hidden_layers+1
        self.n_neurons = n_neurons
        self.nn_dtype = nn_dtype
        self.act_fn = act_fn
        self.output_act_fn = output_act_fn
        self._autocast_dtype = None  # set by setup_fast_train; not persisted
        self.layers = torch.nn.ModuleList()

        # add hidden layers
        for i in range(self.n_hidden_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(n_in, n_neurons,dtype=self.nn_dtype))
                self.layers.append(get_activation_layer(act_fn))
            else:
                self.layers.append(torch.nn.Linear(n_neurons, n_neurons,dtype=self.nn_dtype))
                self.layers.append(get_activation_layer(act_fn))

        # add output layer
        self.layers.append(torch.nn.Linear(n_neurons, n_out,dtype=self.nn_dtype))
        if output_act_fn != 'linear':
            self.layers.append(get_activation_layer(output_act_fn))

    def forward(self, x):
        x = x.to(self.nn_dtype)
        if self._autocast_dtype is None:
            for layer in self.layers:
                x = layer(x)
            return x
        with torch.autocast("cuda", dtype=self._autocast_dtype):
            for layer in self.layers:
                x = layer(x)
        return x.to(self.nn_dtype)

    def to_dict(self) -> dict:
        """Serialize architecture hyperparameters to a JSON-safe plain dict."""
        return {
            "n_in":            self.n_in,
            "n_out":           self.n_out,
            "n_hidden_layers": self.n_hidden_layers,
            "n_neurons":       self.n_neurons,
            "act_fn":          self.act_fn,
            "output_act_fn":   self.output_act_fn,
            "nn_dtype":        str(self.nn_dtype).split(".")[-1],
        }

    def save_weights(self, save_dir, file_name="weights.pt"):
        save_pth = Path(save_dir, file_name)
        torch.save(self.state_dict(), save_pth)

    def load_weights(self, load_dir, file_name="weights.pt", map_location=None):
        load_pth = Path(load_dir, file_name)
        if map_location is None:
            map_location = next(self.parameters()).device
        self.load_state_dict(torch.load(load_pth, map_location=map_location, weights_only=True))


def setup_fast_train(module, tf32=False, bf16=False):
    """Enable GPU training levers on every FeedforwardNN inside `module`.

    tf32: enable TF32 matmuls. NOTE: this is a PROCESS-GLOBAL flag — it affects
        every module in the process and stays on for the process lifetime.
    bf16: run each FeedforwardNN forward under bfloat16 autocast. Weights stay
        at nn_dtype (fp32); autocast rounds a copy per op and is not persisted.
    """
    if tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    for m in module.modules():
        if isinstance(m, FeedforwardNN):
            m._autocast_dtype = torch.bfloat16 if bf16 else None
