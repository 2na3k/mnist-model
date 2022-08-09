from torch import nn, Tensor 

class Layer(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, tensor: Tensor, inference: bool = False) -> Tensor:
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, 
                 input_size: int,
                 neurons: int,
                 dropout: float = 1.0,
                 activation: nn.Module = None) -> None:

        super().__init__()

        self.linear = nn.Linear(input_size, neurons)
        self.activation = activation

        if dropout < 1.0:
            self.dropout = nn.Dropout(1 - dropout)
        
        def forward(self, x: Tensor, inference: bool = False) -> Tensor:
            if inference:
                self.apply(inference_mode)
            
            #multiple with weight, also plus the bias
            self.linear(x)

            if self.activation:
                x = self.activation(x) #pass the x to the activation function and return again to variable x
            
            if hasattr(self, "dropout"):
                x = self.dropout(x)
            
            return x
