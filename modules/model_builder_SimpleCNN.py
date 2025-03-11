import torch
from torch import nn 

class SimpleCNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units* 2, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # with torch.no_grad():
        #     sample_input = torch.randn(1, input_shape, 64, 64)  # Assume input size 64x64
        #     sample_output = self.conv_block_1(sample_input)
        #     sample_output = self.conv_block_2(sample_output)
        #     self.flattened_size = sample_output.view(1, -1).shape[1] 

        self.classifier = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(self.flattened_size, 128),
            # nn.ReLU(),
            # # nn.Dropout(0.5),
            # nn.Linear(128, output_shape)
            nn.Flatten(),
          
            nn.Linear(in_features=hidden_units*16*16, 
                      out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        print(x.shape)
        x = self.conv_block_2(x)
        print(x.shape)
        x = self.classifier(x)
        print(x.shape)
        return x