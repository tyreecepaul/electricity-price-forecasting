### Main Training Directory
----------------------------
```
.
├── layers # all layers including modules and models to be used in final model
├── final_model # final model class composed of layers and modules
├── experiment_config # main training executable to run final model with any feature engineering scripts
├── README.md 
```


## Model Layout (Image for Paper)
-----------------------------------


## Layers
-----------------------------------
Using Pytorch we can easily create modules or layers for the final model and implementation.
Within the layers directory there should only be your designed layer in the form of a child class of nn.Module.
As an example heres chapos MLP:
```
class NeuralNetwork(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.GELU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)
```
All training and evaluation should be done in Experiment Config, preferably run within the final model as a test with the new module.

## Final Model
-----------------------------------
Final model will utilize the layers designed by us and constructed together in a our chosen manner.

## Experiment outline
-----------------------------------

- Our experiment parameters, setup, features selected etc
