import torch.nn as nn
from transformer.py import Forecaster
from pipeline.model import Model


def choose_loss(loss):
    if loss == 'mse':
        return nn.MSELoss()
    elif loss == 'l1':
        return nn.SmoothL1Loss()

class NEMformer(Model):
    model_name = "Transformer"

    def __init__(self, 
        d_model=512, 
        nhead=8, 
        num_layers=6, 
        dim_feedforward=2048,
        epochs=100,
        batch_size=32,
        loss = 'mse',
        lr_rate=0.001
        )->None:
        self.model = Forecaster(
            d_model =d_model, 
            nhead = nhead, 
            num_layers = num_layers, 
            dim_feedforward = dim_feedforward
        ) # need random state?


    def _train_epoch(self, dataloader, loss_fn, optimizer):
    
        for batch, (x,y) in enumerate(dataloader):
            #get prediction and loss
            pred = self.model.predict(x)
            loss = loss_fn(pred, y)

            #backprop
            loss.backward()
            optimizer.step() #update parameters
            optimizer.zero_grad() # zero gradients to avoid double counting

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")

    
        

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:

        # get training data later
        loss = choose_loss(self.loss);
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr_rate)

        for epoch in range(self.epochs):
            self.train_epoch(self, x_train=x_train)



    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
                
        

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        return self.model.eval(x_test)