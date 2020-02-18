import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.Tensor([[0.], [0.], [1.], [1.]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # linear output

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = Model()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, tensor.item(0))
    #print(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()  # initialize
    loss.backward()
    optimizer.step()  # update

    # After training
hour_var = torch.Tensor([[1.0]])
print("predict 1 hour", 1.0, model.forward(hour_var).data[0][0] > 0.5)
hour_var = torch.Tensor([[7.0]])
print("predict 7 hour", 7.0, model.forward(hour_var).data[0][0] > 0.5)
