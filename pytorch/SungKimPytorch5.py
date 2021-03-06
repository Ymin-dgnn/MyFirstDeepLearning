
import torch
from torch.autograd import Variable

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class Model(torch.nn.Module):  # subclass of torch.nn.Module
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """
        In the Forward fuction we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the construcor 
        as well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()  # initialize
    loss.backward()
    optimizer.step()  # update

    # After training
    hour_var = torch.Tensor([[4.0]])
    print("predict (after training)", 4, model.forward(hour_var).data[0][0])

