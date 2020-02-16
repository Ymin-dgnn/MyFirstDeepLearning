import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_1 = Variable(torch.Tensor([1.0]), require_grad=True)  # Any random value
w_2 = Variable(torch.Tensor([1.0]), require_grad=True)
# require_grad : I need to cumpute gradient


def forward(x):
    return x * x * w_2 + x * w_1


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


print("pridict (before training)", 4, forward(4).data[0])

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w_1.grad.data[0])
        w_1.data = w_1.data - 0.01 * w_1.grad.data

        # Manually zero the gradients after updating weights
        w_1.grad.data.zero_()

        print("\tgrad: ", x_val, y_val, w_2.grad.data[0])
        w_2.data = w_2.data - 0.01 * w_2.grad.data

        # Manually zero the gradients after updating weights
        w_2.grad.data.zero_()

    print("progress:", epoch, l.data[0])

print("pridict (after training)", 4, forward(4).data[0])
