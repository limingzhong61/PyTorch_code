import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.Tensor([1.0])
# If **autograd** **mechanics** are required, the element variable **requires_grad** of**Tensor** has to be set to
# **True** .
w.requires_grad = True


# Define the linear model:
def forward(x):
    return x * w


# Define the loss function:
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print(("predict (before training)", 4, forward(4).item()))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # Forward, compute the loss
        l = loss(x, y)
        # Backward, compute grad for Tensor whose requires_grad set to True
        l.backward()
        print("\tgrad:", x, y, w.grad.item())
        # The grad is utilized to update weight.
        w.data = w.data - 0.01 * w.grad.data
        # NOTICE:
        # The grad computed by .backward() will be accumulated .
        # So after update, remember set the grad to ZERO
        w.grad.data.zero_()
    print("progress:", epoch, l.item())

print(("predict (after training)", 4, forward(4).item()))
