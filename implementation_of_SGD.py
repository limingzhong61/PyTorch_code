# Prepare the training set.
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# Initial guess of weight.
w = 1.0


# Define the model:
def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# Define the gradient function
def gradient(x, y):
    return 2 * x * (x * w - y)


print(' Predict (before training)', 4, forward(4))
# Update weight by every grad of sample of train set.
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad
        print(" t grad: ", x, y, grad)
        l = loss(x, y)
    print(' Epoch:', epoch, 'w=', w, 'loss=', l)
print(('Predict (after training)', 4, forward(4)))
