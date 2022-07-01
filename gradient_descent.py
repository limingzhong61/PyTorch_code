# Prepare the training set.
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# Initial guess of weight.
w = 1.0


# Define the model:
def forward(x):
    return x * w


# Define the cost function
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# Define the gradient function
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print(' Predict (before training)', 4, forward(4))
# Do the update
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print(' Epoch:', epoch, 'w=', w, 'loss=', cost_val)
print(('Predict (after training)', 4, forward(4)))
