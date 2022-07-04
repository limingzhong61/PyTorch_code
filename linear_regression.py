# Import necessary library to draw the graph.
import numpy as np
import matplotlib.pyplot as plt

# Prepare the train set.
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# Define the model:
def forward(x):
    return x * w


# Define the loss function:
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# List w_list save the weights 𝝎
# List mse_list save the cost values of each 𝝎
w_list = []
mse_list = []
# Compute cost value at [0.0, 0.1, 0.2, … , 4.0]

for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    # For each sample in train set, the loss
    # function values were computed.
    # ATTENTION:
    # Value of cost function is the sum of loss
    # function.
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
    print('\t', x_val, y_val, y_pred_val, loss_val)
    print((' MSE=', l_sum / 3))

    # Save  𝝎 and correspondence MSE
    w_list.append(w)
    mse_list.append(l_sum / 3)

# Draw the graph
plt.plot(w_list, mse_list)
plt.ylabel('Lost')
plt.xlabel('w')
plt.show()
