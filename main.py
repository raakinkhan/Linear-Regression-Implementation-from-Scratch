#  its a linear regression model made from scratch so that i can learn bts whats happening and understand core concepts for future models


#  linear regression is just a line y = mx + c


# taking example y = 2x for this model to be trained upon
x_vals = [1, 2, 3, 4]
y_vals = [2, 4, 6, 8]

# taking example y = x + 2 for bias checks
x_vals_b = [1, 2, 3, 4]
y_vals_b = [3, 4, 5, 6]

#  taking combinations y = 2x + 2 for final single linear regression model
x_vals_wb = [1, 2, 3, 4]  # wb = weights and bias
y_vals_wb = [4, 6, 8, 10]

y = 0 # some output
w = 1 # some weight for simplicity 1
b = 1 # just for now for simplicity

def predict(X):
    y = w*X + b
    return y

def loss(X, true_val):  #  new learning - this has no use, it only shows how much loss is to the user, it doesnt update the weigths, what updates is the 'mean' gradient, uncovered by differentiating loss func wrt weight
    predicted = predict(X)
    mse_loss = (predicted - true_val) ** 2
    return mse_loss

def gradient(X, true_value):
    #  we already know -> loss depends on prediction, prediction depends on weigths so loss depends on weights
    gradient = 2 * X * (w * X + b - true_value)  #  differentiating loss func wrt w
    bias_gradient = 2 * (w * X + b - true_value)  #  differentiating loss func wrt b
    return [gradient, bias_gradient]

def average_gradient(X_list, true_value_list):  # as per my learning we use avg gradient not single gradient cz, it would lead to overfit to one sample only
    gradient_list = []
    gradient_bias_list = []
    for _, x_val in enumerate(X_list):
        grad = gradient(X=x_val, true_value=true_value_list[_])

        gradient_list.append(grad[0])  #  basically all weights wala gradients
        gradient_bias_list.append(grad[1])  # basically all bias wala gradients

    gradient_avg = sum(gradient_list)/len(gradient_list)
    bias_gradient_avg = sum(gradient_bias_list) / len(gradient_bias_list)
    # print(f"this is the weight gradient list = {gradient_list}")
    print(f"this is the average weight gradient  = {gradient_avg}")

    # print(f"this is the bias gradient list = {gradient_bias_list}")
    print(f"this is the average bias gradient  = {bias_gradient_avg}")
    return [gradient_avg, bias_gradient_avg]

def gradient_descent(X_list, true_value_list, learning_rate=0.01):
    global w, b
    avg_gradient = average_gradient(X_list=X_list, true_value_list=true_value_list)
    w -= learning_rate * avg_gradient[0]  # 0 for weights
    b -= learning_rate * avg_gradient[1]  #
    print(f"new weights are {w}")
    print(f"new bias are {b}")








# print(predict(X=3))
# print(loss(X=2, true_val=5))
# print(gradient(X=1, true_value=10))

for epoch in range(20000):  # idk what epoch meant before but it just means number of repeatations
    gradient_descent(X_list=x_vals_wb, true_value_list=y_vals_wb)

print(f"y = {w}x + {b}")
print(predict(X=6))