import numpy as np

#  we r creating 2 feature linear regression model with single bias ofc

weights = np.random.rand(2, 1) #  2*1
bias = 0  # 1*1, or 1 sample, 1 feature  /// learning ofcourse there must a parent list which contains all samples, each sample is represented by list cz we need to also showcase what feature it is

def predict(X):  #  X must be a 1 x 2 array
    y = X @ weights + bias  #  it will be 1*1, #  before i blundered by not using convention and instead used wx + b, which leads to a lot of confusion
    return y  #  1 x 1


def gradient_descent(X, true_val):
    global weights, bias

    learning_rate = 0.001
    predicted_val = predict(X=X)
    w_gradient = 2 * X * (predicted_val - true_val)  #  no loops required as its matrix... new learning understood how powerful matrix are, and also famous gradient formula is 2 * xi * ( pred - true )  automatically multiplied in matrix multiplication
    # b_gradient = 2 * (predicted_val - true_val)
    mean_weight_grad = np.array([np.mean(w_gradient, axis=0)])  # 1x2
    weights -= learning_rate * mean_weight_grad.T
    print(weights)
    # print(b_gradient)




# print(weights)
# print(bias)
#  y = 3x + 5x
sample = np.array([[2,1], [4,3], [5, 2], [7, 6]])
output1 = np.array([[11], [27], [25], [51]])

for epoch in range(4000):
    gradient_descent(X=sample, true_val=output1)

print("predicting")
print(predict(X=[[8, 5]]))



# sample2 = np.array([[1, 2], [4,5]])
# output2 = np.array()
# print(predict(X=sample2).shape)