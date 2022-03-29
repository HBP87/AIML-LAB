import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

X = X / np.amax(X, axis=0)
y = y / 100




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivatives_sigmoid(x):
    return x * (1 - x)


epoch = 7000
lr = 0.1

inputlayer_neurons = 2
hiddenlayer_neurons = 3
output_neurons = 1

wh = np.random.randn(inputlayer_neurons, hiddenlayer_neurons)
wout = np.random.randn(hiddenlayer_neurons, output_neurons)

bh = np.random.randn(1, hiddenlayer_neurons)
bout = np.random.randn(1, output_neurons)

for i in range(epoch):
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hiddenlayer = sigmoid(hinp)

    outinp1 = np.dot(hiddenlayer, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)

    # Back Propagation
    output_error = y - output
    output_delta = output_error * derivatives_sigmoid(output)

    # hiddenlayer_error = output_delta.dot(wout.T)
    hiddenlayer_error = np.dot(output_delta, wout.T)
    hiddenlayer_delta = hiddenlayer_error * derivatives_sigmoid(hiddenlayer)

    wh += X.T.dot(hiddenlayer_delta) * lr
    wout += hiddenlayer.T.dot(output_delta) * lr

print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n", output)
