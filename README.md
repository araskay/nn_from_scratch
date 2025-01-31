# Building a neural network from scratch using numpy

**This was originally published in a [medium post](https://medium.com/@arask/building-a-neural-network-from-scratch-using-numpy-e0d22667000d) as part of a teaching excercise**

There exist many great tools to build and train artificial neural networks — for example, pytorch and tensorflow are two of the commonly used libraries in python. These libraries are great and make it more convenient and simpler to build and train neural networks. But it is also important to have a solid understanding of the mathematical foundations, which may not be immediately visible when using these libraries. I recently built and trained a neural network from scratch only using numpy, as part of a teaching exercise. My goal in this post is to walk through this exercise.

I will assume the reader is familiar with artificial neural networks and will not go over the basics of perceptrons and layers — there are many great tutorials and blog posts that cover these concepts. Rather, I will focus on the mathematics and coding of the process of building and training the neural network using backpropagation and stochastic gradient descent.

## Dataset
For this exercise we use the familiar MNIST handwritten digits dataset. Data can be downloaded in a simple [csv format from kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).


![minist in csv](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*1U3Y5ZQnv9WCpREh)

## Preprocessing
1. read data from csv files
2. normalize images
3. one-hot encode labels

```
import numpy as np

# preprocessing
class PreProc:
    def __init__(self):
        pass
    
    def one_hot_encode(self, y, levels):
        res = np.zeros((len(y), levels))
        for i in range(len(y)):
            res[i, y[i]] = 1
        return res

    def normalize(self, x):
        return x / np.max(x)

    def read_csv(self, fname):
        data = np.loadtxt(fname, skiprows=1, delimiter=',')
        y = data[:, :1]
        x = data[:, 1:]
        return x, y

    def load_data(self, fname):
        x, y = self.read_csv(fname)
        x = self.normalize(x)
        y = np.int16(y)
        y = self.one_hot_encode(y, levels=10)

        x = np.expand_dims(x, axis = -1)
        y = np.expand_dims(y, axis = -1)
        return x, y

x_train, y_train = PreProc().load_data('mnist_train.csv')
x_test, y_test = PreProc().load_data('mnist_test.csv')
```

## Network structure
 ![Network structure](https://miro.medium.com/v2/resize:fit:1092/format:webp/1*V-TnsqOGU-4-l6ooYhELsg.png)
 
The following defines a NeuralNetwork class and initializes weights with random values.

```
class NeuralNetwork:
    def __init__(self, d_in, d1, d2, d_out, lr = 1e-3):
        self.d_in = d_in
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out
        self.lr = lr
        self.init_weights()
        
    def init_weights(self):
        self.w1 = np.random.randn(self.d1, self.d_in)
        self.b1 = np.random.randn(self.d1, 1)
        
        self.w2 = np.random.randn(self.d2, self.d1)
        self.b2 = np.random.randn(self.d2, 1)
        
        self.w3 = np.random.randn(self.d_out, self.d2)
        self.b3 = np.random.randn(self.d_out, 1)
```

## Forward calculations
Forward calculations are rather straight-forward. In each layer, the input from the previous layer is multiplied by the weights with an additive bias term, and passed to an activation function to form the output of that layer. In matrix notations:

$z = W.x + b$

$a = f(z)$

Where $W$ and $b$ denote the weights and biases for that layer. $x$ is the input to the layer and a is the output.

Assuming a dimensionality (i.e., number of nodes) of $d_i$ for layer $i$ and a dimensionality of $d_{(i-1)}$ for layer $i-1$, the wieght matrix, $W$, at layer $i$ will be of size $d_i \times d_{(i-1)}$ and biases, $b$, will be a vector of size $d_i$.

```
    def relu(self, x):
        return np.maximum(x, 0)

    def drelu(self, x):
        return np.diag(1.0 * (x > 0))

    def soft_max(self, x):
        x = x - np.max(x, axis=0)
        return np.exp(x) / np.sum(np.exp(x), axis=0)
  
    def forward(self, x, y):
        self.x = x
        self.y = y
        
        self.z1 = np.matmul(self.w1, self.x) + self.b1 # z1[d1 x 1] = w1[d1 x d_in] . x[d_in x 1] + b1[d1 x 1]
        self.a1 = np.apply_along_axis(self.relu, 1, self.z1) # a1[d1 x 1] = relu(z1[d1 x 1])
        
        self.z2 = np.matmul(self.w2, self.a1) + self.b2 # z2[d2 x 1] = w2[d2 x d1] . a1[d1 x 1] + b2[d2 x 1]
        self.a2 = np.apply_along_axis(self.relu, 1, self.z2) # a2[d2 x 1] = relu(z2[d2 x 1]) 
        
        self.z3 = np.matmul(self.w3, self.a2) + self.b3 # z3[d_out x 1] = w3[d_out x d2] . a2[d2 x 1] + b3[d_out x 1]
        self.out = np.apply_along_axis(self.soft_max, 1, self.z3) # out[d_out x 1] = soft_max(z3[d_out x 1])
```

## Backward calculation
Backward calculations involve the use of a gradient descent algorithm to update model parameters. The objective is to minimize a loss (aka cost) function. Therefore, we need to calculate the gradient (i.e., partial derivatives) of the loss function with respect to all the parameters.

Since artificial neural networks involve multiple sequential layers, to calculate the partial derivatives we use the chain rule. It turns out we can calculate the gradients in an efficient way by reusing the previously calculated partial derivatives in the “above” layers, rather than naively calculating the gradient with respect to each weight individually. This algorithm is know as backpropagation or backprop.

![forward and backward calculations](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*PjpeE-Tvs1t30vXUcDo3-g.png)
Forward calculations and the corresponding partial derivatives in each layer. The individual partial derivatives are used to calculate the gradients with respect to the weight using the chain rule. (See below for an example.)

![partial derivatives](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*qNX1zYthI2UYwkXjMlKI-A.png)
Partial derivatives of the loss function wit respect to the weights of the output layer. Note the common “reusable” part denoted by delta.

The following code implements the backward calculations using the backprop algorithm. Note the use of deltas to “reuse” previous calculations.

You may notice I have left the derivative of the output layer’s softmax activation for now. This was due to computational reasons as it was destabilizing backward calculation.

```
    def transpose(self, x):
        '''
        helper function to transpose last two dimensions of matrix
        '''
        return np.transpose(x, [0, 2, 1])

    def backward(self):
        # calculate gradients
        delta = 2*self.transpose(self.out - self.y) # [1 x d_out]
        self.dw3 = np.mean(
            np.matmul(self.transpose(delta), self.transpose(self.a2)),
            axis = 0
        ) # [d_out, d2] = [d_out x 1] . [1 x d2]
        self.db3 = np.mean(self.transpose(delta), axis = 0) # [d_out x 1] = [d_out x 1] . 1
        
        delta = np.matmul(
            np.matmul(delta, self.w3), # [1 x d2] = [1 x d_out] . [d_out x d2]
            np.squeeze(np.apply_along_axis(self.drelu, 1, self.z2)) # [d2 x d2]
        ) # [1 x d2]
        self.dw2 = np.mean(
            np.matmul(self.transpose(delta), self.transpose(self.a1)), # [d2 x d1] = [d2 x 1] . [1 x d1]
            axis = 0
        )
        self.db2 = np.mean(self.transpose(delta), axis = 0) # [d2 x 1]
        
        delta = np.matmul(
            np.matmul(delta, self.w2), # [1 x d2] . [d2 x d1]
            np.squeeze(np.apply_along_axis(self.drelu, 1, self.z1)) # [d1 x d1]
        ) # [1 x d1]
        self.dw1 = np.mean(
            np.matmul(self.transpose(delta), self.transpose(self.x)), # [d1 x d_in] = [d1 x 1] . [1 x d_in]
            axis = 0
        )
        self.db1 = np.mean(self.transpose(delta), axis = 0) # [d1 x 1]
        
        # update weights using the calculated gradients (gradient descent)
        self.w3 = self.w3 - self.lr * self.dw3
        self.b3 = self.b3 - self.lr * self.db3
        
        self.w2 = self.w2 - self.lr * self.dw2
        self.b2 = self.b2 - self.lr * self.db2
        
        self.w1 = self.w1 - self.lr * self.dw1
        self.b1 = self.b1 - self.lr * self.db1
```

## Training iterations

```
epochs = 20
batch_size = 1000
shuffle = True
lr = 1e-3

def shuffle(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x[idx,], y[idx,]

x_train, y_train = PreProc().load_data('mnist_train.csv')
x_test, y_test = PreProc().load_data('mnist_test.csv')

nn = NeuralNetwork(x_train.shape[1], 256, 128, y_train.shape[1], lr=lr)

l = []
acc = []

for i in range(epochs):
    loss = 0
    accuracy = 0
    if shuffle:
        x_train, y_train = shuffle(x_train, y_train)
    for batch in range(x_train.shape[0] // batch_size):
        x = x_train[batch*batch_size: (batch+1)*batch_size,]
        y = y_train[batch*batch_size: (batch+1)*batch_size,]
        nn.forward(x, y)
        loss += np.mean((nn.out - nn.y) ** 2)
        accuracy += np.mean(np.argmax(nn.out, axis=1) == np.argmax(nn.y, axis=1))
        nn.backward()
    loss = loss / (x_train.shape[0] // batch_size)
    l.append(loss)
    accuracy = accuracy / (x_train.shape[0] // batch_size)
    acc.append(accuracy)
    print('Epoch {epoch}: loss = {loss}, accuracy = {accuracy}'.format(epoch=i, loss=loss, accuracy=accuracy))
```

## Results
With 20 epochs, I could reach a training accuracy of 85%. Figures below show the training accuracy and loss over these iterations. It seems the accuracy can be further improved by allowing more iterations. However, in the absence of any regularization, I was a bit concerned about overfitting and therefore limited the iterations to 20.

![training accuracy](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*K8Wx1r-46x5dq1CDYOvnIQ.png)
Training accuracy over 20 epochs

![training loss](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bIDJlxJvZrdaqJm4FphxHw.png)
Training loss over 20 epochs

This resulted in a comparable test accuracy of 85%, demonstrating no strong overfitting.

```
nn.forward(x_test, y_test)

test_accuracy = np.mean(np.argmax(nn.out, axis=1) == np.argmax(nn.y, axis=1))
print('test accuracy = {:.3f}'.format(test_accuracy))
```

## Concluding remarks
My objective in this post was to show the fundamental calculations in an artificial neural network, and in particular the calculation of the gradients, using a simple example. Admitedly, I did not spend much time tuning the hyperparameters (e.g., learning rate, batch size, etc.). With this, we achieved an accuracy of 85%. In reality we almost never do this and rather use available packages such as pytorch. Moreover, there exists a wide variety of tecniques and algorithms to improve the performance of neural networks, including optimizing network structure and hyperparameters, using regularization techniques, etc. — just to name a few.
