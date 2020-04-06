---
layout: post
title: "A Neural Network With Range-v3"
date: 2020-02-28
author: Bradley Bauer
---
```
****** Intro  ******
// Present ranges, views, and actions.
// Show a tensor implementation
class Tensor {
  vector<double> data;
  vector<int> shape;
  auto view() { ... }
  // ...
};


****** Data stuff ******
// Show views, actions, and algorithms useful for data manipulation

dataFileContents = open("mnist.data");
labelsFileContents = open("mnist.labels");

data = toView(dataFileContents) | chunk(28*28) | tensors({28*28,1}) | take(MNIST_SZ) | to<vector>
labels = toView(labelsFileContents) | take(MNIST_SZ) | to<vector>

data |= shuffle(std::mt19937{12345})
labels |= shuffle(std::mt19937{12345})

dataMean = mean(data)
dataStd = mean(diff(dataMean) | sqr).sqrt()

X = toView(data) | diff(dataMean) | div(dataStd)
Y = toView(labels)

****** Model and training ******
// Show Linear and activation layer implementations and training loop

template<int indim, int outdim> class Linear {
    W = Tensor{{outdim,indim}}
    b = Tensor{{outdim}})
    x = Tensor{{indim}}
    Tensor fwd(Tensor) { ... }
    Tensor bwd(Tensor) { ... }
};
// Layer composition is not implemented with ranges.
model=Model<Linear<28*28,20>,
            ReLU<20>,
            Linear<20,10>,
            Softmax<10>>{};
obj = NLLoss{};

// Where is begin/end ? Talk about sentinels.
start = clock::now()
train = zip(X, Y)
      | cycle
      | take_while([&](auto){return clock::now()-start < 5mins;})
for (auto[x,y] : train) {
    yhat = model(x)
    loss = obj(yhat,y)
    model.bwd(obj.bwd())
}


****** Convolutional model ******
// This is where things get interesting.
// I will implement convolution on a multichannel tensor using views::sliding, views::stride, ...
// I have code that creates a view of tensors from sliding a window over a source tensor.
// Given this view of tensors i can use the range-ified inner_product algorithm and the filter weights to compute the outputs.
// A nice [visualization](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1) similar to  what I'm doing.


****** Conclusion ******
// Talk about compiler error messages, compile times, what's in std::ranges, ...
```
