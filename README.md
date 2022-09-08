# gradual
Yet another autodiff. 


I have just finished the deep learning specialization at Coursera, and I am very grateful for Andrew Ng clearness to explain a lot of new concepts to me.

During the first courses, we learn how to build some basic neural network models from scratch. Implementing forward and backward propagation steps. For me, it was very enriching to learn a new algorithm that utilized dynamic programming methods (cache) to speed up computation.

At that moment, it was relieving to receive formulas to calculate gradients during backpropagation steps. Because I was already learning a lot of things at the same time and I had already forgotten most of my calculus classes.

But in the end, I ended with the impression that I couldnt implement very custom architecture from scratch just because I cannot break down the differentiation steps. Every time I used those deep learning frameworks (PyTorch/TensorFlow) I ended with the record that I cannot really comprehend what was going on.

HN had already presented me a random blog post on auto differentiation, that lived my see later bookmarks for a long time. But after spending some hours slowly reading it, thanks to my lack of math fluency, my idea about breaking down differentiation steps didn't change.

Behold, a video by Andrej Karpathy popped up on my screen where he teaches how to build micrograd, an auto differentiation library with circa 200 lines of code. Wow, I was amazed by this class, which opened some doors to continue learning this theme.

To finish, this repository is the place where I will organize some auto differentiation experiments (inspired by the karpathy/micrograd).
