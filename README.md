The Problem
---------------

This repo solves a blind signal separation problem on the [LFW(Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/) dataset.The problem goes as follows:

Suppose an image is a superimposition of 5 faces in the dataset with equal weights. Who are these 5 people?

Method
---------------

The idea behind the code is fairly simple. Natural images usually have quite sparse gradients, which, when images are superimposed, makes it less probable that the information get overriden and lost. So we can use gradients as a criterion to match images layer by layer along with a searching algorithm, in this case the A* algorithm(which now has degenerated into a Dijkstra algorithm since it performs better).

[This paper](https://pdfs.semanticscholar.org/ac81/becda896b635bc9c014d365bd8acdce01877.pdf) is also based on the sparsity of image gradients, but aimed for a more complex problem. Check it out for more details.

When I tried to implement it the biggest problem I found was the definition and implementation correctness of loss measure. An inferior definition chokes the algorithm and a slight bug would significantly affect the performance.

Dependencies
---------------

* python >= 3.5
* numpy
* scikit-image
* opencv3, opencv-python
* scipy