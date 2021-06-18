# Grok Log of asynchronous 
Started by reading an Astral Codex Ten poist referencing the paper

In general for machine learning frameworks, we want to work with vectors in the ~1000 dim
so may need matrices of about 1000 x 1000
### First implementation 
First started with plain arrays, cause wanted to use static size checks
with new const generics, to make sure the matrix multiplication went correctly

Realized I didn't want to implement matrix multiplication from scratch
so went with NAlgebra.
### 05-06/2021 
Tried to get a basic implementation going with all of the threading, but 
ran into repeated stack overflow.  Matrices are probably too big
[see this link](https://discourse.nphysics.org/t/force-the-matrix-to-be-heap-allocated/361)
So need to implement some kind of heap allocated matrices.  Went for allocating one row at a time
