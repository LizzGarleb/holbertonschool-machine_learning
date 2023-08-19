# Linear Algebra

> Linear algebra is the branch of mathematics concerning linear equations such as linear maps and their representations in vector spaces and through matrices. It is central to almost all areas of mathematics. In Machine Learning having the practical knowledge of maths in programming speeds up the desing of models and understanding real problems and how to solve them by a critical analysis perspective. This project covers the first part of Linear Algebra implementation with Python for Machine Learning.

## Repository Overview
```
Holberton School Machine Learning
|
└── math
    ├── **linear_algebra**
    |   └── main
    │
    ├── plotting
    │   └── main
    │
    └── probability
        └── main
```

## Learning Objective
At the end of this project I was able to solve these conceptual questions:

- What is a vector?
- What is a matrix?
- What is a transpose?
- What is the shape of a matrix?
- What is an axis?
- What is a slice?
- How do you slice a vector/matrix?
- What are element-wise operations?
- How do you concatenate vectors/matrices?
- What is the dot product?
- What is matrix multiplication?
- What is Numpy?
- What is parallelization and why is it important?
- What is broadcasting?

## Tasks
0. Complete the following source code (found below):

    * `arr1` should be the first two numbers of `arr`
    * `arr2` should be the last five numbers of `arr`
    * `arr3` should be the 2nd through 6th numbers of `arr`
    * You are not allowed to use any loops or conditional statements
    * Your program should be exactly 8 lines

1. Complete the following source code (found below):

    * `the_middle` should be a 2D matrix containing the 3rd and 4th columns of `matrix`
    * You are not allowed to use any conditional statements
    * You are only allowed to use one `for` loop
    * Your program should be exactly 6 lines

2. Write a function `def matrix_shape(matrix):` that calculates the shape of a matrix:

    * You can assume all elements in the same dimension are of the same type/shape
    * The shape should be returned as a list of integers

3. Write a function `def matrix_transpose(matrix):` that returns the transpose of a 2D matrix, `matrix`:

    * You must return a new matrix
    * You can assume that `matrix` is never empty
    * You can assume all elements in the same dimension are of the same type/shape

4. Write a function `def add_arrays(arr1, arr2):` that adds two arrays element-wise:

    * You can assume that `arr1` and `arr2` are lists of ints/floats
    * You must return a new list
    * If `arr1` and `arr2` are not the same shape, return `None`

5. Write a function `def add_matrices2D(mat1, mat2):` that adds two matrices element-wise:

    * You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same type/shape
    * You must return a new matrix
    * If `mat1` and `mat2` are not the same shape, return `None`

6. Write a function def cat_arrays(arr1, arr2): that concatenates two arrays:

    You can assume that arr1 and arr2 are lists of ints/floats
    You must return a new list

7. Write a function `def cat_matrices2D(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:

    * You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same type/shape
    * You must return a new matrix
    * If the two matrices cannot be concatenated, return `None`

8. Write a function `def mat_mul(mat1, mat2):` that performs matrix multiplication:

    * You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same type/shape
    * You must return a new matrix
    * If the two matrices cannot be multiplied, return `None`

9. Complete the following source code (found below):

    * `mat1` should be the middle two rows of `matrix`
    * `mat2` should be the middle two columns of `matrix`
    * `mat3` should be the bottom-right, square, 3x3 matrix of `matrix`
    * You are not allowed to use any loops or conditional statements
    * Your program should be exactly 10 lines

10. Write a function `def np_shape(matrix):` that calculates the shape of a `numpy.ndarray`:

    * You are not allowed to use any loops or conditional statements
    * You are not allowed to use `try/except` statements
    * The shape should be returned as a tuple of integers

11. Write a function `def np_transpose(matrix):` that transposes `matrix`:

    * You can assume that `matrix` can be interpreted as a `numpy.ndarray`
    * You are not allowed to use any loops or conditional statements
    * You must return a new `numpy.ndarray`

12. Write a function `def np_elementwise(mat1, mat2):` that performs element-wise addition, subtraction, multiplication, and division:

    * You can assume that `mat1` and `mat2` can be interpreted as `numpy.ndarray`s
    * You should return a tuple containing the element-wise sum, difference, product, and quotient, respectively
    * You are not allowed to use any loops or conditional statements
    * You can assume that `mat1` and `mat2` are never empty

13. Write a function `def np_cat(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:

    * You can assume that `mat1` and `mat2` can be interpreted as `numpy.ndarray`s
    * You must return a new `numpy.ndarray`
    * You are not allowed to use any loops or conditional statements
    * You may use: `import numpy as np`
    * You can assume that `mat1` and `mat2` are never empty

14. Write a function `def np_matmul(mat1, mat2):` that performs matrix multiplication:

    * You can assume that `mat1` and `mat2` are `numpy.ndarray`s
    * You are not allowed to use any loops or conditional statements
    * You may use: `import numpy as np`
    * You can assume that `mat1` and `mat2` are never empty
