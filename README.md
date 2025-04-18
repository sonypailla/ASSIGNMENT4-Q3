# ASSIGNMENT4-Q3
Scaled Dot-Product Attention Implementation
# Scaled Dot-Product Attention Implementation

## Student Details:
- **Name**: Sony Pailla  
- **Course**: CS5720 Neural Network and Deep Learning  
- **University**: University of Central Missouri  
- **Semester**: Spring 2025  
- **GitHub Username**: [sonypailla](https://github.com/sonypailla)

## Description:
This project implements the **Scaled Dot-Product Attention** mechanism, a core component of the Transformer architecture used in NLP and deep learning. It follows these steps:

1. Computes the dot product of the **Query (Q)** and the transpose of **Key (K)**.
2. Scales the result by dividing by the square root of the key dimension.
3. Applies the softmax function to obtain attention weights.
4. Multiplies the attention weights by the **Value (V)** matrix to get the final output.

## Libraries Used:
- numpy
- scipy.special (for softmax)

## Input:
```python
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
