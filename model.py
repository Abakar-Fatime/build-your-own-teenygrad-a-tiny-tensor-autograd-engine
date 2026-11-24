"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
from typing import Union, Tuple, List

def prod(shape: Union[Tuple[int, ...], List[int]]) -> int:
    """
    Calculate the product of all elements in a shape tuple or list.
    An empty shape represents a scalar, returning 1.

    Args:
        shape: A tuple or list of integers representing tensor dimensions.

    Returns:
        The total flat size of the tensor as a single integer.
    """
    result = 1
    for dim in shape:
        result *= dim
    return result

# Example validations matching your framework specifications:
if __name__ == "__main__":
    print(prod((2, 3, 4)))  # Outputs: 24 (A 3D tensor of shape 2x3x4 has 24 total elements)
    print(prod([5]))        # Outputs: 5  (A 1D tensor with 5 elements)
    print(prod(()))         # Outputs: 1  (A 0D scalar tensor has 1 element)

# Step 2 - argsort
def argsort(values):
    """
    Return the indices that would sort the input sequence in ascending order.
    Maintains a stable sorting order for equivalent values.

    Args:
        values: A sequence (list or tuple) of sortable elements.

    Returns:
        A list of integers representing the sorted index permutation.
    """
    # Create a list of indices [0, 1, 2, ...] and sort them 
    # using the corresponding value from the sequence as the sorting key
    return sorted(range(len(values)), key=lambda i: values[i])

# Example validations matching your framework specifications:
if __name__ == "__main__":
    print(argsort([3, 1, 2]))   # Outputs: [1, 2, 0] -> values[1]=1, values[2]=2, values[0]=3
    print(argsort([10, 10, 5])) # Outputs: [2, 0, 1] -> stable sort preserves index 0 before index 1

# Step 3 - make_op_enums
from enum import Enum

def make_op_enums():
    """
    Create and return the four core enum classes required by the lazy backend 
    to categorize computational graph operations.
    
    Returns:
        tuple: (UnaryOps, BinaryOps, ReduceOps, MovementOps)
    """
    class UnaryOps(Enum):
        NEG = 1
        RELU = 2
        LOG = 3
        EXP = 4
        SQRT = 5
        SIGMOID = 6

    class BinaryOps(Enum):
        ADD = 1
        SUB = 2
        MUL = 3
        DIV = 4
        CMPLT = 5
        MAX = 6

    class ReduceOps(Enum):
        SUM = 1
        MAX = 2

    class MovementOps(Enum):
        RESHAPE = 1
        EXPAND = 2
        PERMUTE = 3

    return UnaryOps, BinaryOps, ReduceOps, MovementOps

# Example validation matching your framework specifications:
if __name__ == "__main__":
    U, B, R, M = make_op_enums()
    print([m.name for m in U])  # Outputs: ['NEG', 'RELU', 'LOG', 'EXP', 'SQRT', 'SIGMOID']
    print(R.SUM.name)            # Outputs: 'SUM'

# Step 4 - LazyBuffer
import numpy as np

class LazyBuffer:
    def __init__(self, data):
        """
        The foundational lazy buffer that wraps a numpy array asset.
        Converts the incoming sequence, caches shape/dtype metadata,
        and hides the underlying array inside a private reference attribute.
        """
        # Convert any input data sequence or raw scalar into a standard numpy array
        self._np = np.array(data)
        
        # Cache metadata fields immediately to allow quick structural introspection
        # without invoking any raw, heavy matrix evaluations down the graph pipeline
        self.shape = self._np.shape
        self.dtype = self._np.dtype

# Example verification matching your framework specifications:
if __name__ == "__main__":
    buf = LazyBuffer([[1.0, 2.0], [3.0, 4.0]])
    print(buf.shape)          # Outputs: (2, 2)
    print(buf.dtype)          # Outputs: dtype('float64')
    print(buf._np.tolist())   # Outputs: [[1.0, 2.0], [3.0, 4.0]]

# Step 5 - lazybuffer_const (not yet solved)
# TODO: implement

# Step 6 - rand (not yet solved)
# TODO: implement

# Step 7 - lazybuffer_unary_e (not yet solved)
# TODO: implement

# Step 8 - lazybuffer_binary_e (not yet solved)
# TODO: implement

# Step 9 - lazybuffer_r (not yet solved)
# TODO: implement

# Step 10 - lazybuffer_reshape (not yet solved)
# TODO: implement

# Step 11 - lazybuffer_expand (not yet solved)
# TODO: implement

# Step 12 - lazybuffer_permute (not yet solved)
# TODO: implement

# Step 13 - Function (not yet solved)
# TODO: implement

# Step 14 - function_forward_backward_stubs (not yet solved)
# TODO: implement

# Step 15 - apply (not yet solved)
# TODO: implement

# Step 16 - Neg (not yet solved)
# TODO: implement

# Step 17 - Relu (not yet solved)
# TODO: implement

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

