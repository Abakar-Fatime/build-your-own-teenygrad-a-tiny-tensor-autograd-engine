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

# Step 5 - lazybuffer_const
import numpy as np

def const(value, shape):
    """
    Create a new LazyBuffer of the given shape filled entirely with a constant value.
    The underlying numpy array is explicitly enforced as a float32 data type.

    Args:
        value: The constant scalar value (float or int) to fill the buffer with.
        shape: A tuple or list of integers representing the target dimensions.

    Returns:
        LazyBuffer: A new buffer instance containing the materialized array.
    """
    # Create a full array of the requested shape filled with the constant scalar,
    # ensuring the dtype is strictly locked to float32
    arr = np.full(shape, value, dtype=np.float32)
    return LazyBuffer(arr)

# Bind the standalone function as a static method onto the LazyBuffer class
LazyBuffer.const = staticmethod(const)

# Example verification matching your framework specifications:
if __name__ == "__main__":
    # Test creating a 2x3 tensor of ones
    ones_buf = LazyBuffer.const(1.0, (2, 3))
    print(ones_buf.shape)          # Outputs: (2, 3)
    print(ones_buf.dtype)          # Outputs: dtype('float32')
    print(ones_buf._np.tolist())   # Outputs: [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

# Step 6 - rand
import numpy as np

def rand(shape, seed=None):
    """
    Produce a LazyBuffer filled with uniformly distributed random float32 values 
    in the half-open interval [0.0, 1.0) with the requested shape.

    Args:
        shape: A tuple or list of integers representing the target tensor dimensions.
        seed: An optional integer to seed the random number generator for reproducibility.

    Returns:
        LazyBuffer: A new buffer instance containing the randomized float32 array.
    """
    # Initialize a localized random generator instance (reproducible if seed is provided)
    rng = np.random.default_rng(seed)
    
    # Generate uniform values and explicitly cast the buffer to float32
    random_array = rng.random(size=shape).astype(np.float32)
    
    return LazyBuffer(random_array)

# Example verification matching your framework specifications:
if __name__ == "__main__":
    # Test reproducibility
    b1 = rand((2, 2), seed=42)
    b2 = rand((2, 2), seed=42)
    b3 = rand((2, 2), seed=100)
    
    print(b1.shape)  # Outputs: (2, 2)
    print(b1.dtype)  # Outputs: dtype('float32')
    
    # Check that matching seeds produce identical arrays
    print(np.array_equal(b1._np, b2._np))  # Outputs: True
    # Check that different seeds produce distinct arrays
    print(np.array_equal(b1._np, b3._np))  # Outputs: False
    
    # Confirm boundaries [0, 1)
    print(bool((b1._np >= 0).all() and (b1._np < 1).all()))  # Outputs: True

# Step 7 - lazybuffer_unary_e
import numpy as np

def e(self, op: 'UnaryOps') -> 'LazyBuffer':
    """
    Apply a unary element-wise mathematical operation onto the underlying 
    array and return a brand new LazyBuffer containing the evaluated result.

    Args:
        op: A member of the UnaryOps enum (NEG, RELU, LOG, EXP, SQRT, SIGMOID).

    Returns:
        LazyBuffer: A new buffer instance containing the transformed float32 data.
    """
    # Fetch the operation enum classes from your make_op_enums definition
    # Assuming UnaryOps is globally accessible or imported into your backend file
    
    if op.name == 'NEG':
        res = -self._np
        
    elif op.name == 'RELU':
        # ReLU(x) = max(0, x)
        res = np.maximum(0.0, self._np)
        
    elif op.name == 'LOG':
        res = np.log(self._np)
        
    elif op.name == 'EXP':
        res = np.exp(self._np)
        
    elif op.name == 'SQRT':
        res = np.sqrt(self._np)
        
    elif op.name == 'SIGMOID':
        # Sigmoid(x) = 1 / (1 + exp(-x))
        res = 1.0 / (1.0 + np.exp(-self._np))
        
    else:
        raise ValueError(f"Unsupported Unary operation: {op}")
        
    # Always return a clean, new LazyBuffer wrapping the result array
    # cast to float32 to preserve numerical dtype stability across the backend
    return LazyBuffer(res.astype(np.float32))

# Bind the standalone function as an instance method onto the LazyBuffer class
LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
import numpy as np

def lazybuffer_binary_e(self: 'LazyBuffer', op: 'BinaryOps', other: 'LazyBuffer') -> 'LazyBuffer':
    """
    Apply a two-operand elementwise operation between two LazyBuffer instances
    and return a brand new evaluated LazyBuffer without mutating inputs.

    Args:
        self: The first operand LazyBuffer
        op: A member of BinaryOps enum (ADD, SUB, MUL, DIV, CMPLT, MAX)
        other: The second operand LazyBuffer

    Returns:
        LazyBuffer: A new buffer instance containing the float32 result.
    """
    # Extract raw arrays from both source buffers
    x = self._np
    y = other._np
    
    if op.name == 'ADD':
        res = x + y
    elif op.name == 'SUB':
        res = x - y
    elif op.name == 'MUL':
        res = x * y
    elif op.name == 'DIV':
        res = x / y
    elif op.name == 'CMPLT':
        # Less-than comparison: convert boolean array to numerical 1.0 or 0.0 mask
        res = (x < y).astype(np.float32)
    elif op.name == 'MAX':
        # Element-wise maximum
        res = np.maximum(x, y)
    else:
        raise ValueError(f"Unsupported Binary operation: {op}")
        
    # Return a new LazyBuffer wrapping the fresh data array
    return LazyBuffer(res.astype(np.float32))

# Step 9 - lazybuffer_r
import numpy as np

def r(self: 'LazyBuffer', op: 'ReduceOps', axis=None) -> 'LazyBuffer':
    """
    Perform a reduction operation (SUM or MAX) along a given axis or axes,
    keeping the collapsed dimensions as size 1 to preserve tensor rank.

    Args:
        self: The source LazyBuffer instance.
        op: A member of the ReduceOps enum (SUM or MAX).
        axis: An int, a tuple of ints, or None to reduce over the entire buffer.

    Returns:
        LazyBuffer: A new buffer instance containing the reduced float32 data.
    """
    x = self._np
    
    if op.name == 'SUM':
        res = np.sum(x, axis=axis, keepdims=True)
    elif op.name == 'MAX':
        res = np.max(x, axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unsupported Reduce operation: {op}")
        
    # Return a new LazyBuffer wrapping the fresh data array forced to float32
    return LazyBuffer(res.astype(np.float32))

# Dynamically bind the standalone utility onto the LazyBuffer class if needed
LazyBuffer.r = r

# Step 10 - lazybuffer_reshape
def reshape(self, arg):
    new_shape = list(arg)
    if new_shape.count(-1) > 1:
        raise ValueError("only one dimension can be -1")
    if -1 in new_shape:
        known = 1
        for d in new_shape:
            if d != -1:
                known *= d
        if known == 0 or self._np.size % known != 0:
            raise ValueError(f"cannot reshape {self._np.shape} into {tuple(arg)}")
        new_shape[new_shape.index(-1)] = self._np.size // known
    if int(np.prod(new_shape)) != self._np.size:
        raise ValueError(f"cannot reshape {self._np.shape} into {tuple(arg)}")
    return LazyBuffer(self._np.reshape(tuple(new_shape)))

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    new_shape = tuple(new_shape)
    if len(new_shape) != self._np.ndim:
        raise ValueError(f"expand needs same rank: {self._np.shape} -> {new_shape}")
    for old, new in zip(self._np.shape, new_shape):
        if old != new and old != 1:
            raise ValueError(f"cannot expand axis of size {old} to {new}")
    return LazyBuffer(np.broadcast_to(self._np, new_shape).copy())

LazyBuffer.expand = expand   # so buf.expand((2, 3)) works too

# Step 12 - lazybuffer_permute
def permute(self, order):
    order = tuple(order)
    if sorted(order) != list(range(self._np.ndim)):
        raise ValueError(f"invalid permutation {order} for {self._np.ndim}-D buffer")
    return LazyBuffer(np.transpose(self._np, order).copy())

LazyBuffer.permute = permute   # so a.permute((1, 0)) works too

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        self.needs_input_grad = [
            tensor.requires_grad for tensor in tensors
        ]

        self.requires_grad = (
            None if any(flag is None for flag in self.needs_input_grad)
            else any(self.needs_input_grad)
        )

        if self.requires_grad:
            self.parents = tuple(
                tensor for tensor in tensors if tensor.requires_grad
            )

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses of Function must implement forward.")

    def backward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses of Function must implement backward.")

    Function.forward = forward
    Function.backward = backward

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    ctx = cls(*tensors)
    result_buf = ctx.forward(*[t.lazydata for t in tensors], **kwargs)
    out = Tensor(result_buf, requires_grad=ctx.requires_grad)
    if any(t.requires_grad for t in tensors):
        out._ctx = ctx
    return out


# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
class Neg(Function):
    def forward(self, x):
        return LazyBuffer(-x._np)

    def backward(self, grad_output):
        return LazyBuffer(-grad_output._np)

# Step 17 - Relu
class Relu(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.RELU)
        return self.ret

    def backward(self, grad_output):
        _, BinaryOps, _, _ = make_op_enums()
        zero = LazyBuffer.const(0, self.ret._np.shape)
        mask = lazybuffer_binary_e(zero, BinaryOps.CMPLT, self.ret)
        return lazybuffer_binary_e(mask, BinaryOps.MUL, grad_output)

# Step 18 - Log
import numpy as np


class Log(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.x = x
        out = x.e(UnaryOps.LOG)
        if hasattr(x, "_np") and hasattr(out, "_np") and out._np.dtype != x._np.dtype:
            out._np = out._np.astype(x._np.dtype)
        return out

    def backward(self, grad_output):
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(grad_output, BinaryOps.DIV, self.x)

# Step 19 - Exp
import numpy as np
from enum import Enum

# ---- step 3: op enums ----
def make_op_enums():
    UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "EXP", "LOG", "SIN", "SQRT", "RECIPROCAL"])
    BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ", "MAX"])
    ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
    MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "EXPAND", "PAD", "SHRINK", "STRIDE"])
    return UnaryOps, BinaryOps, ReduceOps, MovementOps

# ---- step 4: LazyBuffer (keeps the input dtype — no float32 cast) ----
class LazyBuffer:
    def __init__(self, np_array):
        self._np = np.asarray(np_array)

    @property
    def shape(self):
        return self._np.shape

    @property
    def dtype(self):
        return self._np.dtype

    def e(self, op):
        fns = {
            "NOOP": lambda x: x.copy(),
            "NEG": np.negative,
            "EXP": np.exp,
            "LOG": np.log,
            "SIN": np.sin,
            "SQRT": np.sqrt,
            "RECIPROCAL": lambda x: 1.0 / x,
        }
        return LazyBuffer(fns[op.name](self._np))

# ---- step 8: binary elementwise ----
def lazybuffer_binary_e(self, op, other):
    fns = {
        "ADD": np.add,
        "SUB": np.subtract,
        "MUL": np.multiply,
        "DIV": np.divide,
        "POW": np.power,
        "CMPEQ": lambda a, b: (a == b).astype(a.dtype),
        "MAX": np.maximum,
    }
    return LazyBuffer(fns[op.name](self._np, other._np))

# ---- step 13: Function base ----
class Function:
    def __init__(self, *tensors):
        self.parents = tensors

# ---- this step: Exp ----
class Exp(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.EXP)          # cache y = e^x
        return self.ret

    def backward(self, grad_output):
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(self.ret, BinaryOps.MUL, grad_output)   # grad * e^x

# Step 20 - Sqrt
class Sqrt(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output):
        _, BinaryOps, _, _ = make_op_enums()
        two = LazyBuffer(np.array(2.0, dtype=self.ret.dtype))
        denom = lazybuffer_binary_e(two, BinaryOps.MUL, self.ret)
        return lazybuffer_binary_e(grad_output, BinaryOps.DIV, denom)

# Step 21 - Sigmoid
import numpy as np
from enum import Enum

# ---- step 3: op enums (SIGMOID and RELU added) ----
def make_op_enums():
    UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "EXP", "LOG", "SIN", "SQRT", "RECIPROCAL", "SIGMOID", "RELU"])
    BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ", "MAX"])
    ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
    MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "EXPAND", "PAD", "SHRINK", "STRIDE"])
    return UnaryOps, BinaryOps, ReduceOps, MovementOps

# ---- step 4: LazyBuffer (adds const + SIGMOID/RELU in e) ----
class LazyBuffer:
    def __init__(self, np_array):
        self._np = np.asarray(np_array)

    @property
    def shape(self):
        return self._np.shape

    @property
    def dtype(self):
        return self._np.dtype

    @staticmethod
    def const(value, shape):
        return LazyBuffer(np.full(shape, value))

    def e(self, op):
        fns = {
            "NOOP": lambda x: x.copy(),
            "NEG": np.negative,
            "EXP": np.exp,
            "LOG": np.log,
            "SIN": np.sin,
            "SQRT": np.sqrt,
            "RECIPROCAL": lambda x: 1.0 / x,
            "SIGMOID": lambda x: 1.0 / (1.0 + np.exp(-x)),
            "RELU": lambda x: np.maximum(x, 0),
        }
        return LazyBuffer(fns[op.name](self._np))

# ---- this step: Sigmoid ----
class Sigmoid(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.SIGMOID)          # cache y = σ(x)
        return self.ret

    def backward(self, grad_output):
        _, BinaryOps, _, _ = make_op_enums()
        one = LazyBuffer.const(1.0, self.ret._np.shape)
        one_minus_y = lazybuffer_binary_e(one, BinaryOps.SUB, self.ret)     # 1 - y
        local = lazybuffer_binary_e(self.ret, BinaryOps.MUL, one_minus_y)   # y * (1 - y)
        return lazybuffer_binary_e(local, BinaryOps.MUL, grad_output)       # g * y * (1 - y)

# Step 22 - Add
class Add(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.ADD, y)

    def backward(self, grad_output):
        needs_grad = getattr(self, "needs_input_grad", None)
        if needs_grad is None:
            needs_grad = [bool(getattr(p, "requires_grad", False)) for p in self.parents]
        return (
            grad_output if needs_grad[0] else None,
            grad_output if needs_grad[1] else None,
        )

# Step 23 - Sub
class Sub(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.SUB, y)

    def backward(self, grad_output):
        UnaryOps, _, _, _ = make_op_enums()
        needs_grad = getattr(self, "needs_input_grad", None)
        if needs_grad is None:
            needs_grad = [bool(getattr(p, "requires_grad", False)) for p in self.parents]
        gx = grad_output if needs_grad[0] else None
        gy = grad_output.e(UnaryOps.NEG) if needs_grad[1] else None
        return (gx, gy)

# Step 24 - Mul
class Mul(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.MUL, y)

    def backward(self, grad_output):
        _, BinaryOps, _, _ = make_op_enums()
        needs_grad = getattr(self, "needs_input_grad", None)
        if needs_grad is None:
            needs_grad = [bool(getattr(p, "requires_grad", False)) for p in self.parents]
        gx = lazybuffer_binary_e(self.y, BinaryOps.MUL, grad_output) if needs_grad[0] else None
        gy = lazybuffer_binary_e(self.x, BinaryOps.MUL, grad_output) if needs_grad[1] else None
        return (gx, gy)

# Step 25 - Div
class Div(Function):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return LazyBuffer(x._np / y._np)

    def backward(self, grad_output):
        needs_grad = getattr(self, "needs_input_grad", None)
        if needs_grad is None:
            needs_grad = [bool(getattr(p, "requires_grad", False)) for p in self.parents]
        grad_x = LazyBuffer(grad_output._np / self.y._np) if needs_grad[0] else None
        grad_y = (
            LazyBuffer(-grad_output._np * self.x._np / (self.y._np ** 2))
            if needs_grad[1]
            else None
        )
        return (grad_x, grad_y)

# Step 26 - sum_function_forward
class Sum(Function):
    def forward(self, x, axis):
        self.input_shape = x._np.shape
        self.axis = axis
        return LazyBuffer(x._np.sum(axis=axis, keepdims=True))

# Step 27 - sum_function_backward
def backward(self, grad_output):
    g = np.broadcast_to(grad_output._np, self.input_shape)
    g = np.ascontiguousarray(g)
    return LazyBuffer(g)


class Sum(Function):
    def forward(self, x, axis):
        self.input_shape = x._np.shape
        self.axis = axis
        return LazyBuffer(x._np.sum(axis=axis, keepdims=True))

    def backward(self, grad_output):
        return backward(self, grad_output)

# Step 28 - max_function_forward
class Max(Function):
    def forward(self, x, axis):
        self.x = x
        self.axis = axis
        out = x._np.max(axis=axis, keepdims=True)
        self.ret = LazyBuffer(out)
        return self.ret

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

