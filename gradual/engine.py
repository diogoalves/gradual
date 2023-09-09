import math 
import numpy as np

class Value:
  ''' 
      Base class used to define an auto-diff expression graph.
      It represents a single scalar value.
  '''

  def __init__(self, data, _children=(), _op=(), label=''):
    self.data = data
    self.grad = 0
    self.label = label

    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __pow__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data ** other.data, (self,), f'**{other}')

    def _backward():
      self.grad += ((other.data * (self.data ** (other.data) ))/ self.data) * out.grad
      try:
        other.grad += ((self.data ** other.data) * math.log(self.data)) * out.grad
      except:
        other.grad += 0
    out._backward = _backward

    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
      self.grad += (out.data > 0) *  out.grad
    out._backward = _backward

    return out
  
  def tanh(self):
    e2x = math.e ** (2 * self.data)
    out = Value( (e2x - 1)/(e2x + 1), (self,), 'Tanh')

    def _backward():
      self.grad += (1 - out.data**2) * out.grad
    out._backward = _backward

    return out
  
  def exp(self):
    out = Value( math.exp(self.data), (self,), 'Exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out

  # composite operator functions
  def __neg__(self): return self * -1
  def __sub__(self, other): return self + (-other)
  def __truediv__(self, other): return self * (other ** -1)

  # reverse operator functions
  def __radd__(self, other): return self + other # other + self
  def __rsub__(self, other): return other + (-self) # other - self
  def __rmul__(self, other): return self * other # other * self
  def __rtruediv__(self, other): return other * (self ** - 1) # other / self

  # printable representation
  def __repr__(self): return f'Value(data={self.data}, grad={self.grad}, label={self.label})'
  
  def backward(self):
    # topological order all of the children in the graph
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    # go one variable at a time and apply the chain rule to get its gradient
    self.grad = 1
    for v in reversed(topo):
      v._backward()    





class Tensor:
  ''' 
      Tensor class used to define an auto-diff expression graph.
  '''

  def __init__(self, data, _children=(), _op=(), label='', dtype=np.single):
    self.data = np.array(data, dtype=dtype)
    self.grad = np.zeros(self.data.shape)
    self.label = label
    self.dtype = dtype

    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op

  def __add__(self, other):
    if not isinstance(other, Tensor):
      if isinstance(other, (int, float)):
        other = Tensor([other], dtype=self.dtype)
      else:
        other = Tensor(other, dtype=self.dtype)
    out = Tensor(self.data + other.data, (self, other), '+', dtype=self.dtype)

    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    if not isinstance(other, Tensor):
      if isinstance(other, (int, float)):
        other = Tensor([other], dtype=self.dtype)
      else:
        other = Tensor(other, dtype=self.dtype)
    out = Tensor(self.data * other.data, (self, other), '*', dtype=self.dtype)

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __pow__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
    out = Tensor(self.data ** other.data, (self,), f'**{other}', dtype=self.dtype)

    def _backward():
      self.grad += ((other.data * (self.data ** (other.data) ))/ self.data) * out.grad
      try:
        with np.errstate(divide = 'ignore'):
          other.grad += ((self.data ** other.data) * np.log(self.data)) * out.grad
      except:
        other.grad += 0
    out._backward = _backward

    return out

  def relu(self):
    out = Tensor(np.maximum(0, self.data), (self,), 'ReLU', dtype=self.dtype)

    def _backward():
      self.grad += (out.data > 0) *  out.grad
    out._backward = _backward

    return out
  
  def tanh(self):
    out = Tensor(np.tanh(self.data), (self,), 'Tanh', dtype=self.dtype)

    def _backward():
      self.grad += (1 - (out.data ** 2)) * out.grad
    out._backward = _backward

    return out

  def exp(self):
    out = Tensor( np.exp(self.data), (self,), 'Exp', dtype=self.dtype)

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out

  def sum(self):
    out = Tensor([np.sum(self.data)], (self,), 'Sum', dtype=self.dtype)

    def _backward():
      self.grad += out.grad
    out._backward = _backward

    return out
      

  # composite operator functions
  def __neg__(self): return self * [-1]
  def __sub__(self, other): return self + (-other)
  def __truediv__(self, other):
    if not isinstance(other, Tensor):
      if isinstance(other, (int, float)):
        other = Tensor([other], dtype=self.dtype)
      else:
        other = Tensor(other, dtype=self.dtype)
    return self * (other ** -1)

  # reverse operator functions
  def __radd__(self, other): return self + other # other + self
  def __rsub__(self, other): return other + (-self) # other - self
  def __rmul__(self, other): return self * other # other * self
  def __rtruediv__(self, other): return other * (self ** [- 1]) # other / self

  def __repr__(self): return f'Tensor(data={self.data}, grad={self.grad}, label={self.label} dtype={self.dtype})'

  def backward(self):
    # topological order all of the children in the graph
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    # go one variable at a time and apply the chain rule to get its gradient
    self.grad = np.ones(1)
    for v in reversed(topo):
      v._backward()