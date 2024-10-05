import numpy as np

class Tensor:
    def __init__(self, data, parents=[], label=None):
        self.data = np.array(data).astype(np.float32)
        self.label = label
        self.grad = None
        self._backward = None
        self.parents = parents

    def backward(self):
        #Only runs on the top node
        self.grad = np.ones_like(self.data)
            
        self._b()
    def _b(self):
        if self._backward:
            self._backward()
            for parent in self.parents:
                parent._b()
 
    def dot(self, other):
        label = "DOT"
        out = Tensor(self.data.dot(other.data), parents=[self, other], label=label)
    
        def _backward():
            self.grad = out.grad.dot(other.data.T) 
            other.grad = out.grad.T.dot(self.data).T 
        out._backward = _backward
        return out

    #------------- arithmetic -------------#
    def mul(self, other):
        label = "*"
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, parents=[self,other], label=label)

        def _backward():
            other.grad = self.data * out.grad
            self.grad = other.data * out.grad
        out._backward = _backward
        return out
        
    def add(self, other):
        label = "+"
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, parents=[self,other], label=label)

        def _backward():
            other.grad = out.grad
            self.grad = out.grad
        out._backward = _backward
        return out
        
    def sum(self):
        label = "∑"
        out = Tensor(np.array([self.data.sum()]), parents=[self], label=label)

        def _backward():
            self.grad = np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    #------------- activation funcs -------------#
    def relu(self):
        label="ReLu"
        out = Tensor(np.maximum(0,self.data), parents=[self], label=label)

        def _backward():
            self.grad = (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self): 
        label="σ"
        def sig(z):
            return 1/(1+ np.exp(-z))
        def sig_prime(z): #Where z is output.data OR sig(input.data)???
            ret = z * (1-z)
            return ret 
            
        out = Tensor(sig(self.data), parents=[self],label=label)
        def _backward(): #Dont know if this works 
            self.grad = sig_prime(out.data) * out.grad

        out._backward = _backward
        return out

    def square(self):
        label="x^2"
        out = Tensor(self.data**2, parents=[self],label=label)

        def _backward():
            self.grad = self.data * out.grad
            
        out._backward = _backward
        return out
        
    def mean(self):
        div = np.array([1/self.data.size])
        return self.sum().mul(div)

    #-------- clean --------#
    def __add__(self, other):
        return self.add(other)
        
    def __mul__(self, other):
        return self.mul(other)

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, label={self.label})"