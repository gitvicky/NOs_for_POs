#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2nd July, 2024

Vector Operations implemented using the ConvOps Class 
Data used for all operations should be in the shape: BS, Nt, Nx, Ny
"""
# %%
from ConvOps_2d import *

#############################################  
#Vector Operations 
#############################################  

#Dot Product between two vector field in 2D
def dot(a , b):
    return a[0] * b[0] + a[1] * b[1] 

#Cross Product between two vector fields in 2D
def cross(a , b):
    return a[0] * b[1] + a[1] * b[0] 

#Stacking the vectors along the i, j, k=None vectors
def vectorize(a, b):
    return torch.stack((a, b))


class Divergence(ConvOperator):
    def __init__(self, domain=('x','y'), order=1, scale=1.0, taylor_order=2, require_grad=False):
        super(Divergence, self).__init__()
        
        self.grad_x = ConvOperator(domain[0], order, scale, taylor_order, require_grad)
        self.grad_y = ConvOperator(domain[1], order, scale, taylor_order, require_grad)

    def __call__(self, input_x, input_y):

        outputs = self.grad_x(input_x) + self.grad_y(input_y)
        return outputs

class Gradient(ConvOperator):
    def __init__(self, domain=('x','y'), order=1, scale=1.0, taylor_order=2, requires_grad=False):
        super(Gradient, self).__init__()
        
        self.grad_x = ConvOperator(domain[0], order, scale, taylor_order, requires_grad)
        self.grad_y = ConvOperator(domain[1], order, scale, taylor_order, requires_grad)

    def __call__(self, input_x, input_y=None):
        
        if input_y is None: 
            input_y = input_x
        
        outputs = torch.stack((self.grad_x(input_x), self.grad_y(input_y)))
        return outputs
    
class Curl(ConvOperator):
    def __init__(self, domain=('x','y'), order=1, scale=1.0, taylor_order=2, requires_grad=False):
        super(Curl, self).__init__()
        
        self.grad_x = ConvOperator(domain[0], order, scale, taylor_order, requires_grad)
        self.grad_y = ConvOperator(domain[1], order, scale, taylor_order, requires_grad)

    def __call__(self, input_x, input_y):

        outputs = self.grad_x(input_y) - self.grad_y(input_x)
        return outputs
    

class Laplace(ConvOperator):
    def __init__(self, domain=('x','y'), order=2, scale=1.0, taylor_order=2, requires_grad=False):
        super(Laplace, self).__init__()

        self.laplace = ConvOperator(domain, order, scale, taylor_order, requires_grad)

    def __call__(self, input_x, input_y=None):
        
        if input_y is None: 
            input_y = input_x
        
        outputs = torch.stack((self.laplace(input_x) , self.laplace(input_y)))
        return outputs

