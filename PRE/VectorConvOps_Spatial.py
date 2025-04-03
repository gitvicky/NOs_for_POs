#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2nd July, 2024

Vector Operations implemented using the ConvOps Class 
Data used for all operations should be in the shape: BS, Nvar, Nx, Ny

Make sure BS and Nvar are present
"""
# %%

from PRE.ConvOps_Spatial import *
from Utils.boundary_conditions import * 

#############################################  
#Vector Operations 
#############################################  

#Dot Product between two vector field in 2D
def dot(a , b):
    return a[:, 0:1] * b[:, 0:1] + a[:, 1:2] * b[:, 1:2] 


#Cross Product between two vector fields in 2D
def cross(a , b):
    return a[:, 0:1] * b[:, 1:2] + a[:, 1:2] * b[:, 0:1] 

#Stacking the vectors along the i, j, k=None vectors
def vectorize(a, b):
    return torch.cat((a, b), dim=1)

class Gradient(ConvOperator):
    # 1 -> 2 
    def __init__(self, domain=('x','y'), order=1, scale=1.0, taylor_order=2, boundary_cond='periodic', conv='direct', device=torch.device("cpu"), requires_grad=False):
        super(Gradient, self).__init__()
        
        self.grad_x = ConvOperator(domain[0], order, scale, taylor_order, conv, device=torch.device("cuda"), requires_grad=True)
        self.grad_y = ConvOperator(domain[1], order, scale, taylor_order, conv, device=torch.device("cuda"), requires_grad=True)

        #Setting the boundary conditions
        self.bc = BoundaryManager(kernel_size=(taylor_order+1, taylor_order+1))
        self.bc.set_all_boundaries(bc_type=boundary_cond)


    def __call__(self, input_x, input_y=None):
        
        if input_y is None: 
            input_y = input_x

        #Padding to account for the boundary conditions 
        input_x = self.bc.pad_signal(input_x)
        input_y = self.bc.pad_signal(input_y)
        
        outputs = torch.cat((self.grad_x(input_x), self.grad_y(input_y)), dim=1)
        return outputs


class Laplace(ConvOperator):
    # 1 -> 1 Scalars
    # 2 -> 2  Vectors
    def __init__(self, domain=('x','y'), order=2, scale=1.0, taylor_order=2, boundary_cond='periodic', scalar=True, conv='direct', device='cpu', requires_grad=False):
        super(Laplace, self).__init__()

        self.laplace = ConvOperator(domain, order, scale, taylor_order, conv, device, requires_grad)
        self.scalar = scalar

        #Setting the boundary conditions
        self.bc = BoundaryManager(kernel_size=(taylor_order+1, taylor_order+1))
        self.bc.set_all_boundaries(bc_type=boundary_cond)

    def __call__(self, input_x, input_y=None):

        #Padding to account for the boundary conditions 
        input_x = self.bc.pad_signal(input_x)
        
        if self.scalar == True: 
            outputs = self.laplace(input_x)#scalar

        else: 

            if input_y is None: 
                input_y = input_x
            
            #Padding to account for the boundary conditions
            input_y = self.bc.pad_signal(input_y)

            # if input_x.dim() == 3: #No BS
                # outputs = torch.cat((self.laplace(input_x) , self.laplace(input_y)))#vector
            # else: #With BS
            outputs = torch.cat((self.laplace(input_x) , self.laplace(input_y)), dim=1)#vector

        return outputs


class Divergence(ConvOperator):
    # 2 -> 1 
    def __init__(self, domain=('x','y'), order=1, scale=1.0, taylor_order=2, boundary_cond='periodic', conv='direct', device='cpu', requires_grad=False):
        super(Divergence, self).__init__()
        
        self.grad_x = ConvOperator(domain[0], order, scale, taylor_order, conv, device, requires_grad)
        self.grad_y = ConvOperator(domain[1], order, scale, taylor_order, conv, device, requires_grad)
        
        #Setting the boundary conditions
        self.bc = BoundaryManager(kernel_size=(taylor_order+1, taylor_order+1))
        self.bc.set_all_boundaries(bc_type=boundary_cond)

    def __call__(self, input_x, input_y):

        #Padding to account for the boundary conditions 
        input_x = self.bc.pad_signal(input_x)
        input_y = self.bc.pad_signal(input_y)

        outputs = self.grad_x(input_x) + self.grad_y(input_y)
        return outputs



class Curl(ConvOperator):
    # 2 -> 1 
    def __init__(self, domain=('x','y'), order=1, scale=1.0, taylor_order=2, boundary_cond='periodic', conv='direct', device='cpu', requires_grad=False):
        super(Curl, self).__init__()
        
        self.grad_x = ConvOperator(domain[0], order, scale, taylor_order, conv, device, requires_grad)
        self.grad_y = ConvOperator(domain[1], order, scale, taylor_order, conv, device, requires_grad)

        #Setting the boundary conditions
        self.bc = BoundaryManager(kernel_size=(taylor_order+1, taylor_order+1))
        self.bc.set_all_boundaries(bc_type=boundary_cond)

    def __call__(self, input_x, input_y):

        #Padding to account for the boundary conditions 
        input_x = self.bc.pad_signal(input_x)
        input_y = self.bc.pad_signal(input_y)


        outputs = self.grad_x(input_y) - self.grad_y(input_x)
        return outputs
    

class Vector_Gradient(ConvOperator):
    # 2 -> 1 
    def __init__(self, domain=('x','y'), order=1, scale=1.0, taylor_order=2, boundary_cond='periodic', conv='direct', device=torch.device("cpu"), requires_grad=False):
        super(ConvOperator, self).__init__()
        
        self.grad_x = ConvOperator(domain[0], order, scale, taylor_order, conv, device=torch.device("cuda"), requires_grad=True)
        self.grad_y = ConvOperator(domain[1], order, scale, taylor_order, conv, device=torch.device("cuda"), requires_grad=True)


        #Setting the boundary conditions
        self.bc = BoundaryManager(kernel_size=(taylor_order+1, taylor_order+1))
        self.bc.set_all_boundaries(bc_type=boundary_cond)


    def __call__(self, input_x, input_y):
        

        #Padding to account for the boundary conditions 
        input_x = self.bc.pad_signal(input_x)
        input_y = self.bc.pad_signal(input_y)
        
        outputs = self.grad_x(input_x)**2 + self.grad_y(input_y)**2 + 2*self.grad_y(input_x)*self.grad_x(input_y)

        return outputs