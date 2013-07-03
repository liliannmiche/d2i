import numpy
from numpy.random import randn
from numpy import zeros, array, dot, ones, concatenate, arange, transpose, amax, argmax, delete, append, reshape, ix_
from numpy.linalg import inv

# Helper function for MRSR.
#  
# Input:
# XxXx    is the matrix [X x]'*[X x].
# invXX   is the inverse matrix of X'*X.
#
# Output:
# invXxXx is the inverse matrix of [X x]'*[X x].
#
# Reference: 
# Mark JL Orr. Introduction to Radial Basis Function Networks.
# Centre for Cognitive Science, University of Edinburgh, Technical
# Report, April 1996.

def update_inverse(XxXx, invXX):
	m = XxXx.shape[0]-1
	M1 = XxXx[:m, m]
	M1 = M1.reshape(M1.shape[0], 1)
	M2 = numpy.dot(invXX, M1)
	p = XxXx[m+1,m+1]-numpy.dot(M1.T, M2)
	invXxXx = numpy.dot(numpy.vstack((M2, -1)), numpy.hstack((M2.T, -1*numpy.ones((M2.shape[1],1)))))/p
	invXxXx[:m,:m] = invXxXx[:m,:m]+invXX
	return invXxXx


# Multiresponse Sparse Regression algorithm in Python
# Uses Numpy library from Python
#
# mrsr(T, X, kmax) returns W, i1
#  
# Input:
# T    is an (n x p) matrix of targets. The columns of T should
#      have zero mean and same scale (e.g. equal variance).
# X    is an (n x m) matrix of regressors. The columns of X should
#      have zero mean and same scale (e.g. equal variance).
# kmax is an integer fixing the number of steps to be run, which
#      equals to the maximum number of regressors in the model.
#  
# Output:
# W    is an (m x p*kmax) sparse matrix of regression
#      coefficients. It can be converted to full matrix by command   
#      full(W). Regression coefficients of the k:th step are given
#      by W(:,(k-1)*p+1:k*p).
# i1   is a (1 x kmax) vector of indices revealing the order in
#      which the regressors enter model. 
# 
# The estimates for T may be obtained by Y = X*W, where the k:th
# estimate Y(:,(k-1)*p+1:k*p) uses k regressors.
#  
# Reference: 
# Timo Simila, Jarkko Tikka. Multiresponse sparse regression with
# application to multidimensional scaling. International Conference
# on Artificial Neural Networks (ICANN). Warsaw, Poland. September
# 11-15, 2005. LNCS 3697, pp. 97-102.

# Copyright (C) 2005 by Timo Simila and Jarkko Tikka.
#
# This function is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of
# the License, or any later version.   
#
# The function is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
# http://www.gnu.org/copyleft/gpl.html  

def mrsr( T, X, kmax):

	n,m = X.shape
	n,p = T.shape
		
	kmax = min(kmax, m)
		
	i1 = numpy.array([], dtype = int)
	i2 = numpy.array(numpy.arange(m), dtype = int)
	#print X.T.shape
	#print T.shape
	XT = numpy.dot(X.T,T)
	XX = numpy.zeros([m, m])
	
	S = numpy.ones([2**p, p])
	S[0:2**(p-1), 0] = -1
	for j in range(1, p):
		S[:, j] = numpy.concatenate((S[numpy.arange(1, 2**p, 2), j-1], S[numpy.arange(1, 2**p, 2), j-1]))
	
	
	
	# Make the first step
	
	A    = numpy.transpose(XT)
	cmax = numpy.amax(abs(A).sum(0), 0)
	cind = numpy.argmax(abs(A).sum(0), 0)
	A    = numpy.delete(A, cind, 1)
	ind  = int(i2[cind])
	i2   = numpy.delete(i2, cind)
	i1   = numpy.append(i1, ind)
	
	XX[numpy.ix_([ind], [ind])] = numpy.dot(X[:,ind], X[:,ind])
	
	invXX = 1 / XX[ind, :][ind]
	Wols  = invXX * XT[ind, :]
	Yols  = numpy.dot(numpy.reshape(X[:, ind], (-1, 1)), numpy.reshape(Wols, (1,-1)))
	B     = numpy.dot(Yols.T, X[:, i2])
	G     = (cmax+numpy.dot(S,A))/(cmax+numpy.dot(S,B))
	g     = min(G[G>=0])
	
	
	Y = g*Yols
	
	
	
	# Rest of the steps
	for k in numpy.arange(2,kmax+1):
		#print "calculating rank %d/%d" % (k-1, kmax)		
  
		A    = numpy.dot((T-Y).T, X[:, i2])
		cmax = numpy.amax(abs(A).sum(0), 0)
		cind = numpy.argmax(abs(A).sum(0), 0)
		A    = numpy.delete(A, cind, 1)
		ind  = int(i2[cind])
		i2   = numpy.delete(i2, cind)
		i1   = numpy.append(i1, ind)    
		xX   = numpy.dot(X[:, ind].T, X[:, i1])
		
		XX[numpy.ix_([ind], i1)] = xX
		XX[numpy.ix_(i1, [ind])] = numpy.reshape(xX, (i1.size, -1))
		
		invXX = numpy.linalg.inv(XX[i1, :][:, i1])
		
		Wols = numpy.dot(invXX, XT[i1, :])
		Yols = numpy.dot(X[:, i1], Wols)
		B    = numpy.dot(numpy.transpose(Yols-Y), X[:, i2])
		G    = (cmax + numpy.dot(S, A)) / (cmax + numpy.dot(S ,B))
		
		G = numpy.concatenate(([2*(k==m)-1], G.flatten()), 1)
		g = min(G[G>=0])
		
		Y = (1-g)*Y+g*Yols
		
	return i1
	
