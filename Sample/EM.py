import numpy as np

def weighted_mean(X,gamma):
    gamma=gamma[:,:,np.newaxis]
    wX=gamma*X[np.newaxis,:,:]
    return (wX.sum(axis=1))/gamma.sum(axis=1)

def bernoulli_llikelihood(X,pi,theta):
    theta_p=theta[:,np.newaxis,:]
    X_p=X[np.newaxis,:,:]       
    p_p=np.power(theta_p,X_p)*np.power(1-theta_p,1-X_p)
    p=p_p.prod(axis=2)               
    prob=pi[:,np.newaxis]*p
    p=prob.sum(axis=0)
    l=np.log(p)
    return np.sum(l)

def bernoulli_E_step(X,pi,theta):
    theta_p=theta[:,np.newaxis,:]
    X_p=X[np.newaxis,:,:]       
    p_p=np.power(theta_p,X_p)*np.power(1-theta_p,1-X_p)
    p=p_p.prod(axis=2)               
    gamma=pi[:,np.newaxis]*p
    gamma=gamma/gamma.sum(axis=0,keepdims=True) # normalize
    return gamma

def bernoulli_M_step(X,gamma):
    K,N=gamma.shape
    pi=gamma.mean(axis=1) 
    theta=weighted_mean(X,gamma)
    return pi,theta

def bernoulli_estimate_mixture(X,pi_guess,theta_guess,nsteps=100,tol=1e-8):
    pi=pi_guess
    theta=theta_guess
    l0=bernoulli_llikelihood(X,pi,theta)
    print("step",0,"loss = ",l0)
    for t1 in range(nsteps):
        gamma=bernoulli_E_step(X,pi,theta)
        pi,theta=bernoulli_M_step(X,gamma)
        l=bernoulli_llikelihood(X,pi,theta)
        print("step",t1+1,"loss = ",l)
        if abs((l-l0)/l0)<tol:
            break
        l0=l
    return l,pi,theta,gamma