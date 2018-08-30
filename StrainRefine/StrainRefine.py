#!/usr/bin/env python
import numpy as np

from scipy.optimize import  minimize

from scipy import linalg as La
from numpy import pi as pi
from numpy import arccos as acos
from numpy import rad2deg as r2d

class StrainRefine:
    """Class to calculate strain from DAXM Data read from .xml files """
    
    def __init__(self,Q0,Qx,Qy,Qz,H,K,L):
        self.Q0=Q0
        self.Qx=Qx
        self.Qy=Qy
        self.Qz=Qz
        self.H=H
        self.K=K
        self.L=L
    
    
    
    
    def constraint(self,X,eps):
        return len(X)*eps-np.sum(np.abs(X))
      
    
    def InputPar(self):
        Q0=self.Q0
        Qx=self.Qx
        Qy=self.Qy
        Qz=self.Qz
        H,K,L=self.H,self.K,self.L
        qVecs=np.zeros((len(Qx),3))
        LqVecs=len(qVecs)
        Q0=self.Q0
        qVecs=np.zeros((len(Qx),3))
        LqVecs=len(qVecs)
        #Arrange all the Measured Scattering Vectors--They are already in unit vector form from DAXM!
        for i in range(len(Qx)):
            qVecs[i,:]=np.array([Qx[i],Qy[i],Qz[i]])

        hkl=np.zeros((len(H),3))
        for i in range(len(H)):
            hkl[i,0],hkl[i,1],hkl[i,2]=H[i],K[i],L[i]
        Lhkl=len(hkl)
        # Initialize Array to hold ideal g-vectors corresponding to available hkls
        qOb=np.zeros((len(hkl),3))
        for j in range(len(qOb)):
            qOb[j,:]=np.dot(Q0,hkl[j,:])
            qOb[j,:]=qOb[j,:]/La.norm(qOb[j,:])
            DotProd=np.zeros((len(qOb),len(qVecs)))
            for i in range(len(qOb)):
                for j in range(len(qVecs)):
                    DotProd[i,j]=np.dot(qOb[i,:],qVecs[j,:])


        MaxDot=[np.where(DotProd[i]==max(np.abs(DotProd[i])))[0][0]for i in range(len(DotProd))]
        Mhkl=qVecs[MaxDot]
        return qOb,Mhkl


    def DirectoRecip(self,B):
        """Method to convert given direct lattice (with columns as
        individual vectors in a 3x3 array), to equivalent reciprocal lattice"""
        Q=2*np.pi*La.inv(B.T)
        return Q
    
    def ReciptoDirect(self,Q):
        """Method to convert given reciprocal lattice (with columns as
        individual vectors in a 3x3 array), to equivalent direct lattice"""
        B=2*np.pi*La.inv(Q.T)
        return B
    
    
    def GetDefGrad(self,qOb,X0,Mhkl,Q0,maxiter,eps,tol):
        """Method to calculate deformation gradient using an Error minimization scheme.
        For details refer to work by Zhang et al: 
        https://doi.org/10.1016/j.scriptamat.2018.05.028"""
        def Objective(X,Mhkl,qOb):
            """Set up Objective Function"""
            FStar=X.reshape(3,3)
            FStar=1e-18*np.eye(3,3)+FStar#This is to prevent singular matrix from occurring
            QSum=0
            N=len(Mhkl)
            for i in range(len(Mhkl)):
                Fq=np.dot(FStar,qOb[i])
                Fq=Fq/La.norm(Fq)
                qTemp2=La.norm(Mhkl[i]-Fq)
                qTemp2=qTemp2**2
                QSum=QSum+qTemp2
            QSum=QSum/N
            return np.sqrt(QSum)
        
        def Volume(X,Q0):
            """Constrain Volume of lattice [This step is important for deviatoric Strain]"""
            FStar=X.reshape(3,3)
            F=La.inv(FStar.T)
            B0=2*pi*La.inv(Q0.T)
            B=np.dot(F,B0)
            J=La.det(F) # Evaluate Jacobian |F|
            V0=La.det(B0)
            V=J*V0
            return -(J-1)+1e-14
        
        cont={'type':'ineq','fun':lambda x:Volume(x,Q0)}
        self.sol=minimize(Objective,X0,args=(Mhkl,qOb),constraints=cont,\
             method='COBYLA',tol=tol,options={'maxiter':maxiter,})
        X=self.sol.x
        #If the optimization is successful then deformation gradient is returned else NaN
        if self.sol.success==True:
           FStar=X.reshape(3,3)
           F=La.inv(FStar.T)
           Fd=F/(np.abs(La.det(F)))**(1/3.0)
           
        else:
           Fd=float('NaN')*np.ones((3,3))
        FuncEval=Objective(self.sol.x,Mhkl,qOb)
        return Fd,FuncEval #Deviatoric Deformation Gradient and Calculated Magnitude of Error
    
    def GetStrain(self,F):
        """Method to calculate strain tensor:given deformation gradient"""
        RU=La.polar(F) #Polar decomposition of F into Rotational and Stretch components
        Eig=La.eig(RU[1])
        V=Eig[1].T
        D=np.eye(3,3)
        D[0,0]=np.real(Eig[0][0])
        D[1,1]=np.real(Eig[0][1])
        D[2,2]=np.real(Eig[0][2])
        D[0,0],D[1,1],D[2,2]=np.sqrt(D[0,0]),np.sqrt(D[1,1]),np.sqrt(D[2,2])
        Usqrt=np.dot(V,D)
        Usqrt=np.dot(Usqrt,La.inv(V))
        STensor=Usqrt-np.eye(3,3)
        return STensor
    
    def GetMises(self,epsilon):
        """Method to calculate von Mises strain given strain tensor.
           Formula used is from http://www.continuummechanics.org/vonmisesstress.html"""
        Hydrostatic=(epsilon[0,0]+epsilon[1,1]+epsilon[2,2])\
                    /3.0
        Deviatoric=epsilon-Hydrostatic*np.eye(3,3)
        vMStrain=(2.0/3.0)*np.sqrt\
                 (np.tensordot(Deviatoric,Deviatoric,axes=2)) 
        
        return vMStrain

    def LatticeParams(self,Q0,Fd):
        """Method to return Lattice parameters of the strained unit cell"""
        B0=2*np.pi*La.inv(Q0.T)
        B=np.dot(Fd,B0)
        gamma=np.arccos(np.dot(B[:,0]/La.norm(B[:,0]),\
                       B[:,1]/La.norm(B[:,1])))*180/pi
        alpha=np.arccos(np.dot(B[:,0]/La.norm(B[:,0]),\
                       B[:,2]/La.norm(B[:,2])))*180/pi
        beta=np.arccos(np.dot(B[:,1]/La.norm(B[:,1]),\
                      B[:,2]/La.norm(B[:,2])))*180/pi

        a=La.norm(B[:,0])
        b=La.norm(B[:,1])
        c=La.norm(B[:,2])
        return [a,b,c,alpha,beta,gamma]
    
    def LatticeParamsRef(self,Q0):
        """Method to return Lattice parameters of the strain-free unit cell"""
        B0=2*np.pi*La.inv(Q0.T)
        gamma=np.arccos(np.dot(B0[:,0]/La.norm(B0[:,0]),\
                       B0[:,1]/La.norm(B0[:,1])))*180/pi
        alpha=np.arccos(np.dot(B0[:,0]/La.norm(B0[:,0]),\
                       B0[:,2]/La.norm(B0[:,2])))*180/pi
        beta=np.arccos(np.dot(B0[:,1]/La.norm(B0[:,1]),\
                      B0[:,2]/La.norm(B0[:,2])))*180/pi

        a=La.norm(B0[:,0])
        b=La.norm(B0[:,1])
        c=La.norm(B0[:,2])
        return [a,b,c,alpha,beta,gamma]
        
        
