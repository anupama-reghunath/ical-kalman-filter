"""
A python based implementation of Kalman Filter based on work by Kolahal B

Author: Anupama Reghunath

"""

import time                                     
start = time.time()      

import math   
import numpy as np       
import pandas as pd                             
import sympy as sym                             
import matplotlib.pyplot as plt                 
from scipy.interpolate import CubicSpline       #fitting

#import sys
#sys.stdout = open('output_results.txt', 'w')   #printing output onto a txt file

TrueEnergy=10

m=0.105658                      #muon mass in GeV/c^2
m_e=0.511*10**-3                #electron mass in GeV/c^2
K=0.299792458*10**-3            #GeV c^-1 T^-1 mm^-1
q=0.303                         #charge of mu+ (1 C in natural units)
by=0                            #magnetic field along y axis


class Datahandling:
            
    def __init__(self,df,event_index):
    
        self.eid_ = df[df['eid']==event_index]
        self.num_of_layers=len(self.eid_)               #no. of the layers per event
        self.data_per_eid=np.array(self.eid_)           #array of dataframe pertaining to one particular event
        self.x_m=self.data_per_eid[:,4]
        self.z_m=self.data_per_eid[:,5]
        self.y_m=self.data_per_eid[:,6]  #interchange of z to y because of ICAL simulation geometry(y is the perpendicular axis)

        self.M_k =np.array([[ None for i in range(2) ] for j in range(len(self.data_per_eid)) ] )#measured state vector
                
        for i in range(len(self.data_per_eid)): 
            self.M_k[i][0]=self.x_m[i]
            self.M_k[i][1]=self.y_m[i]
    
    def Initialisation(self):                       #Runs only once per event
        
        self.initial_E=1000.0                      
        self.E=[]   
        self.E.append(self.initial_E)
        
        C_k =np.array( 10**6*np.identity(5,  dtype=np.float))              #Covariance
        
        prior_sv = np.array([ None for i in range(5)] )
        
        #Defining the initial state vector from measured
        prior_sv[0]= self.x_m[0]
        prior_sv[1]= self.y_m[0]
        prior_sv[2]= (self.x_m[1]-self.x_m[0])/(self.z_m[1]-self.z_m[0]) #tx_0 =(x1-x0)/(z1-z0)
        prior_sv[3]= (self.y_m[1]-self.y_m[0])/(self.z_m[1]-self.z_m[0]) #ty_0 =(y1-y0)/(z1-z0)
        prior_sv[4]= 10**-7                                              #initial q/p
        
        self.true_sv=prior_sv
        return prior_sv,C_k
        

class Symbolic_expressions:                #Prediction equations in symbolic format;run only once to reduce compiling time
    
    def Prediction_expressions(self):
    
        x,y,tx,ty,qp,dz,p,bx=sym.symbols('x y tx ty qp dz p bx')    #qp denotes q/p        

        #Magnetic Field Integrals
        S_x=0.5*bx*dz**2
        S_y=0.5*by*dz**2
        S_xx=bx**2*dz**3 /6
        S_yy=by**2*dz**3 /6
        S_xy=bx*by*dz**3 /6
        S_yx=S_xy
        R_x=bx*dz
        R_y=by*dz
        R_xy=0.5*bx*by*dz**2
        R_xx=0.5*bx*bx*dz**2
        R_yy=0.5*by*by*dz**2
        R_yx=R_xy

        def h():
            return(K*qp*(1+ (tx)**2 +(ty)**2)**0.5)

        #Symbolic definitions of Prediction equations(for x, y, tx, ty)

        def f_x():
            return (x  + tx*dz + h()*(tx*ty*S_x-(1+tx**2)*S_y) + h()**2*(tx*(3*ty**2 +1)*S_xx -ty*(3*tx**2 +1)*S_xy -ty*(3*tx**2 +1)*S_yx +tx*(3*tx**2 +3)*S_yy) )
        
        def f_y():
            return(y  + ty*dz + h()*( (1+ty**2)*S_x -tx*ty*S_y ) + h()**2*( ty*(3*ty**2 +3)*S_xx -tx*(3*ty**2 +1)*S_xy -tx*(3*ty**2 +1)*S_yx +ty*(3*tx**2 +1)*S_yy) )
    
        def f_tx():
            return(tx + h()*(tx*ty*R_x-(1+tx**2)*R_y) + h()**2*(tx*(3*ty**2 +1)*R_xx -ty*(3*tx**2 +1)*R_xy -ty*(3*tx**2 +1)*R_yx +tx*(3*tx**2 +3)*R_yy))
        
        def f_ty():
            return(ty + h()*((1+ty**2)*R_x -tx*ty*R_y) + h()**2*(ty*(3*(ty**2) +3)*R_xx -tx*(3*ty**2 +1)*R_xy -tx*(3*ty**2 +1)*R_yx +ty*(3*tx**2 +1)*R_yy))

        #Converting symbolic to mathematical

        self.Prediction_x = sym.lambdify((x,y,tx,ty,qp,dz,bx), f_x(), "numpy")
        self.Prediction_y = sym.lambdify((x,y,tx,ty,qp,dz,bx), f_y(), "numpy")
        self.Prediction_tx= sym.lambdify((x,y,tx,ty,qp,dz,bx), f_tx(),"numpy")
        self.Prediction_ty= sym.lambdify((x,y,tx,ty,qp,dz,bx), f_ty(),"numpy")
        #****************************************************************************
        
        #Derivatives of the prediction equations for Propagator Matrix Elements

        #Row 1
        def Prediction_xprimex():   
            return f_x().diff(x)
        def Prediction_xprimey():   
            return f_x().diff(y)
        def Prediction_xprimetx():  
            return f_x().diff(tx)
        def Prediction_xprimety():  
            return f_x().diff(ty)
        def Prediction_xprimeqp(): 
            return f_x().diff(qp)    

        #Row 2
        def Prediction_yprimex():   
            return f_y().diff(x)
        def Prediction_yprimey():   
            return f_y().diff(y)
        def Prediction_yprimetx():  
            return f_y().diff(tx)
        def Prediction_yprimety():  
            return f_y().diff(ty)
        def Prediction_yprimeqp(): 
            return f_y().diff(qp)

        #Row 3
        def Prediction_txprimex():
            return f_tx().diff(x)
        def Prediction_txprimey():
            return f_tx().diff(y)
        def Prediction_txprimetx():
            return f_tx().diff(tx)
        def Prediction_txprimety():
            return f_tx().diff(ty)
        def Prediction_txprimeqp():
            return f_tx().diff(qp)

        #Row 4
        def Prediction_typrimex():
            return f_ty().diff(x)
        def Prediction_typrimey():
            return f_ty().diff(y)
        def Prediction_typrimetx():
            return f_ty().diff(tx)
        def Prediction_typrimety():
            return f_ty().diff(ty)
        def Prediction_typrimeqp():
            return f_ty().diff(qp)

        #Converting symbolic to mathematical

        self.dx_dx  = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_xprimex(), "numpy") 
        self.dx_dy  = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_xprimey(), "numpy")
        self.dx_dtx = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_xprimetx(),"numpy")
        self.dx_dty = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_xprimety(),"numpy")
        self.dx_dqp = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_xprimeqp(),"numpy")    

        self.dy_dx  = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_yprimex(), "numpy")
        self.dy_dy  = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_yprimey(), "numpy")
        self.dy_dtx = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_yprimetx(),"numpy")
        self.dy_dty = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_yprimety(),"numpy")
        self.dy_dqp = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_yprimeqp(),"numpy")    

        self.dtx_dx = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_txprimex(), "numpy")
        self.dtx_dy = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_txprimey(), "numpy")
        self.dtx_dtx= sym.lambdify( (tx,ty,qp,dz,bx), Prediction_txprimetx(),"numpy")
        self.dtx_dty= sym.lambdify( (tx,ty,qp,dz,bx), Prediction_txprimety(),"numpy")
        self.dtx_dqp= sym.lambdify( (tx,ty,qp,dz,bx), Prediction_txprimeqp(),"numpy")    

        self.dty_dx = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_typrimex(), "numpy")
        self.dty_dy = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_typrimey(), "numpy")
        self.dty_dtx= sym.lambdify( (tx,ty,qp,dz,bx), Prediction_typrimetx(),"numpy")
        self.dty_dty= sym.lambdify( (tx,ty,qp,dz,bx), Prediction_typrimety(),"numpy")
        self.dty_dqp= sym.lambdify( (tx,ty,qp,dz,bx), Prediction_typrimeqp(),"numpy")    
        #****************************************************************************

        #Range-momentum relation for Error Propagation of q/p  

        cb =pd.read_csv("muon-iron-energyLossTable3.txt",sep=" ") 
        cb1=np.array(cb)

        pp   =cb1[:,3]*10**-3             #converting MeV/c into GeV/c
        l_r  =cb1[:,10]*10/7.874          #in q/cm^2 to mm (/iron density)
        
        #Alternate equation for Energy-loss in Iron
            #ene_r=cb1[:,9]*7.874            
            #self.EnergylossIron_CubicSpline = CubicSpline(pp,ene_r,bc_type='natural')     

        self.range_l = CubicSpline(pp,l_r,bc_type='natural')   #l=f(p)

        fl=CubicSpline(l_r,pp,bc_type='natural')               #p=f(l)

        self.fl__1=CubicSpline.derivative(fl,nu=1)    #f'(l)
        self.fl__2=CubicSpline.derivative(fl,nu=2)    #f''(l)
        self.fl__3=CubicSpline.derivative(fl,nu=3)    #f'''(l)
        #****************************************************************************
        

        def Prediction_xprimep():
            def f_x():                  #Prediction(x) with p explicitly defined
                h=K*q/p*(1+ (tx)**2 +(ty)**2)**0.5
                return x  + tx*dz + h*(tx*ty*S_x-(1+tx**2)*S_y) + h**2*( tx*(3*ty**2 +1)*S_xx -ty*(3*tx**2 +1)*S_xy -ty*(3*tx**2 +1)*S_yx +tx*(3*tx**2 +3)*S_yy )  
            return f_x().diff(p)        #dx/dp

        def Prediction_yprimep():
            def f_y():                  #Prediction(y) with p explicitly defined
                h=K*q/p*(1+ (tx)**2 +(ty)**2)**0.5
                return y  + ty*dz +h*((1+ty**2)*S_x -tx*ty*S_y) + h**2*(ty*(3*ty**2 +3)*S_xx -tx*(3*ty**2 +1)*S_xy -tx*(3*ty**2 +1)*S_yx +ty*(3*tx**2 +1)*S_yy)
            return f_y().diff(p)        #dy/dp

        def Prediction_txprimep():
            def f_tx():                 #Prediction(tx) with p explicitly defined
                h=K*q/p*(1+ (tx)**2 +(ty)**2)**0.5
                return (tx + h*(tx*ty*R_x-(1+tx**2)*R_y) + h**2*(tx*(3*ty**2 +1)*R_xx -ty*(3*tx**2 +1)*R_xy -ty*(3*tx**2 +1)*R_yx +tx*(3*tx**2 +3)*R_yy))
            return f_tx().diff(p)       #d(tx)/dp

        def Prediction_typrimep():
            def f_ty():                 #Prediction(ty) with p explicitly defined
                h=K*q/p*(1+ (tx)**2 +(ty)**2)**0.5
                return (ty + h*((1+ty**2)*R_x -tx*ty*R_y) + h**2*(ty*(3*(ty**2) +3)*R_xx -tx*(3*ty**2 +1)*R_xy -tx*(3*ty**2 +1)*R_yx +ty*(3*tx**2 +1)*R_yy))
            return f_ty().diff(p)       #d(ty)/dp

        #Symbolic to mathematical
        self.dx_dp  = sym.lambdify((x,y,tx,ty,qp,dz,p,bx), Prediction_xprimep(),  "numpy")
        self.dy_dp  = sym.lambdify((x,y,tx,ty,qp,dz,p,bx), Prediction_yprimep(),  "numpy")
        self.dtx_dp = sym.lambdify((x,y,tx,ty,qp,dz,p,bx), Prediction_txprimep(), "numpy")
        self.dty_dp = sym.lambdify((x,y,tx,ty,qp,dz,p,bx), Prediction_typrimep(), "numpy")
    

class Loop_initialisation:
    
    def __init__(self,j,D,temp_,C_k):
        
        self.D=D
        self.C_k=C_k
        self.x_, self.y_, self.tx_, self.ty_, self.qp_=temp_
        
        if (j==0 or j==57):
            self.dz_=20*self.D       #step size in mm for air gap
            self.material="Air"
            self.b_x=0.0             #set as zero
        else:
            self.dz_=1*self.D        #step size in mm for iron plate
            self.material="Iron"
            self.b_x=1.5             #in Tesla for iron plate


class Bethe_Bloch(Loop_initialisation):
    
    def material_constants(self):            #constants for the BetheBloch equation
        
        if self.material=="Iron":
            self.rho=7.874                   #g/cm^3
            self.ZbA=26.0/55.845             #Z/A
            self.I=286*10**-9                #Mean Excitation Potential in GeV
            #Density correction constants
            self.X_0=-0.0012
            self.X_1=3.15
            self.m_d=2.96
            self.a=0.1468
            self.C0=-4.29
        
        if self.material=="Air":
            self.rho=1.205*10**-3            #g/cm^3
            self.ZbA=0.49919                 #Z/A
            self.I=85.7*10**-9               #Mean Excitation Potential in GeV  
            #Density correction constants
            self.X_0=1.742
            self.X_1=4.28
            self.m_d=3.40
            self.a=0.1091
            self.C0=-10.6
        
    def beta(self,qp_):
        return (q/qp_)/np.sqrt((q/qp_)**2+m**2) 
    def gamma(self,qp_):
        return 1/np.sqrt(1-(self.beta(qp_))**2)              
    def T_max(self,qp_):                   #Maximum energy transfer in a single collision
        return 2*m_e*(self.beta(qp_)*self.gamma(qp_))**2/( 1 + 2*(m_e/m)*np.sqrt(1+(self.beta(qp_)*self.gamma(qp_))**2+(m_e/m)**2) ) 

    def Energyloss(self,qp_):
        
        self.material_constants()
        
        #Density Correction Calculation
        X_d=math.log( self.beta(qp_)*self.gamma(qp_),10)
        if X_d<self.X_0:
            delta=0.0
        if (X_d>self.X_0 and X_d<self.X_1):
            delta=4.6052*X_d + self.C0 + (self.a*((self.X_1-X_d)**self.m_d))
        if X_d>self.X_1:
            delta=4.6052*X_d+self.C0

        dEds= self.rho*0.307075/((self.beta(qp_))**2)*self.ZbA*( 0.5*np.log(2*m_e*(self.beta(qp_)*self.gamma(qp_))**2*self.T_max(qp_)/(self.I**2))-(self.beta(qp_)**2)-delta*0.5) #MeV cm^-1
        
        return dEds


class Prior_Predictions(Bethe_Bloch):                       #Prediction of prior state vectors 
    
    def Prediction_qp(self,tx_,ty_,qp_,dz_):

        if qp_<0:
            qp_=abs(qp_)
        
        dEds =self.Energyloss(qp_)                          #calculation using Bethe Bloch
        
        self.dl =abs(dz_*np.sqrt(1+tx_**2+ty_**2))*self.D   #differential arc length
        
        E_at_j = eid.E[-1]-self.D*abs((dEds*self.dl*10**-4))   #Energy at a j location
        
        if ((E_at_j<m)):
            qp_= q/10                                       #re-initialising p as 10 GeV
            eid.E.append(np.sqrt((q/obj3.X_k[4])**2+m**2))  #save the previous posterior estimate as Energy
        else:    
            qp_ = q/np.sqrt(E_at_j**2-m**2)            
            eid.E.append(E_at_j)
        
        return qp_
    
    def Error(self):        #Error Propagation for prior state vectors
        
        dz_=self.dz_
        dl=self.dl
        D=self.D
        b_x=self.b_x
        
        x_,y_,tx_,ty_,qp_=self.temp_sv
        p_=q/qp_
        

        def fl_1(p_):
            return obj.fl__1(obj.range_l(p_)) #f'(l)
        def fl_2(p_):
            return obj.fl__2(obj.range_l(p_)) #f''(l)
        def fl_3(p_):
            return obj.fl__3(obj.range_l(p_)) #f'''(l)

        #Calculation for Propagator Matrix Elements(Row 5)

        def dqp_dx(tx_,ty_,p_,dl):      #d(q/p)/dx
            return K*(fl_1(p_)+fl_2(p_)*dl)*p_*np.sqrt(1+tx_**2+ty_**2)*dl*(-by)

        def dqp_dy(tx_,ty_,p_,dl,b_x):  #d(q/p)/dy    
            return K*(fl_1(p_)+fl_2(p_)*dl)*p_*np.sqrt(1+tx_**2+ty_**2)*dl*(b_x)

        def dqp_dtx(tx_,ty_,qp_,dz_,dl): #d(q/p)/dtx
            return (fl_1(p_)+fl_2(p_)*dl)*dz_*(tx_/np.sqrt(1+tx_**2+ty_**2))

        def dqp_dty(tx_,ty_,p_,dz_,dl): #d(q/p)/dty
            return  (fl_1(p_)+fl_2(p_)*dl)*dz_*(ty_/np.sqrt(1+tx_**2+ty_**2))

        def dqp_dqp(p_,dl):             #d(q/p)/d(q/p)
            return 1+ (fl_2(p_))/fl_1(p_)*dl + 0.5*fl_3(p_)/fl_1(p_)*dl**2


        #Propagator Matrix
        
        self.F_k =   np.array([[None for j in range (5)] for i in range (5)],   dtype=float)   

        self.F_k[0][0] = obj.dx_dx (x_,y_,tx_,ty_,qp_,dz_,b_x)
        self.F_k[0][1] = obj.dx_dy (x_,y_,tx_,ty_,qp_,dz_,b_x)
        self.F_k[0][2] = obj.dx_dtx(x_,y_,tx_,ty_,qp_,dz_,b_x)
        self.F_k[0][3] = obj.dx_dty(x_,y_,tx_,ty_,qp_,dz_,b_x)
        self.F_k[0][4] = obj.dx_dqp(x_,y_,tx_,ty_,qp_,dz_,b_x)
    
        self.F_k[1][0] = obj.dy_dx (x_,y_,tx_,ty_,qp_,dz_,b_x)
        self.F_k[1][1] = obj.dy_dy (x_,y_,tx_,ty_,qp_,dz_,b_x)
        self.F_k[1][2] = obj.dy_dtx(x_,y_,tx_,ty_,qp_,dz_,b_x)
        self.F_k[1][3] = obj.dy_dty(x_,y_,tx_,ty_,qp_,dz_,b_x)
        self.F_k[1][4] = obj.dy_dqp(x_,y_,tx_,ty_,qp_,dz_,b_x)
    
        self.F_k[2][0] = obj.dtx_dx (tx_,ty_,qp_,dz_,b_x)
        self.F_k[2][1] = obj.dtx_dy (tx_,ty_,qp_,dz_,b_x)
        self.F_k[2][2] = obj.dtx_dtx(tx_,ty_,qp_,dz_,b_x)
        self.F_k[2][3] = obj.dtx_dty(tx_,ty_,qp_,dz_,b_x)
        self.F_k[2][4] = obj.dtx_dqp(tx_,ty_,qp_,dz_,b_x)
    
        self.F_k[3][0] = obj.dty_dx (tx_,ty_,qp_,dz_,b_x)
        self.F_k[3][1] = obj.dty_dy (tx_,ty_,qp_,dz_,b_x)
        self.F_k[3][2] = obj.dty_dtx(tx_,ty_,qp_,dz_,b_x)
        self.F_k[3][3] = obj.dty_dty(tx_,ty_,qp_,dz_,b_x)
        self.F_k[3][4] = obj.dty_dqp(tx_,ty_,qp_,dz_,b_x)

        self.F_k[4][0] = dqp_dx (tx_,ty_,p_,dl)
        self.F_k[4][1] = dqp_dy (tx_,ty_,p_,dl,b_x)
        self.F_k[4][2] = dqp_dtx(tx_,ty_,p_,dz_,dl)
        self.F_k[4][3] = dqp_dty(tx_,ty_,p_,dz_,dl)
        self.F_k[4][4] = dqp_dqp(p_,dl)
        #****************************************************************************
        #Random Error
        
        #Effects of Multiple Scattering
        def CMS(qp_,dl):    #Highland-Lynch-Dahl variance formula
            
            Z     = 26.0
            p_    = q/qp_
            l_rad = 17.57      #radiation length for iron im mm
            ls    = l_rad*((Z+1)/Z)*np.log(287*Z**(-1/2))/np.log(159*Z**(-1/3))  

            return (0.015/(self.beta(qp_)*p_))**2*(dl/ls)

        def cov_txtx(tx_,ty_,qp_,dl):                 
            return (1+tx_**2)*(1+tx_**2+ty_**2)*CMS(qp_,dl)
        
        def cov_tyty(tx_,ty_,qp_,dl):
            return (1+ty_**2)*(1+tx_**2+ty_**2)*CMS(qp_,dl)
        
        def cov_txty(tx_,ty_,qp_,dl):
            return tx_*ty_*(1+tx_**2+ty_**2)*CMS(qp_,dl)

        
        #Effects of Energy loss Straggling 
        def xi(qp_):                     #Mean energy loss in GeV
            
            rho = 7.874
            ZbA = 26.0/55.845 
            d   = 1                       #thickness of the medium in mm
            
            return (0.1534*ZbA/self.beta(qp_)**2)*rho*d*10**-4

        def sig2_E(qp_):  #variance of the energy loss fluctuation distribution 
            
            k= xi(qp_)/(self.T_max(qp_))   #defines the nature of the energyloss fluctuation

            if k>0.01:                                         
                return  xi(qp_)*self.T_max(qp_)*(1-(self.beta(qp_)**2)/2) #Gaussain distribution in GeV^2
            
            if k<0.01:                                         
                return  (15.76*xi(qp_))**2                                  #Vavilov distribution in GeV^2
        
        #Random Error Matrix
        self.Q_l =   np.array([[None for j in range (5)] for i in range (5)],   dtype=float)  #Random Error Matrix initialisation
        
        if (self.material=="Air"):
            self.Q_l =[[0 for j in range (5)] for i in range (5)]        #No random error in air
        
        if (self.material=="Iron"):
        
            self.Q_l[0][0] = cov_txtx(tx_,ty_,qp_,dl)*(dl**3)/3
            self.Q_l[0][1] = cov_txty(tx_,ty_,qp_,dl)*(dl**3)/3
            self.Q_l[0][2] = cov_txtx(tx_,ty_,qp_,dl)*(dl**2)*D/2
            self.Q_l[0][3] = cov_txty(tx_,ty_,qp_,dl)*(dl**2)*D/2
            self.Q_l[0][4] = -1*obj.dx_dp(x_,y_,tx_,ty_,qp_,dz_,p_,b_x)*sig2_E(qp_)*q*(eid.E[-1]**2)/(p_**4)
        
            self.Q_l[1][0] = cov_txty(tx_,ty_,qp_,dl)*(dl**3)/3
            self.Q_l[1][1] = cov_tyty(tx_,ty_,qp_,dl)*(dl**3)/3
            self.Q_l[1][2] = cov_txty(tx_,ty_,qp_,dl)*(dl**2)*D/2
            self.Q_l[1][3] = cov_tyty(tx_,ty_,qp_,dl)*(dl**2)*D/2
            self.Q_l[1][4] = -1*obj.dy_dp(x_,y_,tx_,ty_,qp_,dz_,p_,b_x)*sig2_E(qp_)*q*(eid.E[-1]**2)/(p_**4)

            self.Q_l[2][0] = cov_txtx(tx_,ty_,qp_,dl)*(dl**2)*D/2
            self.Q_l[2][1] = cov_txty(tx_,ty_,qp_,dl)*(dl**2)*D/2
            self.Q_l[2][2] = cov_txtx(tx_,ty_,qp_,dl)*dl
            self.Q_l[2][3] = cov_txty(tx_,ty_,qp_,dl)*dl
            self.Q_l[2][4] = -1*obj.dtx_dp(x_,y_,tx_,ty_,qp_,dz_,p_,b_x)*sig2_E(qp_)*q*(eid.E[-1]**2)/(p_**4)
        
            self.Q_l[3][0] = cov_txty(tx_,ty_,qp_,dl)*(dl**2)*D/2
            self.Q_l[3][1] = cov_tyty(tx_,ty_,qp_,dl)*(dl**2)*D/2
            self.Q_l[3][2] = cov_txtx(tx_,ty_,qp_,dl)*dl
            self.Q_l[3][3] = cov_tyty(tx_,ty_,qp_,dl)*dl
            self.Q_l[3][4] = -1*obj.dty_dp(x_,y_,tx_,ty_,qp_,dz_,p_,b_x)*sig2_E(qp_)*q*(eid.E[-1]**2)/(p_**4)

            self.Q_l[4][0] = -1*obj.dx_dp(x_,y_,tx_,ty_,qp_,dz_,p_,b_x)*sig2_E(qp_)*q*(eid.E[-1]**2)/(p_**4)
            self.Q_l[4][1] = -1*obj.dy_dp(x_,y_,tx_,ty_,qp_,dz_,p_,b_x)*sig2_E(qp_)*q*(eid.E[-1]**2)/(p_**4)
            self.Q_l[4][2] = -1*obj.dtx_dp(x_,y_,tx_,ty_,qp_,dz_,p_,b_x)*sig2_E(qp_)*q*(eid.E[-1]**2)/(p_**4)
            self.Q_l[4][3] = -1*obj.dty_dp(x_,y_,tx_,ty_,qp_,dz_,p_,b_x)*sig2_E(qp_)*q*(eid.E[-1]**2)/(p_**4)
            self.Q_l[4][4] = eid.E[-1]**2/(p_**6)*sig2_E(qp_)

            self.Q_l=np.array(self.Q_l, dtype=float)
        
        #Covariance Calculation for prior estimate

        self.C_k = self.F_k@self.C_k@((self.F_k).T) + self.Q_l
        self.C_k = np.array(self.C_k, dtype=float) 
        
        return self.F_k,self.C_k

        
    def f_predictions(self):   
        
        x_  = obj.Prediction_x (self.x_,self.y_,self.tx_,self.ty_,self.qp_,self.dz_,self.b_x)       
        y_  = obj.Prediction_y (self.x_,self.y_,self.tx_,self.ty_,self.qp_,self.dz_,self.b_x)       
        tx_ = obj.Prediction_tx(self.x_,self.y_,self.tx_,self.ty_,self.qp_,self.dz_,self.b_x)       
        ty_ = obj.Prediction_ty(self.x_,self.y_,self.tx_,self.ty_,self.qp_,self.dz_,self.b_x)       
        qp_ = self.Prediction_qp(self.tx_,self.ty_,self.qp_,self.dz_)
        
        self.temp_sv = x_, y_, tx_, ty_, qp_
        
        self.F_k, self.C_k=self.Error()
        return self.temp_sv,self.C_k,self.F_k


class Kalman_Filter:
    
    def Kalman(self,i,X_k,C_k,F_k):       
        
        V_k = np.array([[0,0],[0,0]])                          #measurement error matrix
        H_k = np.array([[1,0,0,0,0],[0,1,0,0,0]])              #projector matrix
        I   = np.identity(5,dtype=float)
        
        self.X_k=np.array(X_k)
        
        #Kalman Gain matrix
        K_k = C_k @(H_k.T)@ (np.linalg.inv(np.array((H_k@ C_k@(H_k.T)+V_k),dtype=float)))
        
        #Posterior estimate for state vector       
        self.X_k = (self.X_k.T + K_k@( (eid.M_k[i,:].T) - (H_k@(self.X_k.T)) ) ).T
        
        #Covariance matrix for posterior estimate
        C_k_k = (I-(K_k@H_k)) @ C_k @ ( (I-(K_k@H_k)).T )   +   K_k@V_k@(K_k.T)
        
        eid.E.append(np.sqrt((q/self.X_k[4])**2+m**2))
        
        return self.X_k,C_k_k

#**********************************************************************************************
#Main starts here
#**********************************************************************************************


#filename="dataGeV/"+str(TrueEnergy)+"GeVmu+100eve.txt"

filename="dataGeV/10GeVmu+100eve.txt"

issue=[]
#residual_dist,mean_per_event,sigma_per_event=[],[],[] #for residual distribution plots
Pulldist=[] #for pull plots

#print("\n\nTrue Energy of the muon:",TrueEnergy," GeV")

df=pd.read_csv(filename,sep="\t") 
df1=np.array(df)

num_of_events=max(df1[:,0])
print("\nNo. of events:",(num_of_events+1))

num_of_iter=2                          #number of iterations for filtering+smoothing
print("\nIterations set:",num_of_iter,'\n\n')

obj=Symbolic_expressions()        
obj.Prediction_expressions()

est_start=time.time()        
for event_index in range(num_of_events+1):
    try:
        #print("\n\n------------------------------------Event No.",event_index+1,"------------------------------------")    
        
        eid = Datahandling(df,event_index)

        #print('\nNo.of layers:',eid.num_of_layers,'\n')
        
        prior_sv, Cov_matrix = eid.Initialisation()
        
        for iter_index in range(num_of_iter): 
            
            P_F_lyr=[]
            
            D=1
            for i in range(1,len(eid.data_per_eid)):   #Forward loop for 150 (air+iron+air) combo                
                for j in range(58):                  #[ air('0') + iron('1'-'56') + air('57') ]            
                    obj2 = Prior_Predictions(j, D, prior_sv, Cov_matrix) 
                    prior_sv, Cov_matrix, Prop_matrix = obj2.f_predictions()
                
                obj3 = Kalman_Filter()
                post_sv, Cov_matrix = obj3.Kalman(i,prior_sv,Cov_matrix,Prop_matrix)
                prior_sv = post_sv
            
            if iter_index==(num_of_iter-1):   #saving momentum of last hit for histogram plotting 
                    P_F_lyr.append((q/post_sv[4]))                         
            
            D=-1
            for i in reversed(range(len(eid.data_per_eid)-1)):   #Reverse loop for 150 (air+iron+air) combo
                for j in reversed(range(58)):           #[ air('57') + iron('56'-'1') + air('0')]
                    obj2 = Prior_Predictions(j, D, prior_sv, Cov_matrix) 
                    prior_sv, Cov_matrix, Prop_matrix = obj2.f_predictions()
                
                obj3 = Kalman_Filter()
                post_sv, Cov_matrix = obj3.Kalman(i, prior_sv, Cov_matrix, Prop_matrix)
                prior_sv = post_sv

                if iter_index==(num_of_iter-1):   #saving momentum for histogram plotting 
                    P_F_lyr.append((q/post_sv[4])) 
           
            #print('    Incident energy of muon ( iteration',iter_index+1,') = \t',np.sqrt((q/post_sv[4])**2+m**2))      
        
        #print('\nEnergy of the muon for event',event_index+1,' = ',np.sqrt((q/post_sv[4])**2+m**2))
        
        eid.true_sv[4]=q/np.sqrt(TrueEnergy**2-m**2)
        
        pull_value = eid.true_sv-post_sv
        
        pull_x  = pull_value[0]/np.sqrt(Cov_matrix[0][0])
        pull_y  = pull_value[1]/np.sqrt(Cov_matrix[1][1])
        pull_tx = pull_value[2]/np.sqrt(Cov_matrix[2][2])
        pull_ty = pull_value[3]/np.sqrt(Cov_matrix[3][3])
        pull_qp = pull_value[4]/np.sqrt(Cov_matrix[4][4])
        
        Pulldist.append(pull_qp)
        
        if event_index==0:
            est=time.time()
            est_time=(num_of_events+1)*(est - est_start)/60
            est_time=int(est_time)
            print("Estimated runtime of the program is ~ {} mins\n".format(est_time))
        
        #P_F_lyr = P_F_lyr[::-1]       #since appending was done in reverse
        #diff_p = ( (eid.data_per_eid[:,19]**2+eid.data_per_eid[:,18]**2+eid.data_per_eid[:,20]**2)**0.5 )- np.array(P_F_lyr) #difference in momentum per layer in GeV
        #for i in range(len(eid.data_per_eid)):   
        #    residual_dist.append(diff_p[i])
        
        #mean_per_event.append(np.mean(np.array(diff_p)))      
        #sigma_per_event.append(np.std(np.array(diff_p)))      
    
    except:
        issue.append(event_index)       #failed events

print("\n\n------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")

#Pull plots
plt.style.use('ggplot')
mean=np.mean(np.array(Pulldist))      
print("Mean of pull distribution of momentum:","{:.4f}".format(mean))
sigma=np.std(np.array(Pulldist))     
print("Sigma:","{:.4f}".format(sigma))    
count, bins, p_=plt.hist(Pulldist,bins=50,density=True)

plt.title("Pull of q/P for %i GeV" %TrueEnergy)
plt.xlabel("$(q/P_{true}$- $q/P_{Kalman})$/ \u03C3(q/P)")



#Residual plot

#mean_=np.mean(np.array(mean_per_event))          # for all events pertaining to a particular energy regime
#print("Mean of the Residual distribution of momentum:","{:.4f}".format(mean))
#sigma=np.std(np.array(sigma_per_event))          # for all events pertaining to a particular energy regime
#print("Sigma:","{:.4f}".format(sigma))
#residual_dist=np.array(residual_dist)
#count, bins, p_=plt.hist(residual_dist,bins=3000,density=True)
#plt.title("Residual distribution of P for %i GeV" %TrueEnergy)
#plt.xlabel("$P_{true}$- $P_{Kalman}$ (GeV/c)")

max_bin_height=0
for item in p_:
    if max_bin_height<item.get_height():
        max_bin_height=item.get_height()

plt.plot(bins, max_bin_height*1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mean)**2 / (2 * sigma**2) ),linewidth=2, color='black') #gaussian curve plot

plt.axvline(x=mean,linestyle='dashed',color='black')
plt.axvline(x=mean+sigma,linestyle='dotted',color='black')
plt.axvline(x=mean-sigma,linestyle='dotted',color='black')

plt.ylabel("Frequency")
plt.xlim((mean-3*sigma),(mean+3*sigma))
#plt.xlim(-2,8)
plt.ylim(0,1.1*max_bin_height)

end = time.time()  
tot_time=(end - start)/60
tot_time="{:.2f}".format(tot_time)

print("\nEvent ids with issue running the code",issue)
print(f"Runtime of the program is {tot_time} mins")
print("\n\n------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")

#plt.savefig("plots/{}GeV_{}eve_{}iter.png".format(TrueEnergy,num_of_events+1,num_of_iter))
plt.show()
