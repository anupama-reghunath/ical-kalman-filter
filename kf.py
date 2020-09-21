""" A python based implementation of Kalman filter  
    Author: Anupama Reghunath    """

import time              #for code execution time
import math              
import numpy as np       #for mathematical calculations
import matplotlib.pyplot as plt #for plotting
import pandas as pd      #for data frame creation
import sympy as sym      #for symbolic calculations
from scipy.interpolate import CubicSpline  #fitting
#import sys
#sys.stdout = open('output.txt', 'w') #printing output onto output.txt

start = time.time()            

#***************************************************************
number_of_iterations=4                         #number of iterations for filtering+smoothing

#READING MEASURED DATA

df=pd.read_csv("dataGeV/mu+10Gev10eve.txt",sep="\t") #muon+ ;10 events ;10 GeV
#df=pd.read_csv("dataGeV/mu+5GeV100eve.txt",sep="\t")  #muon+ ;100 events; 5 GeV

df1=np.array(df)

eid=df1[:,0]
number_of_events=max(eid)
print("No. of events:",(number_of_events+1))
print("Iterations set:",number_of_iterations)
mu_event,sigma_event=[],[]




#V_k=[[0.1/12,0],[0,0.1/12]]     
V_k=[[0,0],[0,0]]                          #measurement error matrix
#V_k=[[1,0],[0,1]]               
V_k=np.array(V_k)
H_k=np.array([[1,0,0,0,0],[0,1,0,0,0]])    #projector matrix


#***************************************************************
#CONSTANTS

m=0.105658                      #mass of the particle(muon) in GeV/c^2
m_e=0.511*10**-3                #mass of electron in GeV/c^2
K=0.299792458*10**-3            #GeV c-1 T-1 mm-1
q=0.303                         #+1 C in natural units               


by=0                            #magnetic field along y axis in tesla


#cov_C =10**6*np.identity(5,  dtype=np.float)                #Initial value for covariance
#cov_C =np.array(cov_C)




#***************************************************************
#PREDICTION

x,y,tx,ty,qp,dz,p,bx=sym.symbols('x y tx ty qp dz p bx') #symbols in prediction

#integrals

def sx():
    return(0.5*bx*dz**2)
def sy():
    return(0.5*by*dz**2)
def sxx():
    return(bx**2*dz**3 /6)
def syy():
     return(by**2*dz**3 /6)
def sxy():
     return(bx*by*dz**3 /6)
def syx():
     return(sxy())
def rx():
     return(bx*dz)
def ry():
    return(by*dz)
def rxy():
     return(0.5*bx*by*dz**2)
def rxx():
     return(0.5*bx*bx*dz**2)
def ryy():
     return(0.5*by*by*dz**2)
def ryx(): 
     return(rxy())
def h():
    return(K*qp*(1+ (tx)**2 +(ty)**2)**0.5)

#Defining the Prediction equations

def Prediction_xe():
    return (x  + tx*dz + h()*(tx*ty*sx()-(1+tx**2)*sy()) + h()**2*(tx*(3*ty**2 +1)*sxx() -ty*(3*tx**2 +1)*sxy() -ty*(3*tx**2 +1)*syx() +tx*(3*tx**2 +3)*syy()))
def Prediction_ye():
    return(y  + ty*dz + h()*((1+ty**2)*sx() -tx*ty*sy()) + h()**2*(ty*(3*ty**2 +3)*sxx() -tx*(3*ty**2 +1)*sxy() -tx*(3*ty**2 +1)*syx() +ty*(3*tx**2 +1)*syy()))
def Prediction_txe():
    return(tx + h()*(tx*ty*rx()-(1+tx**2)*ry()) + h()**2*(tx*(3*ty**2 +1)*rxx() -ty*(3*tx**2 +1)*rxy() -ty*(3*tx**2 +1)*ryx() +tx*(3*tx**2 +3)*ryy()))
def Prediction_tye():
    return(ty + h()*((1+ty**2)*rx() -tx*ty*ry()) + h()**2*(ty*(3*(ty**2) +3)*rxx() -tx*(3*ty**2 +1)*rxy() -tx*(3*ty**2 +1)*ryx() +ty*(3*tx**2 +1)*ryy()))

#Energy Loss Prediction

def beta(qp_e):
    return (q/qp_e)/np.sqrt((q/qp_e)**2+m**2) 
def gamma(qp_e):
    return 1/np.sqrt(1-(beta(qp_e))**2)              
def f_xd(qp_e):                 #required for Density Correction for Bethe Bloche Formula
    return math.log(beta(qp_e)*gamma(qp_e),10)
def Tmax(qp_e):
    return 2*m_e*(beta(qp_e)*gamma(qp_e))**2/( 1 + 2*(m_e/m)*np.sqrt(1+(beta(qp_e)*gamma(qp_e))**2+(m_e/m)**2) ) # Max Kinetic Eenrgy   

def EnergylossIron(qp_e):
    rho=7.874
    ZbA=26.0/55.845             #Z/A
    I=286*10**-9                #in GeV
    xd0=-0.0012
    xd1=3.15
    md=2.96
    a=0.1468
    C0=-4.29
    if qp_e<0:
        qp_e=-qp_e
     #print(qp_e,f_b(qp_e),f_g(qp_e),f_xd(qp_e))
    xd=f_xd(qp_e)
    if xd<xd0:
       delta=0.0
    if xd>xd0 and xd<xd1:
       delta=4.6052*xd+C0+(a*((xd1-xd)**md))
    if xd>xd1:
       delta=4.6052*xd+C0
    
    dEds= rho*0.307075/((beta(qp_e))**2)*ZbA*( 0.5*np.log(2*m_e*(beta(qp_e)*gamma(qp_e))**2*Tmax(qp_e)/(I**2))-(beta(qp_e)**2)-delta*0.5) #Bethe Bloch Formula in MeVcm**2/g        
    
    return dEds
   
def EnergylossAir(qp_e):
    rho=1.205*10**-3
    ZbA=0.49919                 #Z/A
    I=85.7*10**-9               #in GeV  
    xd0=1.742
    xd1=4.28
    md=3.40
    a=0.1091
    C0=-10.6
    if qp_e<0:
        qp_e=-qp_e
    #print(qp_e,f_b(qp_e),f_g(qp_e),f_xd(qp_e))
    xd=f_xd(qp_e)
    if xd<xd0:
        delta=0.0
    if xd>xd0 and xd<xd1:
        delta=4.6052*xd+C0+(a*((xd1-xd)**md))
    if xd>xd1:
        delta=4.6052*xd+C0   

    dEds= rho*0.307075/((beta(qp_e))**2)*ZbA*( 0.5*np.log(2*m_e*(beta(qp_e)*gamma(qp_e))**2*Tmax(qp_e)/(I**2))-(beta(qp_e)**2)-delta*0.5) #Bethe Bloch Formula in MeVcm**2/g                     

    return dEds


#Converting symbolic to mathematical

f_x = sym.lambdify((x,y,tx,ty,qp,dz,bx), Prediction_xe(), "numpy")
f_y = sym.lambdify((x,y,tx,ty,qp,dz,bx), Prediction_ye(), "numpy")
f_tx= sym.lambdify((x,y,tx,ty,qp,dz,bx), Prediction_txe(),"numpy")
f_ty= sym.lambdify((x,y,tx,ty,qp,dz,bx), Prediction_tye(),"numpy")

#***************************************************************
#PROPAGATION OF ERRORS

#Calculation for Propagator Matrix Elements(Rows 1,2,3)
def Prediction_xprimex():
    return Prediction_xe().diff(x)
def Prediction_xprimey():
    return Prediction_xe().diff(y)
def Prediction_xprimetx():
    return Prediction_xe().diff(tx)
def Prediction_xprimety():
    return Prediction_xe().diff(ty)
def Prediction_xprimeqbp():
    return Prediction_xe().diff(qp)

def Prediction_yprimex():
    return Prediction_ye().diff(x)
def Prediction_yprimey():
    return Prediction_ye().diff(y)
def Prediction_yprimetx():
    return Prediction_ye().diff(tx)
def Prediction_yprimety():
    return Prediction_ye().diff(ty)
def Prediction_yprimeqbp():
    return Prediction_ye().diff(qp)

def Prediction_txprimex():
    return Prediction_txe().diff(x)
def Prediction_txprimey():
    return Prediction_txe().diff(y)
def Prediction_txprimetx():
    return Prediction_txe().diff(tx)
def Prediction_txprimety():
    return Prediction_txe().diff(ty)
def Prediction_txprimeqbp():
    return Prediction_txe().diff(qp)

def Prediction_typrimex():
    return Prediction_tye().diff(x)
def Prediction_typrimey():
    return Prediction_tye().diff(y)
def Prediction_typrimetx():
    return Prediction_tye().diff(tx)
def Prediction_typrimety():
    return Prediction_tye().diff(ty)
def Prediction_typrimeqbp():
    return Prediction_tye().diff(qp)

#Converting symbolic to mathematical

f_xprimex   = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_xprimex(),   "numpy")
f_xprimey   = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_xprimey(),   "numpy")
f_xprimetx  = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_xprimetx(),  "numpy")
f_xprimety  = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_xprimety(),  "numpy")
f_xprimeqbp = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_xprimeqbp(), "numpy")    

f_yprimex   = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_yprimex(),   "numpy")
f_yprimey   = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_yprimey(),   "numpy")
f_yprimetx  = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_yprimetx(),  "numpy")
f_yprimety  = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_yprimety(),  "numpy")
f_yprimeqbp = sym.lambdify( (x,y,tx,ty,qp,dz,bx), Prediction_yprimeqbp(), "numpy")    

f_txprimex  = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_txprimex(),  "numpy")
f_txprimey  = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_txprimey(),  "numpy")
f_txprimetx = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_txprimetx(), "numpy")
f_txprimety = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_txprimety(), "numpy")
f_txprimeqbp= sym.lambdify( (tx,ty,qp,dz,bx), Prediction_txprimeqbp(),"numpy")    

f_typrimex  = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_typrimex(),  "numpy")
f_typrimey  = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_typrimey(),  "numpy")
f_typrimetx = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_typrimetx(), "numpy")
f_typrimety = sym.lambdify( (tx,ty,qp,dz,bx), Prediction_typrimety(), "numpy")
f_typrimeqbp= sym.lambdify( (tx,ty,qp,dz,bx), Prediction_typrimeqbp(),"numpy")    

#***************************************************************
#RANGE-MOMENTUM RELATION FROM CURVE FITTING FOR ERROR IN Q/P

#obtaining data points for Cubic Spline interpolation
cb =pd.read_csv("muon-iron-energyLossTable3.txt",sep=" ") 
cb1=np.array(cb)

pp =cb1[:,3]*10**-3             #converting MeV/c into GeV/c
l_r=cb1[:,10]*10/7.874          #in q/cm^2 to mm (/iron density)
ene_r=cb1[:,9]*7.874

EnergylossIron_CubicSpline = CubicSpline(pp,ene_r,bc_type='natural')     #Alternate equation for Energy-loss in Iron

range_l = CubicSpline(pp,l_r,bc_type='natural')     #range as a function of p

fl=CubicSpline(l_r,pp,bc_type='natural')            #p as a function of range


#Derivatives calculated using Central Difference formula of third order Richardson's Extrapolation Method

hh=0.001                        #stepsize

def fl_1(pe):                 #f'(l)
    return (fl(pe+hh)-fl(pe-hh))/(2*hh)

def fl_2(pe):                 #f''(l)
    return (fl(pe+2*hh)-2*fl(pe)+fl(pe-2*hh))/(2*hh)**2

def fl_3(pe):                 #f'''(l)
    return (fl(pe+3*hh)-3*fl(pe+hh)+3*fl(pe-hh)-fl(pe-3*hh))/(2*hh)**3


#Calculation for Propagator Matrix Elements(Row 4)
def f_qbpprimeqbp(qp_e,dl):
    pe=q/qp_e
    return 1+ (fl_2(pe))/fl_1(pe)*dl + 0.5*fl_3(pe)/fl_1(pe)*dl**2

def f_qbpprimex(tx_e,ty_e,qp_e,dl):
    pe=q/qp_e
    return K*(fl_1(pe)+fl_2(pe)*dl)*fl(pe)*np.sqrt(1+tx_e**2+ty_e**2)*dl*(-by)

def f_qbpprimey(tx_e,ty_e,qp_e,dl,b_x):
    pe=q/qp_e
    return K*(fl_1(pe)+fl_2(pe)*dl)*fl(pe)*np.sqrt(1+tx_e**2+ty_e**2)*dl*(b_x)

def f_qbpprimetx(tx_e,ty_e,qp_e,dz_e,dl):
    pe=q/qp_e
    return (fl_1(pe)+fl_2(pe)*dl)*dz_e*(tx_e/np.sqrt(1+tx_e**2+ty_e**2))

def f_qbpprimety(tx_e,ty_e,qp_e,dz_e,dl):
    pe=q/qp_e
    return  (fl_1(pe)+fl_2(pe)*dl)*dz_e*(ty_e/np.sqrt(1+tx_e**2+ty_e**2))


 

def Propagator(x_e,y_e,tx_e,ty_e,qp_e,dz_e,dl,b_x):
    
    global F_k    
    
    F_k[0][0] = f_xprimex  (x_e,y_e,tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[0][1] = f_xprimey  (x_e,y_e,tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[0][2] = f_xprimetx (x_e,y_e,tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[0][3] = f_xprimety (x_e,y_e,tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[0][4] = f_xprimeqbp(x_e,y_e,tx_e,ty_e,qp_e,dz_e,b_x)
   
    F_k[1][0] = f_yprimex  (x_e,y_e,tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[1][1] = f_yprimey  (x_e,y_e,tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[1][2] = f_yprimetx (x_e,y_e,tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[1][3] = f_yprimety (x_e,y_e,tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[1][4] = f_yprimeqbp(x_e,y_e,tx_e,ty_e,qp_e,dz_e,b_x)
   
    F_k[2][0] = f_txprimex  (tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[2][1] = f_txprimey  (tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[2][2] = f_txprimetx (tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[2][3] = f_txprimety (tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[2][4] = f_txprimeqbp(tx_e,ty_e,qp_e,dz_e,b_x)
   
    F_k[3][0] = f_typrimex  (tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[3][1] = f_typrimey  (tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[3][2] = f_typrimetx (tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[3][3] = f_typrimety (tx_e,ty_e,qp_e,dz_e,b_x)
    F_k[3][4] = f_typrimeqbp(tx_e,ty_e,qp_e,dz_e,b_x)

    F_k[4][0] = f_qbpprimex  (tx_e,ty_e,qp_e,dl)
    F_k[4][1] = f_qbpprimey  (tx_e,ty_e,qp_e,dl,b_x)
    F_k[4][2] = f_qbpprimetx (tx_e,ty_e,qp_e,dz_e,dl)
    F_k[4][3] = f_qbpprimety (tx_e,ty_e,qp_e,dz_e,dl)
    F_k[4][4] = f_qbpprimeqbp(qp_e,dl)

    
    return ()

#***************************************************************
#RANDOM ERROR CALCULATION

def CMS(qp_e,dl):    #Highland-Lynch-Dahl variance formula
        
    Z= 26.0
    l_rad=17.57 #radiation length for iron im mm
    
    ls = l_rad*((Z+1)/Z)*(287*Z**(-1/2))/(159*Z**(-1/3))  
    pe = q/qp_e
    return (0.015/(beta(qp_e)*pe))**2*(dl/ls)

def cov_txtx(tx_e,ty_e,qp_e,dl):                 
    return (1+tx_e**2)*(1+tx_e**2+ty_e**2)*CMS(qp_e,dl)
def cov_tyty(tx_e,ty_e,qp_e,dl):
    return (1+ty_e**2)*(1+tx_e**2+ty_e**2)*CMS(qp_e,dl)
def cov_txty(tx_e,ty_e,qp_e,dl):
    return tx_e*ty_e*(1+tx_e**2+ty_e**2)*CMS(qp_e,dl)

#Calculation for Multiple Scattering Elements in the Random error Matrix

def Prediction_xprimep():
    def P_x():                  #Prediction_x with p explicitly defined
        return (x  + tx*dz + (K*q/p*(1+ (tx)**2 +(ty)**2)**0.5)*(tx*ty*sx()-(1+tx**2)*sy()) + (K*q/p*(1+ (tx)**2 +(ty)**2)**0.5)**2*(tx*(3*ty**2 +1)*sxx() -ty*(3*tx**2 +1)*sxy() -ty*(3*tx**2 +1)*syx() +tx*(3*tx**2 +3)*syy()))  
    return P_x().diff(p)

def Prediction_yprimep():
    def P_y():                  #Prediction_y with p explicitly defined
        return(y  + ty*dz + (K*q/p*(1+ (tx)**2 +(ty)**2)**0.5)*((1+ty**2)*sx() -tx*ty*sy()) + (K*q/p*(1+ (tx)**2 +(ty)**2)**0.5)**2*(ty*(3*ty**2 +3)*sxx() -tx*(3*ty**2 +1)*sxy() -tx*(3*ty**2 +1)*syx() +ty*(3*tx**2 +1)*syy()))
    return P_y().diff(p)

def Prediction_txprimep():
    def P_tx():                 #Prediction_tx with p explicitly defined
        return(tx + (K*q/p*(1+ (tx)**2 +(ty)**2)**0.5)*(tx*ty*rx()-(1+tx**2)*ry()) + (K*q/p*(1+ (tx)**2 +(ty)**2)**0.5)**2*(tx*(3*ty**2 +1)*rxx() -ty*(3*tx**2 +1)*rxy() -ty*(3*tx**2 +1)*ryx() +tx*(3*tx**2 +3)*ryy()))
    return P_tx().diff(p)

def Prediction_typrimep():
    def P_ty():                 #Prediction_x with p explicitly defined
        return(ty + (K*q/p*(1+ (tx)**2 +(ty)**2)**0.5)*((1+ty**2)*rx() -tx*ty*ry()) + (K*q/p*(1+ (tx)**2 +(ty)**2)**0.5)**2*(ty*(3*(ty**2) +3)*rxx() -tx*(3*ty**2 +1)*rxy() -tx*(3*ty**2 +1)*ryx() +ty*(3*tx**2 +1)*ryy()))
    return P_ty().diff(p)


cov_xqbp = sym.lambdify((x,y,tx,ty,qp,dz,p,bx), Prediction_xprimep(),  "numpy")
cov_yqbp = sym.lambdify((x,y,tx,ty,qp,dz,p,bx), Prediction_yprimep(),  "numpy")
cov_txqbp= sym.lambdify((x,y,tx,ty,qp,dz,p,bx), Prediction_txprimep(), "numpy")
cov_tyqbp= sym.lambdify((x,y,tx,ty,qp,dz,p,bx), Prediction_typrimep(), "numpy")


#Calculation for Energy loss Struggling Elements in the Random error Matrix

def xi(qp_e):                                               #Mean energy loss in GeV
    
    rho = 7.874
    ZbA = 26.0/55.845 
    d = 1 #56?                      #thickness of the medium in mm
    return (0.1534*q**2*ZbA/beta(qp_e)**2)*rho*d*10**-4

def sig2E(qp_e):                                            #var of the Gaussain distribution in GeV^2
    return  xi(qp_e)*Tmax(qp_e)*10**-3*(1-(beta(qp_e)**2)/2)


def k_para(qp_e):
    return xi(qp_e)/Tmax(qp_e)

#Random Error Matrix definition

def RandomError(x_e,y_e,tx_e,ty_e,qp_e,dz_e,dl,D,material):                #Random Error Matrix


    global Q_l
        
    pe=q/qp_e

    if (material=="Air"):
        Q_l =[[0 for j in range (5)] for i in range (5)]
    
    else:
        b_x=1.5
        Q_l[0][0]= cov_txtx(tx_e,ty_e,qp_e,dl)*(dl**3)/3
        Q_l[0][1]= cov_txty(tx_e,ty_e,qp_e,dl)*(dl**3)/3
        Q_l[0][2]= cov_txtx(tx_e,ty_e,qp_e,dl)*(dl**2)*D/2
        Q_l[0][3]= cov_txty(tx_e,ty_e,qp_e,dl)*(dl**2)*D/2
        Q_l[0][4]=-cov_xqbp(x_e,y_e,tx_e,ty_e,qp_e,dz_e,pe,b_x)*sig2E(qp_e)*q*(E[-1]**2)/(pe**4)
    
        Q_l[1][0]=cov_txty(tx_e,ty_e,qp_e,dl)*(dl**3)/3
        Q_l[1][1]=cov_tyty(tx_e,ty_e,qp_e,dl)*(dl**3)/3
        Q_l[1][2]=cov_txty(tx_e,ty_e,qp_e,dl)*(dl**2)*D/2
        Q_l[1][3]=cov_tyty(tx_e,ty_e,qp_e,dl)*(dl**2)*D/2
        Q_l[1][4]=-cov_yqbp(x_e,y_e,tx_e,ty_e,qp_e,dz_e,pe,b_x)*sig2E(qp_e)*q*(E[-1]**2)/(pe**4)

        Q_l[2][0]=cov_txtx(tx_e,ty_e,qp_e,dl)*(dl**2)*D/2
        Q_l[2][1]=cov_txty(tx_e,ty_e,qp_e,dl)*(dl**2)*D/2
        Q_l[2][2]=cov_txtx(tx_e,ty_e,qp_e,dl)*dl
        Q_l[2][3]=cov_txty(tx_e,ty_e,qp_e,dl)*dl
        Q_l[2][4]=-cov_txqbp(x_e,y_e,tx_e,ty_e,qp_e,dz_e,pe,b_x)*sig2E(qp_e)*q*(E[-1]**2)/(pe**4)
    
        Q_l[3][0]=cov_txty(tx_e,ty_e,qp_e,dl)*(dl**2)*D/2
        Q_l[3][1]=cov_tyty(tx_e,ty_e,qp_e,dl)*(dl**2)*D/2
        Q_l[3][2]=cov_txtx(tx_e,ty_e,qp_e,dl)*dl
        Q_l[3][3]=cov_tyty(tx_e,ty_e,qp_e,dl)*dl
        Q_l[3][4]=-cov_tyqbp(x_e,y_e,tx_e,ty_e,qp_e,dz_e,pe,b_x)*sig2E(qp_e)*q*(E[-1]**2)/(pe**4)

        Q_l[4][0]=-cov_xqbp(x_e,y_e,tx_e,ty_e,qp_e,dz_e,pe,b_x)*sig2E(qp_e)*q*(E[-1]**2)/(pe**4)
        Q_l[4][1]=-cov_yqbp(x_e,y_e,tx_e,ty_e,qp_e,dz_e,pe,b_x)*sig2E(qp_e)*q*(E[-1]**2)/(pe**4)
        Q_l[4][2]=-cov_txqbp(x_e,y_e,tx_e,ty_e,qp_e,dz_e,pe,b_x)*sig2E(qp_e)*q*(E[-1]**2)/(pe**4)
        Q_l[4][3]=-cov_tyqbp(x_e,y_e,tx_e,ty_e,qp_e,dz_e,pe,b_x)*sig2E(qp_e)*q*(E[-1]**2)/(pe**4)
        Q_l[4][4]=E[-1]**2/(pe**6)*sig2E(qp_e)

    Q_l=np.array(Q_l, dtype=float)

#***************************************************************
#COVARIANCE

#Covariance Matrix definition

def Covariance():

    global cov_C,F_k 
    
    cov_C = F_k@cov_C@(F_k.T)+Q_l
    cov_C = np.array(cov_C, dtype=float)
    
    
#***************************************************************
#KALMAN FILTER


def Kalman(i,X_k):   

    I=np.identity(5,dtype=float)
    
    X_k=np.array(X_k)
    
    global cov_C
    
    #Kalman Gain matrix
    K_F=cov_C@(H_k.T)@(np.linalg.inv(np.array((H_k@cov_C@(H_k.T)+V_k),dtype=float)))
  
    #Kalman estimate for state vector       
    X_k=(X_k.T+K_F@(M_k[i,:].T-H_k@(X_k.T))).T
    
    #Kalman estimate for filtered error covariance
    cov_C= (I-(K_F@H_k)) @ cov_C @ ((I-(K_F@H_k)).T)   +   K_F @ V_k @ (K_F.T)
    
    return X_k

#***************************************************************
#STATE VECTOR UPDATION

#Appending the new state vector value
def vector_updation(i,temp_stv):    
    global state_vector
    state_vector[i]=temp_stv
    
#***************************************************************
#PLOTS

def Plot_x(event_index):
    
    #for X coordinates

    plt.plot(x_m,z_m)
    plt.plot(state_vector[:,0],z_m,linestyle='dashed',c='red') 
    #plt.scatter(x_m,z_m)
    #plt.scatter(state_vector[:,0],z_m) 

    plt.legend(("Simulated x",'Predicted x' ))
    plt.xlabel('x - values') 
    plt.ylabel('y - values') 
    plt.title('Prediction vs Simulation for x coordinates for event',event_index+1,'(10 GeV)' )
    
    plt.grid(b=None, which='major', axis='both')
    plt.show() 

def Plot_z(event_index):
    
    #Z coordinates
    
    plt.plot(y_m,z_m)
    plt.plot(state_vector[:,1],z_m,linestyle='dashed',c='red')   
    #plt.scatter(y_m,z_m)
    #plt.scatter(state_vector[:,1],z_m)
 
    plt.legend(("Simulated z",'Predicted z' ))
    plt.xlabel('z - values') 
    plt.ylabel('y - values') 
    plt.title('Prediction vs Simulation for z coordinates for event',event_index+1,'(10 GeV)' )
    
    plt.grid(b=None, which='major', axis='both')
    plt.show() 
    
#***************************************************************
#MAIN LOOP STARTS HERE

for event_index in range(number_of_events+1):
    
    eid_ = df[df['eid']==event_index]
    
    number_of_layers=len(eid_)               # no. of the layers per event
    
    data_per_eid=np.array(eid_)          # array of dataframe pertaining to one particular event
    
    print("\n------------------------------------Event No.",event_index+1,"------------------------------------")
    
    x_m=data_per_eid[:,4]
    z_m=data_per_eid[:,5]
    y_m=data_per_eid[:,6] #interchange of z to y because by convention of the equation, the z is the perpendicular axis.
    M_k =[[ None for i in range(2) ] for j in range(len(data_per_eid)) ] #measured state vector
    M_k=np.array(M_k)
    
    print("No.of layers:",number_of_layers,'\n')
    
    for i in range(len(data_per_eid)): #saving the measured data 
        M_k[i][0]=x_m[i]
        M_k[i][1]=y_m[i]

    #***************************************************************
    #INITIALISATION


    #E_inc=1000.0                    
    E_inc=10000.0                    #Initial value of Incident Energy


    state_vector = np.array([[ None for i in range(5) ] for j in range(len(data_per_eid)) ]) #state vector
    
    
    
    #Defining the initial state vector from measured
    state_vector[0][0]= x_m[0]
    state_vector[0][1]= y_m[0]
    state_vector[0][2]= (x_m[1]-x_m[0])/(z_m[1]-z_m[0]) #tx_0 =(x1-x0)/(z1-z0)
    state_vector[0][3]= (y_m[1]-y_m[0])/(z_m[1]-z_m[0]) #ty_0 =(y1-y0)/(z1-z0)
    state_vector[0][4]= 10**-7           #Initial value of Momentum in GeV/c
      
      
    cov_C = 10**6*np.identity(5,  dtype=np.float)                #Initial value for covariance
    cov_C = np.array(cov_C)

    E=[]                            
    E.append(E_inc)                 #saving the initial value of energy
    
    
    for iteration_index in range(number_of_iterations): #controls the number of iterations; 
        
        forward_stv=[]              #list to save the predictions in forward loop
        P_F_lyr=[]                  #list for momentum of each layer

        temp_stv=state_vector[0]    
        
        F_k = [[None for j in range (5)] for i in range (5)]    #Propagator Matrix initialisation
        F_k = np.array(F_k,   dtype=float)

        Q_l =   [[None for j in range (5)] for i in range (5)]  #Random Error Matrix initialisation
        Q_l =   np.array(Q_l,   dtype=float)


        for i in range(len(data_per_eid)):   #Forward loop for 150 (air+iron+air) combo
            
            D=1
            
            x_,y_,tx_,ty_,qp_=temp_stv

            for j in range(58):     #Subloop for each layer( air(0) + iron(1-56) + air(57) )            

                if (j==0 or j==57):
                    dz_=20*D       #in mm for air gap between rpc and iron plate    
                    material="Air"
                    b__x=0.0       #magnetic field set as zero
                else:
                    dz_=1*D        #in mm for iron plate
                    material="Iron"
                    b__x=1.5       #magnetic field set as 1.5 Tesla in iron plate

                x_  = f_x (x_,y_,tx_,ty_,qp_,dz_,b__x)       #Updating x
                y_  = f_y (x_,y_,tx_,ty_,qp_,dz_,b__x)       #Updating y
                tx_ = f_tx(x_,y_,tx_,ty_,qp_,dz_,b__x)       #Updating tx
                ty_ = f_ty(x_,y_,tx_,ty_,qp_,dz_,b__x)       #Updating ty
            
                dl =dz_*np.sqrt(1+tx_**2+ty_**2)*D           #differential arc length
                
                if (material=="Iron"):
                    #dEds=EnergylossIron(qp_)                #manual calculation using equation
                    dEds=EnergylossIron_CubicSpline(q/qp_)   #taken from energyloss_data_table
                
                else:
                    dEds=EnergylossAir(qp_)                  #manual calculation using equation
                
                E_cal = E[-1]-(dEds*dl*10**-4)               #Updating Energy 

                E.append(E_cal)

                
                if ((E_cal<m)):
                    qp_= q/10                       #re-initialising as 10 GeV
                    E.append(np.sqrt((q/temp_stv[4])**2+m**2))
                else:
                    qp_ = q/np.sqrt(E_cal**2-m**2)            #Updating q/p    
                
                Propagator(x_,y_,tx_,ty_,qp_,dz_,dl,b__x)       #Updating Propagator matrix

                RandomError(x_,y_,tx_,ty_,qp_,dz_,dl,D,material)#Updating Random Error matrix

                Covariance()                                    #Updating the Covariance matrix
        
                temp_stv=x_,y_,tx_,ty_,qp_  

                forward_stv.append(temp_stv)                    #saving the state vector for smoothing
                #dummy=k_para(qp_)                 #function to check the nature of fluctuation of the energyloss for
                #print(dummy)
                        
            temp_stv=Kalman(i,temp_stv) #Kalman filtering
            
            vector_updation(i,temp_stv) #updating the state vector

            forward_stv[-1]=temp_stv               #updating the filtered state vector at the i-th layer (for smoothing)
            
            E.append(np.sqrt((q/temp_stv[4])**2+m**2))          
        
            #print(iteration_index,i,'\t',np.sqrt((q/temp_stv[4])**2+m**2)) #print energy esti. per layer
                    
        #LOOP IN REVERSE FOR SMOOTHING
        for i in reversed(range(len(data_per_eid))):       #Main loop for 150 (air+iron+air) combo forward+backwards
            
            D=-1
            
            for j in reversed(range(58)):     #Subloop for each combo            
                
                x_,y_,tx_,ty_,qp_=forward_stv[i*57+j]
                
                #x_,y_,tx_,ty_,qp_=temp_stv

                if (j==0 or j==57):
                    dz_=20*D       #in mm for air gap between rpc and iron      
                    material="Air"
                    b__x=0.0
                else:
                    dz_=1*D        #in mm for iron plate
                    material="Iron"
                    b__x=1.5
                
                
                #x_  = f_x (x_,y_,tx_,ty_,qp_,dz_,b__x)       #Updating x 
                #y_  = f_y (x_,y_,tx_,ty_,qp_,dz_,b__x)       #Updating y
                #tx_ = f_tx(x_,y_,tx_,ty_,qp_,dz_,b__x)       #Updating tx
                #ty_ = f_ty(x_,y_,tx_,ty_,qp_,dz_,b__x)       #Updating ty
            
                dl =dz_*np.sqrt(1+tx_**2+ty_**2)                  #differential arc length
                
                
                #if (material=="Iron"):
                #    dEds=EnergylossIron(qp_)
                #    #dEds=EnergylossIron_CubicSpline(q/qp_)
                
                #else:
                #    dEds=0.0#EnergylossAir(qp_)                 
                
                #E_cal = E[-1]-(dEds*dl*10**-4)                  #Updating Energy 

                #E.append(E_cal)

                
                #if ((E_cal<m)):
                #    qp_= q/10         #re-initialising as 10 GeV
                #    E.append(np.sqrt((q/temp_stv[4])**2+m**2))
                #else:
                #    qp_ = q/np.sqrt(E_cal**2-m**2)            #Updating q/p    
                
                Propagator(x_,y_,tx_,ty_,qp_,dz_,dl,b__x)        #Updating propagator matrix

                RandomError(x_,y_,tx_,ty_,qp_,dz_,dl,D,material) #Updating Random Error matrix

                Covariance()                                     #Updating the Covariance matrix
        
                temp_stv=x_,y_,tx_,ty_,qp_              
                
            
            temp_stv=Kalman(i,temp_stv) #Kalman filtering
            
            vector_updation(i,temp_stv) #updating the state vector
            
            E.append(np.sqrt((q/temp_stv[4])**2+m**2))          
            
            #print(iteration_index,i,'\t',np.sqrt((q/temp_stv[4])**2+m**2)) #print energy esti. per layer
            
            if iteration_index==(number_of_iterations-1):   #saving momentum only at the last iteration            
                P_F_lyr.append((q/temp_stv[4])) 
            
            #print(((data_per_eid[i,19]**2+data_per_eid[i,18]**2+data_per_eid[i,20]**2)**0.5),(q/temp_stv[4]))
        
        print('Energy of the muon for iteration',iteration_index+1,' = \t',np.sqrt((q/state_vector[0][4])**2+m**2))      
        
    #Plot_x(event_index)   #prints reconstructed track plot of x coordinate for a given event
    #Plot_z(event_index)   #prints reconstructed track plot of z coordinate for a given event
    
    P_F_lyr=P_F_lyr[::-1]   #since appending was done in reverse(smoothing)

    print('Energy of the muon for event',event_index+1,' = \t',np.sqrt((q/state_vector[0][4])**2+m**2))

    diff_p=((data_per_eid[:,19]**2+data_per_eid[:,18]**2+data_per_eid[:,20]**2)**0.5)-np.array(P_F_lyr)

    mu_event.append(np.mean(np.array(diff_p)))      #for single  event
    sigma_event.append(np.std(np.array(diff_p)))    #for single event

mu=np.mean(np.array(mu_event))          # for all events pertaining to a particular energy regime
sigma=np.std(np.array(sigma_event))     # for all events pertaining to a particular energy regime
print("mean difference in momentum per layer:",mu)
print("sigma:",sigma)


s = np.random.normal(mu, sigma, 10000000) #random samples from a normal (Gaussian) distribution with mu and sigma
count, bins, ignored = plt.hist(s, 30000, density=True) #histogram plots
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='b') #gaussian curve plot
plt.xlabel("$P_{true}$- $P_{Kalman}$ GeV")
plt.ylabel("Frequency")

end = time.time()  
print(f"Runtime of the program is {(end - start)/60} mins")

plt.show()
