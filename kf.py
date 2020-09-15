""" A python based implementation of Kalman filter  
    Author: Anupama Reghunath    """
import time

# starting time
start = time.time()
import math
import numpy as np       #for mathematical calculations
import matplotlib.pyplot as plt #for plotting
import pandas as pd      #for data frame creation
import sympy as sym      #for symbolic calculations
from scipy.interpolate import CubicSpline
#import sys
#sys.stdout = open('output.txt', 'w') #printing output onto output.txt


#***************************************************************
#READING MEASURED DATA
index=2                         #number of iterations

#df=pd.read_csv("dataGeV/mp20.txt",sep="\t")
df=pd.read_csv("dataGeV/mp_twoevent_4.txt",sep="\t") #muon plus ;2 events; 8 GeV

df1=np.array(df)


eid=df1[:,0]
n_event=max(eid)
print("No. of events:",(n_event+1))
mu_event,sigma_event=[],[]



Hk=[[1,0,0,0,0],[0,1,0,0,0]]    #projector matrix
#Vk=[[0.1/12,0],[0,0.1/12]]     #measurement error matrix
Vk=[[0,0],[0,0]]                #measurement error matrix
#Vk=[[1,0],[0,1]]                #measurement error matrix
Hk=np.array(Hk)
Vk=np.array(Vk)

#***************************************************************
#CONSTANTS

m=0.105658                      #mass of muon in GeV/c^2
me=0.511*10**-3                 #mass of electron in GeV/c^2
K=0.299792458*10**-3            #GeV c-1 T-1 mm-1
q=0.303                         #1 C in natural units               

bx=1.5                          #magnetic field in tesla
by=0

prop_f = [[None for j in range (5)] for i in range (5)]
prop_f = np.array(prop_f,dtype=float)

Q_l =[[None for j in range (5)] for i in range (5)]
Q_l=np.array(Q_l,dtype=float)

cov_C =10**6*np.identity(5,  dtype=np.float)                #Initial value for covariance
cov_C =np.array(cov_C)




#***************************************************************
#PREDICTION

x,y,tx,ty,qbp,dz,p=sym.symbols('x y tx ty qbp dz p') #symbols in prediction


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
    return(K*qbp*(1+ (tx)**2 +(ty)**2)**0.5)

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
def beta(qbpe):
    return (q/qbpe)/np.sqrt((q/qbpe)**2+m**2) 
def gamma(qbpe):
    #if (beta(qbpe))**2==1:
    #    return 10**-7
    #else:
    #print(qbpe,beta(qbpe))
    return 1/np.sqrt(1-(beta(qbpe))**2)              
def f_xd(qbpe):                 #required for Density Correction for Bethe Bloche Formula
    return math.log(beta(qbpe)*gamma(qbpe),10)
def Tmax(qbpe):
    return 2*me*(beta(qbpe)*gamma(qbpe))**2/( 1 + 2*(me/m)*np.sqrt(1+(beta(qbpe)*gamma(qbpe))**2+(me/m)**2) ) # Max Kinetic Eenrgy   

def EnergylossIron(qbpe):
    rho=7.874
    ZbA=26.0/55.845             #Z/A
    I=286*10**-9                #in GeV
    xd0=-0.0012
    xd1=3.15
    md=2.96
    a=0.1468
    C0=-4.29
    if qbpe<0:
        qbpe=-qbpe
     #print(qbpe,f_b(qbpe),f_g(qbpe),f_xd(qbpe))
    xd=f_xd(qbpe)
    if xd<xd0:
       delta=0.0
    if xd>xd0 and xd<xd1:
       delta=4.6052*xd+C0+(a*((xd1-xd)**md))
    if xd>xd1:
       delta=4.6052*xd+C0
    
    dEds= rho*0.307075/((beta(qbpe))**2)*ZbA*( 0.5*np.log(2*me*(beta(qbpe)*gamma(qbpe))**2*Tmax(qbpe)/(I**2))-(beta(qbpe)**2)-delta*0.5) #Bethe Bloch Formula in MeVcm**2/g        
    
    return dEds
   
def EnergylossAir(qbpe):
    rho=1.205*10**-3
    ZbA=0.49919                 #Z/A
    I=85.7*10**-9               #in GeV  
    xd0=1.742
    xd1=4.28
    md=3.40
    a=0.1091
    C0=-10.6
    if qbpe<0:
        qbpe=-qbpe
    #print(qbpe,f_b(qbpe),f_g(qbpe),f_xd(qbpe))
    xd=f_xd(qbpe)
    if xd<xd0:
        delta=0.0
    if xd>xd0 and xd<xd1:
        delta=4.6052*xd+C0+(a*((xd1-xd)**md))
    if xd>xd1:
        delta=4.6052*xd+C0   

    dEds= rho*0.307075/((beta(qbpe))**2)*ZbA*( 0.5*np.log(2*me*(beta(qbpe)*gamma(qbpe))**2*Tmax(qbpe)/(I**2))-(beta(qbpe)**2)-delta*0.5) #Bethe Bloch Formula in MeVcm**2/g                     

    return dEds


#Converting symbolic to mathematical

f_x = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_xe(), "numpy")
f_y = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_ye(), "numpy")
f_tx= sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_txe(),"numpy")
f_ty= sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_tye(),"numpy")

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
    return Prediction_xe().diff(qbp)

def Prediction_yprimex():
    return Prediction_ye().diff(x)
def Prediction_yprimey():
    return Prediction_ye().diff(y)
def Prediction_yprimetx():
    return Prediction_ye().diff(tx)
def Prediction_yprimety():
    return Prediction_ye().diff(ty)
def Prediction_yprimeqbp():
    return Prediction_ye().diff(qbp)

def Prediction_txprimex():
    return Prediction_txe().diff(x)
def Prediction_txprimey():
    return Prediction_txe().diff(y)
def Prediction_txprimetx():
    return Prediction_txe().diff(tx)
def Prediction_txprimety():
    return Prediction_txe().diff(ty)
def Prediction_txprimeqbp():
    return Prediction_txe().diff(qbp)

def Prediction_typrimex():
    return Prediction_tye().diff(x)
def Prediction_typrimey():
    return Prediction_tye().diff(y)
def Prediction_typrimetx():
    return Prediction_tye().diff(tx)
def Prediction_typrimety():
    return Prediction_tye().diff(ty)
def Prediction_typrimeqbp():
    return Prediction_tye().diff(qbp)

#Converting symbolic to mathematical

f_xprimex   = sym.lambdify( (x,y,tx,ty,qbp,dz), Prediction_xprimex(),   "numpy")
f_xprimey   = sym.lambdify( (x,y,tx,ty,qbp,dz), Prediction_xprimey(),   "numpy")
f_xprimetx  = sym.lambdify( (x,y,tx,ty,qbp,dz), Prediction_xprimetx(),  "numpy")
f_xprimety  = sym.lambdify( (x,y,tx,ty,qbp,dz), Prediction_xprimety(),  "numpy")
f_xprimeqbp = sym.lambdify( (x,y,tx,ty,qbp,dz), Prediction_xprimeqbp(), "numpy")    

f_yprimex   = sym.lambdify( (x,y,tx,ty,qbp,dz), Prediction_yprimex(),   "numpy")
f_yprimey   = sym.lambdify( (x,y,tx,ty,qbp,dz), Prediction_yprimey(),   "numpy")
f_yprimetx  = sym.lambdify( (x,y,tx,ty,qbp,dz), Prediction_yprimetx(),  "numpy")
f_yprimety  = sym.lambdify( (x,y,tx,ty,qbp,dz), Prediction_yprimety(),  "numpy")
f_yprimeqbp = sym.lambdify( (x,y,tx,ty,qbp,dz), Prediction_yprimeqbp(), "numpy")    

f_txprimex  = sym.lambdify( (tx,ty,qbp,dz), Prediction_txprimex(),  "numpy")
f_txprimey  = sym.lambdify( (tx,ty,qbp,dz), Prediction_txprimey(),  "numpy")
f_txprimetx = sym.lambdify( (tx,ty,qbp,dz), Prediction_txprimetx(), "numpy")
f_txprimety = sym.lambdify( (tx,ty,qbp,dz), Prediction_txprimety(), "numpy")
f_txprimeqbp= sym.lambdify( (tx,ty,qbp,dz), Prediction_txprimeqbp(),"numpy")    

f_typrimex  = sym.lambdify( (tx,ty,qbp,dz), Prediction_typrimex(),  "numpy")
f_typrimey  = sym.lambdify( (tx,ty,qbp,dz), Prediction_typrimey(),  "numpy")
f_typrimetx = sym.lambdify( (tx,ty,qbp,dz), Prediction_typrimetx(), "numpy")
f_typrimety = sym.lambdify( (tx,ty,qbp,dz), Prediction_typrimety(), "numpy")
f_typrimeqbp= sym.lambdify( (tx,ty,qbp,dz), Prediction_typrimeqbp(),"numpy")    

#***************************************************************
#RANGE-MOMENTUM RELATION FROM CURVE FITTING FOR ERROR IN Q/P

#obtaining data points for Cubic Spline interpolation
cb =pd.read_csv("muon-iron-energyLossTable3.txt",sep=" ") 
cb1=np.array(cb)

pp =cb1[:,3]*10**-3             #converting MeV/c into GeV/c
l_r=cb1[:,10]*10/7.874          #in q/cm^2 to mm (/iron density)
ene_r=cb1[:,9]*7.874

#EnergylossIron_C = CubicSpline(pp,ene_r,bc_type='natural')     #Alternate equation for Energy-loss in Iron

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
def f_qbpprimeqbp(txe,tye,qbpe,dze,dl):
    
    pe=qbpe
    return 1+ (fl_2(pe))/fl_1(pe)*dl + 0.5*fl_3(pe)/fl_1(pe)*dl**2

def f_qbpprimex(txe,tye,qbpe,dze,dl):
    
    pe=qbpe
    return K*(fl_1(pe)+fl_2(pe)*dl)*fl(pe)*np.sqrt(1+txe**2+tye**2)*dl*(-by)

def f_qbpprimey(txe,tye,qbpe,dze,dl):
    
    pe=qbpe
    return K*(fl_1(pe)+fl_2(pe)*dl)*fl(pe)*np.sqrt(1+txe**2+tye**2)*dl*(bx)

def f_qbpprimetx(txe,tye,qbpe,dze,dl):
    
    pe=qbpe
    return (fl_1(pe)+fl_2(pe)*dl)*dze*(txe/np.sqrt(1+txe**2+tye**2))

def f_qbpprimety(txe,tye,qbpe,dze,dl):
    
    pe=qbpe
    return  (fl_1(pe)+fl_2(pe)*dl)*dze*(tye/np.sqrt(1+txe**2+tye**2))


 

def Propagator(xe,ye,txe,tye,qbpe,dze,dl):
    
    global prop_f    
    
    prop_f[0][0] = f_xprimex  (xe,ye,txe,tye,qbpe,dze)
    prop_f[0][1] = f_xprimey  (xe,ye,txe,tye,qbpe,dze)
    prop_f[0][2] = f_xprimetx (xe,ye,txe,tye,qbpe,dze)
    prop_f[0][3] = f_xprimety (xe,ye,txe,tye,qbpe,dze)
    prop_f[0][4] = f_xprimeqbp(xe,ye,txe,tye,qbpe,dze)
   
    prop_f[1][0] = f_yprimex  (xe,ye,txe,tye,qbpe,dze)
    prop_f[1][1] = f_yprimey  (xe,ye,txe,tye,qbpe,dze)
    prop_f[1][2] = f_yprimetx (xe,ye,txe,tye,qbpe,dze)
    prop_f[1][3] = f_yprimety (xe,ye,txe,tye,qbpe,dze)
    prop_f[1][4] = f_yprimeqbp(xe,ye,txe,tye,qbpe,dze)
   
    prop_f[2][0] = f_txprimex  (txe,tye,qbpe,dze)
    prop_f[2][1] = f_txprimey  (txe,tye,qbpe,dze)
    prop_f[2][2] = f_txprimetx (txe,tye,qbpe,dze)
    prop_f[2][3] = f_txprimety (txe,tye,qbpe,dze)
    prop_f[2][4] = f_txprimeqbp(txe,tye,qbpe,dze)
   
    prop_f[3][0] = f_typrimex  (txe,tye,qbpe,dze)
    prop_f[3][1] = f_typrimey  (txe,tye,qbpe,dze)
    prop_f[3][2] = f_typrimetx (txe,tye,qbpe,dze)
    prop_f[3][3] = f_typrimety (txe,tye,qbpe,dze)
    prop_f[3][4] = f_typrimeqbp(txe,tye,qbpe,dze)

    prop_f[4][0] = f_qbpprimex  (txe,tye,qbpe,dze,dl)
    prop_f[4][1] = f_qbpprimey  (txe,tye,qbpe,dze,dl)
    prop_f[4][2] = f_qbpprimetx (txe,tye,qbpe,dze,dl)
    prop_f[4][3] = f_qbpprimety (txe,tye,qbpe,dze,dl)
    prop_f[4][4] = f_qbpprimeqbp(txe,tye,qbpe,dze,dl)

    
    return ()

#***************************************************************
#RANDOM ERROR CALCULATION

def CMS(txe,tye,qbpe,dl):    #Highland-Lynch-Dahl variance formula
        
    Z= 26.0
    l_rad=17.57 #thickness of the medium in mm
    
    ls = l_rad*((Z+1)/Z)*(289*Z**(-1/2))/(159*Z**(-1/3))  #17.57= radiation length of iron in mm
    pe = q/qbpe
    return (0.015/(beta(qbpe)*pe))**2*(dl/ls)

def cov_txtx(txe,tye,qbpe,dl):                 
    return (1+txe**2)*(1+txe**2+tye**2)*CMS(txe,tye,qbpe,dl)
def cov_tyty(txe,tye,qbpe,dl):
    return (1+tye**2)*(1+txe**2+tye**2)*CMS(txe,tye,qbpe,dl)
def cov_txty(txe,tye,qbpe,dl):
    return txe*tye*(1+txe**2+tye**2)*CMS(txe,tye,qbpe,dl)

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


cov_xqbp = sym.lambdify((x,y,tx,ty,qbp,dz,p), Prediction_xprimep(),  "numpy")
cov_yqbp = sym.lambdify((x,y,tx,ty,qbp,dz,p), Prediction_yprimep(),  "numpy")
cov_txqbp= sym.lambdify((x,y,tx,ty,qbp,dz,p), Prediction_txprimep(), "numpy")
cov_tyqbp= sym.lambdify((x,y,tx,ty,qbp,dz,p), Prediction_typrimep(), "numpy")


#Calculation for Energy loss Struggling Elements in the Random error Matrix

def xi(qbpe):                                               #Mean energy loss in GeV
    
    rho = 7.874
    ZbA = 26.0/55.845 
    d = 56                      #thickness of the medium in mm
    return (0.1534*q**2*ZbA/beta(qbpe)**2)*rho*d*10**-4

def sig2E(qbpe):                                            #var of the Gaussain distribution in GeV^2
    return  xi(qbpe)*Tmax(qbpe)*10**-3*(1-(beta(qbpe)**2)/2)


#Random Error Matrix definition

def RandomError(xe,ye,txe,tye,qbpe,dze,dl,D,material):                #Random Error Matrix


    global Q_l
        
    pe=q/qbpe
    if (material=="Air"):
        Q_l =[[0 for j in range (5)] for i in range (5)]
    else:

        Q_l[0][0]= cov_txtx(txe,tye,qbpe,dl)*(dl**3)/3
        Q_l[0][1]= cov_txty(txe,tye,qbpe,dl)*(dl**3)/3
        Q_l[0][2]= cov_txtx(txe,tye,qbpe,dl)*(dl**2)*D/2
        Q_l[0][3]= cov_txty(txe,tye,qbpe,dl)*(dl**2)*D/2
        Q_l[0][4]=-cov_xqbp(xe,ye,txe,tye,qbpe,dze,pe)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
    
        Q_l[1][0]=cov_txty(txe,tye,qbpe,dl)*(dl**3)/3
        Q_l[1][1]=cov_tyty(txe,tye,qbpe,dl)*(dl**3)/3
        Q_l[1][2]=cov_txty(txe,tye,qbpe,dl)*(dl**2)*D/2
        Q_l[1][3]=cov_tyty(txe,tye,qbpe,dl)*(dl**2)*D/2
        Q_l[1][4]=-cov_yqbp(xe,ye,txe,tye,qbpe,dze,pe)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)

        Q_l[2][0]=cov_txtx(txe,tye,qbpe,dl)*(dl**2)*D/2
        Q_l[2][1]=cov_txty(txe,tye,qbpe,dl)*(dl**2)*D/2
        Q_l[2][2]=cov_txtx(txe,tye,qbpe,dl)*dl
        Q_l[2][3]=cov_txty(txe,tye,qbpe,dl)*dl
        Q_l[2][4]=-cov_txqbp(xe,ye,txe,tye,qbpe,dze,pe)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
    
        Q_l[3][0]=cov_txty(txe,tye,qbpe,dl)*(dl**2)*D/2
        Q_l[3][1]=cov_tyty(txe,tye,qbpe,dl)*(dl**2)*D/2
        Q_l[3][2]=cov_txtx(txe,tye,qbpe,dl)*dl
        Q_l[3][3]=cov_tyty(txe,tye,qbpe,dl)*dl
        Q_l[3][4]=-cov_tyqbp(xe,ye,txe,tye,qbpe,dze,pe)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)

        Q_l[4][0]=-cov_xqbp(xe,ye,txe,tye,qbpe,dze,pe)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
        Q_l[4][1]=-cov_yqbp(xe,ye,txe,tye,qbpe,dze,pe)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
        Q_l[4][2]=-cov_txqbp(xe,ye,txe,tye,qbpe,dze,pe)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
        Q_l[4][3]=-cov_tyqbp(xe,ye,txe,tye,qbpe,dze,pe)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
        Q_l[4][4]=E[-1]**2/(pe**6)*sig2E(qbpe)

    Q_l=np.array(Q_l, dtype=float)

#***************************************************************
#COVARIANCE

#Covariance Matrix definition

def Covariance():

    global cov_C,prop_f 
    
    cov_C = prop_f@cov_C@(prop_f.T)+Q_l
    cov_C = np.array(cov_C, dtype=float)
    
    
#***************************************************************
#KALMAN FILTER

ide=np.identity(5,dtype=float)

def Kalman(i,temp_stv):   
    
    temp_stv=np.array(temp_stv)
    global cov_C
      
    #Kalman Gain matrix
    Kf=cov_C@(Hk.T)@(np.linalg.inv(np.array((Hk@cov_C@(Hk.T)+Vk),dtype=float)))
  
    #Kalman estimate for state vector       
    temp_stv=(temp_stv.T+Kf@(mk[i,:].T-Hk@(temp_stv.T))).T
    
    #Kalman estimate for filtered error covariance
    cov_C=(ide-Kf@Hk)@cov_C@((ide-Kf@Hk).T)+Kf@Vk@(Kf.T)
    
    return temp_stv

#***************************************************************
#STATE VECTOR UPDATION

#Appending the new state vector value
def vector_updation(i,temp,z):    
    global stv
    stv[i][0],stv[i][1],stv[i][2],stv[i][3],stv[i][4]=temp
    xpp.append(stv[i][0])
    ypp.append(stv[i][1])
    zpp.append(z)
    

#***************************************************************
#PLOTS

def Plot_x():
    
    #for X coordinates

    plt.plot(xm,zm)
    plt.plot(xpp,zpp,linestyle='dashed',c='red') 
    #plt.scatter(xm,zm)
    #plt.scatter(xpp,zpp) 

    plt.legend(("Simulated x",'Predicted x' ))
    plt.xlabel('x - values') 
    plt.ylabel('y - values') 
    plt.title('Prediction vs Simulation for x coordinates (20 GeV)') 
    
    plt.grid(b=None, which='major', axis='both')
    plt.show() 

def Plot_z():
    
    #Z coordinates
    
    plt.plot(ym,zm)
    plt.plot(ypp,zpp,linestyle='dashed',c='red')   
    #plt.scatter(ym,zm)
    #plt.scatter(ypp,zpp)
 
    plt.legend(("Simulated z",'Predicted z' ))
    plt.xlabel('z - values') 
    plt.ylabel('y - values') 
    plt.title('Prediction vs Simulation for z coordinates (20 GeV)') 
    
    plt.grid(b=None, which='major', axis='both')
    plt.show() 
    
#***************************************************************
#MAIN LOOP STARTS HERE

for event_index in range(n_event+1):
    eid_ds = df[df['eid']==event_index]
    layer=len(eid_ds)               # no. of the layers per event
    eid_a=np.array(eid_ds)          # array of dataframe pertaining to one particular event
    print("\n#----------------------------------------Event No.",event_index+1,"----------------------------------------#")
    xm=eid_a[:,4]
    zm=eid_a[:,5]
    ym=eid_a[:,6] #interchange of z to y because by convention of the equation, the z is the perpendicular axis.
    mk =[[ None for i in range(2) ] for j in range(len(eid_a)) ] #state vector
    mk=np.array(mk)
    #print(len(eid_a))
    for i in range(len(eid_a)):
        mk[i][0]=xm[i]
        mk[i][1]=ym[i]
#plt.plot(mk[:,0],zm)
    #***************************************************************
    #INITIALISATION


    #E_inc=10.0                      
    #E_inc=20.0                      
    #E_inc=50.0                      
    #E_inc=100.0                     
    E_inc=1000.0                    #Initial value of Incident Energy


    stv = [[ None for i in range(5) ] for j in range(len(eid_a)) ] #state vector
    stv = np.array(stv)
    forward_stv=np.array([[ None for i in range(5) ] for j in range(len(eid_a)) ]) #saving the state vector after forward iteration
    #Defining the initial state vector from measured
    stv[0][0]= eid_a[0,4]
    stv[0][1]= eid_a[0,6]
    stv[0][2]= (eid_a[1,4]-eid_a[0,4])/(zm[1]-zm[0]) #tx_0 =(x1-x0)/(z1-z0)
    stv[0][3]= (eid_a[1,6]-eid_a[0,6])/(zm[1]-zm[0]) #ty_0 =(y1-y0)/(z1-z0)
    stv[0][4]= 10**-7#0.0           #Initial value of Momentum in GeV/c
    P_F=[q/stv[0][4]]
#    print("No.of layers",layer)
    P_F_lyr=[]
    E=[]
    
    #Propagator matrix definition
    prop_f = [[None for j in range (5)] for i in range (5)]
    prop_f = np.array(prop_f,dtype=float)

    Q_l =[[None for j in range (5)] for i in range (5)]
    Q_l=np.array(Q_l,dtype=float)

    cov_C =10**6*np.identity(5,  dtype=np.float)                #Initial value for covariance
    cov_C =np.array(cov_C)

    E.append(E_inc)                 #saving the initial value of energy
    zpp,xpp,ypp=[],[],[]            #for plotting
    
    for iterations in range(index): #controls the number of iterations; 
        temp_stv=stv[0]
        ze=zm[0]                        #initial value for z ;for plotting

        for i in range(len(eid_a)):   #Main loop for 150 (air+iron+air) combo
            
            D=1
            x_e,y_e,tx_e,ty_e,qbp_e=temp_stv

            for j in range(58):     #Subloop for each combo            

                if (j==0 or j==57):
                    dz_e=20*D       #in mm for air gap between rpc and iron      
                    material="Air"
                else:
                    dz_e=1*D        #in mm for iron plate
                    material="Iron"
            

                x_e  = f_x (x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating x
                y_e  = f_y (x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating y
                tx_e = f_tx(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating tx
                ty_e = f_ty(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating ty
            
                dl =dz_e*np.sqrt(1+tx_e**2+ty_e**2)*D             #differential arc length
                
                
                if (material=="Iron"):
                    dEds=EnergylossIron(qbp_e)
                    #dEds=EnergylossIron_C(q/qbp_e)
                else:
                    dEds=EnergylossAir(qbp_e)                 
                
                E_cal = E[-1]-(dEds*dl*10**-4)                  #Updating Energy 
                E.append(E_cal)

                
                if ((E_cal**2-m**2)<0):#10**-7):
                    #qbp_e=qbp_e
                    qbp_e= q/10         #re-initialising as 10 GeV
                    E.append(np.sqrt((q/temp_stv[4])**2+m**2))
                else:
                    qbp_e = q/np.sqrt(E_cal**2-m**2)            #Updating q/p    
                
                Propagator(x_e,y_e,tx_e,ty_e,qbp_e,dz_e,dl)        #Updating propagator matrix
                RandomError(x_e,y_e,tx_e,ty_e,qbp_e,dz_e,dl,D,material)

                Covariance()                                    #Updating the Covariance matrix
        
                #ze = ze - dz_e                                  #Updating z
                temp_stv=x_e,y_e,tx_e,ty_e,qbp_e    
                
                #print(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)
                
                #printing the propagator matrix for each j   
                #print('\n',i,j,'F',prop_f[0][0],prop_f[0][1],prop_f[0][2],prop_f[0][3],prop_f[0][4],'\n','\t',prop_f[1][0],prop_f[1][1],prop_f[1][2],prop_f[1][3],prop_f[1][4],'\n','\t',prop_f[2][0],prop_f[2][1],prop_f[2][2],prop_f[2][3],prop_f[2][4],'\n','\t',prop_f[3][0],prop_f[3][1],prop_f[3][2],prop_f[3][3],prop_f[3][4],'\n','\t',prop_f[4][0],prop_f[4][1],prop_f[4][2],prop_f[4][3],prop_f[4][4],'\n________________________________________________________________________________________________________________________________\n')        

                #printing the random error matrix for each j   
                #print('\n',i,j,'Q',Q_l[0][0],Q_l[0][1],Q_l[0][2],Q_l[0][3],Q_l[0][4],'\n','\t',Q_l[1][0],Q_l[1][1],Q_l[1][2],Q_l[1][3],Q_l[1][4],'\n','\t',Q_l[2][0],Q_l[2][1],Q_l[2][2],Q_l[2][3],Q_l[2][4],'\n','\t',Q_l[3][0],Q_l[3][1],Q_l[3][2],Q_l[3][3],Q_l[3][4],'\n','\t',Q_l[4][0],Q_l[4][1],Q_l[4][2],Q_l[4][3],Q_l[4][4],'\n________________________________________________________________________________________________________________________________\n')        
                
                #printing the covariance matrix for each j   
                #print('\n',i,j,'C',cov_C[0][0],cov_C[0][1],cov_C[0][2],cov_C[0][3],cov_C[0][4],'\n','\t',cov_C[1][0],cov_C[1][1],cov_C[1][2],cov_C[1][3],cov_C[1][4],'\n','\t',cov_C[2][0],cov_C[2][1],cov_C[2][2],cov_C[2][3],cov_C[2][4],'\n','\t',cov_C[3][0],cov_C[3][1],cov_C[3][2],cov_C[3][3],cov_C[3][4],'\n','\t',cov_C[4][0],cov_C[4][1],cov_C[4][2],cov_C[4][3],cov_C[4][4],'\n________________________________________________________________________________________________________________________________\n')            
            
                #print(iterations,i,j,'\t',E[-1])
                
                
            #print(i,E[-1])
            temp_stv=Kalman(i,temp_stv) #Kalman filtering
            
            E.append(np.sqrt((q/temp_stv[4])**2+m**2))          #updating the energy with new q/p
            
            forward_stv[i][0],forward_stv[i][1],forward_stv[i][2],forward_stv[i][3],forward_stv[i][4]=temp_stv #saving the state vector for smoothing
            
            #vector_updation(i,forward_stv,ze)                  #saves the state vector only at the last iteration
            
            P_F.append(q/temp_stv[4])
            
            #print(iterations,i,'\t',E[-1])
            #print(iterations,i,temp_stv[0],temp_stv[1],temp_stv[2],temp_stv[3],temp_stv[4],'\t',E[-1])
        
    #LOOP REPEATED IN REVERSE FOR SMOOTHING
        for i in reversed(range(len(eid_a))):       #Main loop for 150 (air+iron+air) combo forward+backwards
            
            D=-1
            
            x_e,y_e,tx_e,ty_e,qbp_e=forward_stv[i]
            
            for j in range(58):     #Subloop for each combo            
                
                if (j==0 or j==57):
                    dz_e=20#*D       #in mm for air gap between rpc and iron      
                    material="Air"
                else:
                    dz_e=1#*D        #in mm for iron plate
                    material="Iron"
                

                #x_e  = f_x (x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating x
                #y_e  = f_y (x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating y
                #tx_e = f_tx(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating tx
                #ty_e = f_ty(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating ty
                
                dl =dz_e*np.sqrt(1+tx_e**2+ty_e**2)#*D             #differential arc length
            
                if (i==0 and j==57):
                #    dEds=EnergylossIron(qbp_e)
                    #dEds=EnergylossIron_C(q/qbp_e)
                #else:
                    dEds=EnergylossAir(qbp_e)          
            
                    E_cal = E[-1]+(dEds*dl*10**-4)                  #Updating Energy
                
                    E.append(E_cal)

                    #flag=flag+1
                    #count.append(flag)   
                
                    #if ((E_cal**2-m**2)<10**-7):
                    #    qbp_e= (q/10)                 #re-initialising as 10 GeV
                    #    E.append(np.sqrt((q/temp_stv[4])**2+m**2))
                    #else:
                    qbp_e = q/np.sqrt(E_cal**2-m**2)            #Updating q/p    
                
                
                Propagator(x_e,y_e,tx_e,ty_e,qbp_e,dz_e,dl)        #Updating propagator matrix
                RandomError(x_e,y_e,tx_e,ty_e,qbp_e,dz_e,dl,D,material)

                Covariance()                                    #Updating the Covariance matrix
        
                ze = ze - dz_e                                  #Updating z
                temp_stv=x_e,y_e,tx_e,ty_e,qbp_e    
        
                #printing the propagator matrix for each j   
                #print('\n',i,j,'F',prop_f[0][0],prop_f[0][1],prop_f[0][2],prop_f[0][3],prop_f[0][4],'\n','\t',prop_f[1][0],prop_f[1][1],prop_f[1][2],prop_f[1][3],prop_f[1][4],'\n','\t',prop_f[2][0],prop_f[2][1],prop_f[2][2],prop_f[2][3],prop_f[2][4],'\n','\t',prop_f[3][0],prop_f[3][1],prop_f[3][2],prop_f[3][3],prop_f[3][4],'\n','\t',prop_f[4][0],prop_f[4][1],prop_f[4][2],prop_f[4][3],prop_f[4][4],'\n________________________________________________________________________________________________________________________________\n')        

                #printing the random error matrix for each j   
                #print('\n',i,j,'Q',Q_l[0][0],Q_l[0][1],Q_l[0][2],Q_l[0][3],Q_l[0][4],'\n','\t',Q_l[1][0],Q_l[1][1],Q_l[1][2],Q_l[1][3],Q_l[1][4],'\n','\t',Q_l[2][0],Q_l[2][1],Q_l[2][2],Q_l[2][3],Q_l[2][4],'\n','\t',Q_l[3][0],Q_l[3][1],Q_l[3][2],Q_l[3][3],Q_l[3][4],'\n','\t',Q_l[4][0],Q_l[4][1],Q_l[4][2],Q_l[4][3],Q_l[4][4],'\n________________________________________________________________________________________________________________________________\n')        
                
                #printing the covariance matrix for each j   
                #print('\n',i,j,'C',cov_C[0][0],cov_C[0][1],cov_C[0][2],cov_C[0][3],cov_C[0][4],'\n','\t',cov_C[1][0],cov_C[1][1],cov_C[1][2],cov_C[1][3],cov_C[1][4],'\n','\t',cov_C[2][0],cov_C[2][1],cov_C[2][2],cov_C[2][3],cov_C[2][4],'\n','\t',cov_C[3][0],cov_C[3][1],cov_C[3][2],cov_C[3][3],cov_C[3][4],'\n','\t',cov_C[4][0],cov_C[4][1],cov_C[4][2],cov_C[4][3],cov_C[4][4],'\n________________________________________________________________________________________________________________________________\n')            
                
                #print(i,j,'\t',E[-1])
                
            temp_stv=Kalman(i,temp_stv)                         #Kalman filtering 
            
            E.append(np.sqrt((q/temp_stv[4])**2+m**2))          #updating the energy with new q/p
            P_F.append(q/temp_stv[4])
            
            if iterations==(index-1):               
                vector_updation(i,temp_stv,ze*D)                  #saves the state vector only at the last iteration
                P_F_lyr.append(P_F[-1])
            
            #print(((eid_a[i,19]**2+eid_a[i,18]**2+eid_a[i,20]**2)**0.5),P_F[-1])

            #print(iterations,i,'\t',E[-1])
            #print(iterations,i,temp_stv[0],temp_stv[1],temp_stv[2],temp_stv[3],temp_stv[4],'\t',E[-1])
        #print('Iteration',iterations+1,' Energy of the muon = \t',E[-1])
    #Plot_x()
    #Plot_z()
    P_F_lyr=P_F_lyr[::-1]
    print('Energy of the muon for event',event_index+1,' = \t',E[-1])

    diff_p=((eid_a[:,19]**2+eid_a[:,18]**2+eid_a[:,20]**2)**0.5)-np.array(P_F_lyr)
#    diff_eve.append(diff_p)
#    print("diff",diff_p)    
    mu_event.append(np.mean(np.array(diff_p)))      #for single  event
    sigma_event.append(np.std(np.array(diff_p)))    #for single event

mu=np.mean(np.array(mu_event))          # for all events pertaining to a particular energy regime
sigma=np.std(np.array(sigma_event))     # for all events pertaining to a particular energy regime
print("mean:",mu)
print("sigma:",sigma)
s = np.random.normal(mu, sigma, 1000000)#0)
count, bins, ignored = plt.hist(s, 30000, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='b')
# x=((eid_a[:,19]**2+eid_a[:,18]**2+eid_a[:,20]**2)**0.5)
# plt.plot(x,y)
# plt.hist(y)
plt.xlabel("$P_{true}$- $P_{Kalman}$ GeV")
plt.ylabel("Frequency")
plt.show()

end = time.time()
print(f"Runtime of the program is {end - start}")