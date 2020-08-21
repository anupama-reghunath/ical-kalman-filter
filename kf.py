""" A python based implementation of a kalman filter 
    Author: Anupama Reghunath    """
import time

# starting time
start = time.time()

import numpy as np       #for mathematical calculations
import matplotlib.pyplot as plt #for plotting
import pandas as pd      #for data frame creation
import sympy as sym      #for symbolic calculations
#import scipy as sp
#import sys
#sys.stdout = open('output.txt', 'w') #printing output onto output.txt

#***************************************************************
#to find the f(l) for q/p error prediction by fitting data from muon-iron-energyLossTable

#coeffecients of p
c00,c01,c02,c03,c04,c05=20.31019610992487, 818.1567338049706, -21.382302321528552, 1.379247983139621, -0.04875634389271173, 0.0006735041922117595
#l as a function of p    
def range_l():
    return c00+c01*(q/qbp)+c02*(q/qbp)**2+c03*(q/qbp)**3+c04*(q/qbp)**4+c05*(q/qbp)**5

c_0,c_1,c_2,c_3,c_4,c_5=-0.00022297311773143545, 0.0011797929938868693, 6.069134604935027e-08, -5.389046335483464e-12, 2.8207009909973285e-16, -5.86475197560795e-21
#p as a function of l
def fl():
    return c_0+c_1*range_l()+c_2*range_l()**2+c_3*range_l()**3+c_4*range_l()**4+c_5*range_l()**5
def fl_1():
    return c_1+2*c_2*range_l()+3*c_3*range_l()**2+4*c_4*range_l()**3+5*c_5*range_l()**4
def fl_2():
    return 2*c_2+3*2*c_3*range_l()+4*3*c_4*range_l()**2+5*4*c_5*range_l()**3
def fl_3():
    return 3*2*c_3+4*3*2*c_4*range_l()+5*4*3*c_5*range_l()**2


#***************************************************************
#Reading measured data
df=pd.read_csv("dataGeV/mp20.txt",sep="\t")
df1=np.array(df)

xm=df1[:,4]
zm=df1[:,5]
ym=df1[:,6] #interchange of z to y because by convention of the equation, the z is the perpendicular axis.

mk =[[ None for i in range(2) ] for j in range(len(df1)) ] #state vector
mk=np.array(mk)
for i in range(len(df1)):
    mk[i][0]=xm[i]
    mk[i][1]=ym[i]
Hk=[[1,0,0,0,0],[0,1,0,0,0]]    #projector matrix
#Vk=[[0.1/12,0],[0,0.1/12]]      #measurement error matrix
Vk=[[0,0],[0,0]]      #measurement error matrix
Hk=np.array(Hk)
Vk=np.array(Vk)


#***************************************************************

m=0.105658                      #mass of muon in GeV/c^2
me=0.511*10**-3                 #mass of electron in GeV/c**2
K=0.299792458*10**-3            #GeV c-1 T-1 mm-1
q=0.303                         #1 C in natural units               

bx=1.5                          #magnetic field in tesla
by=0

#***************************************************************
#Initialisation of values
#E_inc=1.0                       #Initial value of Incident Energy
#E_inc=5.0                       #Initial value of Incident Energy
#E_inc=10.0                      #Initial value of Incident Energy
#E_inc=20.0                      #Initial value of Incident Energy
E_inc=50.0                       #Initial value of Incident Energy
#E_inc=100.0                     #Initial value of Incident Energy


stv = [[ None for i in range(5) ] for j in range(len(df1)) ] #state vector
stv = np.array(stv)

#Defining the initial state vector from measured
stv[0][0]= df1[0,4]
stv[0][1]= df1[0,6]
stv[0][2]= (df1[1,4]-df1[0,4])/(zm[1]-zm[0]) #tx_0 =(x1-x0)/(z1-z0)
stv[0][3]= (df1[1,6]-df1[0,6])/(zm[1]-zm[0]) #ty_0 =(y1-y0)/(z1-z0)
stv[0][4]= 10**-7 #0.0            #Initial value of Momentum in GeV/c

#***************************************************************
#Defining the Prediction equations

x,y,tx,ty,qbp,dz=sym.symbols('x y tx ty qbp dz') #symbols in prediction


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


def Prediction_xe():
    return (x  + tx*dz + h()*(tx*ty*sx()-(1+tx**2)*sy()) + h()**2*(tx*(3*ty**2 +1)*sxx() -ty*(3*tx**2 +1)*sxy() -ty*(3*tx**2 +1)*syx() +tx*(3*tx**2 +3)*syy()))
def Prediction_ye():
    return(y  + ty*dz + h()*((1+ty**2)*sx() -tx*ty*sy()) + h()**2*(ty*(3*ty**2 +3)*sxx() -tx*(3*ty**2 +1)*sxy() -tx*(3*ty**2 +1)*syx() +ty*(3*tx**2 +1)*syy()))
def Prediction_txe():
    return(tx + h()*(tx*ty*rx()-(1+tx**2)*ry()) + h()**2*(tx*(3*ty**2 +1)*rxx() -ty*(3*tx**2 +1)*rxy() -ty*(3*tx**2 +1)*ryx() +tx*(3*tx**2 +3)*ryy()))
def Prediction_tye():
    return(ty + h()*((1+ty**2)*rx() -tx*ty*ry()) + h()**2*(ty*(3*(ty**2) +3)*rxx() -tx*(3*ty**2 +1)*rxy() -tx*(3*ty**2 +1)*ryx() +ty*(3*tx**2 +1)*ryy()))

#Energy Loss Prediction

def beta():
    return (q/qbp)/sym.sqrt((q/qbp)**2+m**2) 
def gamma():
    return 1/sym.sqrt(1-(beta())**2)              
def x_d():        
    return sym.log(beta()*gamma(),10)

def T():
    return 2*me*(beta()*gamma())**2/( 1 + 2*(me/m)*sym.sqrt(1+(beta()*gamma())**2+(me/m)**2) ) # Max Kinetic Eenrgy   
     
f_g  = sym.lambdify((qbp), gamma(), "numpy")     
f_b  = sym.lambdify((qbp), beta(), "numpy")     
f_xd = sym.lambdify((qbp), x_d(), "numpy")
f_T  = sym.lambdify((qbp), T(), "numpy")


def EnergylossIron(qbpe):
    global q
    rho=7.874
    ZbA=26.0/55.845 #Z/A
    I=286*10**-9 #in GeV
    xd0=-0.0012
    xd1=3.15
    md=2.96
    a=0.1468
    C0=-4.29
    if qbpe<0:
        qbpe=-qbpe
        #q=-0.303
    #if qbpe>0:
     #   q=0.303
     #print(qbpe,f_b(qbpe),f_g(qbpe),f_xd(qbpe))
    xd=f_xd(qbpe)
    if xd<xd0:
            delta=0.0
    if xd>xd0 and xd<xd1:
            delta=4.6052*xd+C0+(a*((xd1-xd)**md))
    if xd>xd1:
            delta=4.6052*xd+C0
     

    dEds= rho*0.307075/((f_b(qbpe))**2)*ZbA*( 0.5*np.log(2*me*(f_b(qbpe)*f_g(qbpe))**2*f_T(qbpe)/(I**2))-(f_b(qbpe)**2)-delta*0.5) #Bethe Bloch Formula in MeVcm**2/g        
    return dEds

     


def EnergylossAir(qbpe):
    global q
    rho=1.205*10**-3
    ZbA=0.49919   #Z/A
    I=85.7*10**-9 #in GeV  
    xd0=1.742
    xd1=4.28
    md=3.40
    a=0.1091
    C0=-10.6
    if qbpe<0:
        qbpe=-qbpe
        #q=-0.303
    #if qbpe>0:
        #q=0.303
    #print(qbpe,f_b(qbpe),f_g(qbpe),f_xd(qbpe))
    xd=f_xd(qbpe)
    if xd<xd0:
        delta=0.0
    if xd>xd0 and xd<xd1:
        delta=4.6052*xd+C0+(a*((xd1-xd)**md))
    if xd>xd1:
        delta=4.6052*xd+C0
     
    dEds= rho*0.307075/((f_b(qbpe))**2)*ZbA*( 0.5*np.log(2*me*(f_b(qbpe)*f_g(qbpe))**2*f_T(qbpe)/(I**2))-(f_b(qbpe)**2)-delta*0.5) #Bethe Bloch Formula in MeVcm**2/g        
              
#     dEds= rho*0.307075/((beta())**2)*ZbA*( 0.5*sym.log(2*me*(beta()*gamma())**2*T()/(I**2))-(beta())**2)#-delta*0.5) #Bethe Bloch Formula in MeVcm**2/g
    return dEds


#Converting symbolic to mathematical

f_x = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_xe(), "numpy")
f_y = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_ye(), "numpy")
f_tx= sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_txe(),"numpy")
f_ty= sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_tye(),"numpy")

#***************************************************************

#Calculation for Propagator Matrix Elements

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



def Prediction_qbpprimeqbp():
    dl =dz*sym.sqrt(1+tx**2+ty**2)
    return 1+ (fl_2())/fl_1()*dl + 0.5*fl_3()/fl_1()*dl**2
def Prediction_qbpprimex():
    dl =dz*sym.sqrt(1+tx**2+ty**2)
    return K*(fl_1()+fl_2()*dl)*fl()*sym.sqrt(1+tx**2+ty**2)*dl*(-by)
def Prediction_qbpprimey():
    dl =dz*sym.sqrt(1+tx**2+ty**2)
    return K*(fl_1()+fl_2()*dl)*fl()*sym.sqrt(1+tx**2+ty**2)*dl*(bx)
def Prediction_qbpprimetx():
    dl =dz*sym.sqrt(1+tx**2+ty**2)
    return (fl_1()+fl_2()*dl)*dz*(tx/sym.sqrt(1+tx**2+ty**2))
def Prediction_qbpprimety():
    dl =dz*sym.sqrt(1+tx**2+ty**2)
    return  (fl_1()+fl_2()*dl)*dz*(ty/sym.sqrt(1+tx**2+ty**2))


#Converting symbolic to mathematical

f_xprimex   = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_xprimex(), "numpy")
f_xprimey   = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_xprimey(), "numpy")
f_xprimetx  = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_xprimetx(), "numpy")
f_xprimety  = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_xprimety(), "numpy")
f_xprimeqbp = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_xprimeqbp(), "numpy")    
f_yprimex   = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_yprimex(), "numpy")
f_yprimey   = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_yprimey(), "numpy")
f_yprimetx  = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_yprimetx(), "numpy")
f_yprimety  = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_yprimety(), "numpy")
f_yprimeqbp = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_yprimeqbp(), "numpy")    
f_txprimex  = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_txprimex(),"numpy")
f_txprimey  = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_txprimey(),"numpy")
f_txprimetx = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_txprimetx(),"numpy")
f_txprimety = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_txprimety(),"numpy")
f_txprimeqbp= sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_txprimeqbp(), "numpy")    
f_typrimex  = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_typrimex(),"numpy")
f_typrimey  = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_typrimey(),"numpy")
f_typrimetx = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_typrimetx(),"numpy")
f_typrimety = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_typrimety(),"numpy")
f_typrimeqbp= sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_typrimeqbp(), "numpy")    

f_qbpprimex = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_qbpprimex(), "numpy")    
f_qbpprimey = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_qbpprimey(), "numpy")    
f_qbpprimetx = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_qbpprimetx(), "numpy")    
f_qbpprimety = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_qbpprimety(), "numpy")    
f_qbpprimeqbp = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_qbpprimeqbp(), "numpy")    
    



#Propagator matrix calculation
prop_f =[[None for j in range (5)] for i in range (5)]
prop_f=np.array(prop_f,dtype=float)

def Propagator(xe,ye,txe,tye,qbpe,dze):
    
    global prop_f    
    
    prop_f[0][0]=f_xprimex(xe,ye,txe,tye,qbpe,dze)
    prop_f[0][1]=f_xprimey(xe,ye,txe,tye,qbpe,dze)
    prop_f[0][2]=f_xprimetx(xe,ye,txe,tye,qbpe,dze)
    prop_f[0][3]=f_xprimety(xe,ye,txe,tye,qbpe,dze)
    prop_f[0][4]=f_xprimeqbp(xe,ye,txe,tye,qbpe,dze)
    prop_f[1][0]=f_yprimex(xe,ye,txe,tye,qbpe,dze)
    prop_f[1][1]=f_yprimey(xe,ye,txe,tye,qbpe,dze)
    prop_f[1][2]=f_yprimetx(xe,ye,txe,tye,qbpe,dze)
    prop_f[1][3]=f_yprimety(xe,ye,txe,tye,qbpe,dze)
    prop_f[1][4]=f_yprimeqbp(xe,ye,txe,tye,qbpe,dze)
    prop_f[2][0]=f_txprimex(xe,ye,txe,tye,qbpe,dze)
    prop_f[2][1]=f_txprimey(xe,ye,txe,tye,qbpe,dze)
    prop_f[2][2]=f_txprimetx(xe,ye,txe,tye,qbpe,dze)
    prop_f[2][3]=f_txprimety(xe,ye,txe,tye,qbpe,dze)
    prop_f[2][4]=f_txprimeqbp(xe,ye,txe,tye,qbpe,dze)
    prop_f[3][0]=f_typrimex(xe,ye,txe,tye,qbpe,dze)
    prop_f[3][1]=f_typrimey(xe,ye,txe,tye,qbpe,dze)
    prop_f[3][2]=f_typrimetx(xe,ye,txe,tye,qbpe,dze)
    prop_f[3][3]=f_typrimety(xe,ye,txe,tye,qbpe,dze)
    prop_f[3][4]=f_typrimeqbp(xe,ye,txe,tye,qbpe,dze)
    prop_f[4][0]=f_qbpprimex(xe,ye,txe,tye,qbpe,dze)
    prop_f[4][1]=f_qbpprimey(xe,ye,txe,tye,qbpe,dze)
    prop_f[4][2]=f_qbpprimetx(xe,ye,txe,tye,qbpe,dze)
    prop_f[4][3]=f_qbpprimety(xe,ye,txe,tye,qbpe,dze)
    prop_f[4][4]=f_qbpprimeqbp(xe,ye,txe,tye,qbpe,dze)

    
    return ()

#***************************************************************
#Defining the Random Error Matrix

def CMS_I():                           #Highland-Lynch-Dahl variance formula
    dl=dz*sym.sqrt(1+tx**2+ty**2)
    Z=26.0
    ls=17.57*((Z+1)/Z)*(289*Z**(-1/2))/(159*Z**(-1/3))
    p=q/qbp
    return (0.015/(beta()*p))**2*(dl/ls)

def cov_txtx():                 
    return (1+tx**2)*(1+tx**2+ty**2)*CMS_I()
def cov_tyty():
    return (1+ty**2)*(1+tx**2+ty**2)*CMS_I()
def cov_txty():
    return tx*ty*(1+tx**2+ty**2)*CMS_I()


c_txtx = sym.lambdify((tx,ty,qbp,dz), cov_txtx(), "numpy")    
c_tyty = sym.lambdify((tx,ty,qbp,dz), cov_tyty(), "numpy")    
c_txty = sym.lambdify((tx,ty,qbp,dz), cov_txty(), "numpy")    

def Prediction_xprimep():  
    p=q/qbp
    return ((-K*q/p**2*(1+ (tx)**2 +(ty)**2)**0.5)*(tx*ty*sx()-(1+tx**2)*sy()) + -2*(K*q*(1+ (tx)**2 +(ty)**2)**0.5)**2/(p**3)*(tx*(3*ty**2 +1)*sxx() -ty*(3*tx**2 +1)*sxy() -ty*(3*tx**2 +1)*syx() +tx*(3*tx**2 +3)*syy()))
def Prediction_yprimep():
    p=q/qbp
    return((-K*q/p**2*(1+ (tx)**2 +(ty)**2)**0.5)*((1+ty**2)*sx() -tx*ty*sy()) + -2*(K*q*(1+ (tx)**2 +(ty)**2)**0.5)**2/(p**3)*(ty*(3*ty**2 +3)*sxx() -tx*(3*ty**2 +1)*sxy() -tx*(3*ty**2 +1)*syx() +ty*(3*tx**2 +1)*syy()))
def Prediction_txprimep():
    p=q/qbp
    return((-K*q/p**2*(1+ (tx)**2 +(ty)**2)**0.5)*(tx*ty*rx()-(1+tx**2)*ry()) + -2*(K*q*(1+ (tx)**2 +(ty)**2)**0.5)**2/(p**3)*(tx*(3*ty**2 +1)*rxx() -ty*(3*tx**2 +1)*rxy() -ty*(3*tx**2 +1)*ryx() +tx*(3*tx**2 +3)*ryy()))
def Prediction_typrimep():
    p=q/qbp
    return((-K*q/p**2*(1+ (tx)**2 +(ty)**2)**0.5)*((1+ty**2)*rx() -tx*ty*ry()) + -2*(K*q*(1+ (tx)**2 +(ty)**2)**0.5)**2/(p**3)*(ty*(3*(ty**2) +3)*rxx() -tx*(3*ty**2 +1)*rxy() -tx*(3*ty**2 +1)*ryx() +ty*(3*tx**2 +1)*ryy()))

f_xprimep   = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_xprimep(), "numpy")
f_yprimep   = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_yprimep(), "numpy")
f_txprimep   = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_txprimep(), "numpy")
f_typrimep   = sym.lambdify((x,y,tx,ty,qbp,dz), Prediction_typrimep(), "numpy")


def xi(qbpe):       #mean energy loss in GeV
    rho=7.874
    ZbA=26.0/55.845 
    return (0.1534*q**2*ZbA/f_b(qbpe)**2)*rho*56*10**-4

def sig2E(qbpe):    #variance of the Gaussain distribution in GeV^2
    return  xi(qbpe)*f_T(qbpe)*10**-3*(1-(f_b(qbpe)**2)/2)


Q_l =[[None for j in range (5)] for i in range (5)]
Q_l=np.array(Q_l,dtype=float)


def Scattering(xe,ye,txe,tye,qbpe,dze,l,D):  #Random Error Matrix
    global Q_l
    
    pe=q/qbpe
   
    Q_l[0][0]= c_txtx(txe,tye,qbpe,dze)*(l**3)/3
    Q_l[0][1]= c_txty(txe,tye,qbpe,dze)*(l**3)/3
    Q_l[0][2]= c_txtx(txe,tye,qbpe,dze)*(l**2)*D/2
    Q_l[0][3]= c_txty(txe,tye,qbpe,dze)*(l**2)*D/2
    Q_l[0][4]=-f_xprimep(xe,ye,txe,tye,qbpe,dze)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
    
    Q_l[1][0]=c_txty(txe,tye,qbpe,dze)*(l**3)/3
    Q_l[1][1]=c_tyty(txe,tye,qbpe,dze)*(l**3)/3
    Q_l[1][2]=c_txty(txe,tye,qbpe,dze)*(l**2)*D/2
    Q_l[1][3]=c_tyty(txe,tye,qbpe,dze)*(l**2)*D/2
    Q_l[1][4]=-f_yprimep(xe,ye,txe,tye,qbpe,dze)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)

    Q_l[2][0]=c_txtx(txe,tye,qbpe,dze)*(l**2)*D/2
    Q_l[2][1]=c_txty(txe,tye,qbpe,dze)*(l**2)*D/2
    Q_l[2][2]=c_txtx(txe,tye,qbpe,dze)*l
    Q_l[2][3]=c_txty(txe,tye,qbpe,dze)*l
    Q_l[2][4]=-f_txprimep(xe,ye,txe,tye,qbpe,dze)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
    
    Q_l[3][0]=c_txty(txe,tye,qbpe,dze)*(l**2)*D/2
    Q_l[3][1]=c_tyty(txe,tye,qbpe,dze)*(l**2)*D/2
    Q_l[3][2]=c_txtx(txe,tye,qbpe,dze)*l
    Q_l[3][3]=c_tyty(txe,tye,qbpe,dze)*l
    Q_l[3][4]=-f_typrimep(xe,ye,txe,tye,qbpe,dze)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)

    Q_l[4][0]=-f_xprimep(xe,ye,txe,tye,qbpe,dze)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
    Q_l[4][1]=-f_yprimep(xe,ye,txe,tye,qbpe,dze)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
    Q_l[4][2]=-f_txprimep(xe,ye,txe,tye,qbpe,dze)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
    Q_l[4][3]=-f_typrimep(xe,ye,txe,tye,qbpe,dze)*sig2E(qbpe)*q*(E[-1]**2)/(pe**4)
    Q_l[4][4]=E[-1]**2/(pe**6)*sig2E(qbpe)

    Q_l=np.array(Q_l, dtype=float)

#***************************************************************

#Covariance Matrix 

covp =10**6*np.identity(5,  dtype=np.float) #Initial prediction cov
covp=np.array(covp)

def Covariance():
    
    global covp,prop_f 

    covp = prop_f@covp@(prop_f.T)+Q_l
    covp=np.array(covp, dtype=float)
    
    
#***************************************************************

ide=np.identity(5,dtype=float)

def Kalman(i,temp_stv):   
    
    temp_stv=np.array(temp_stv)
    global covp
      
    #Kalman Gain matrix
    Kf=covp@(Hk.T)@(np.linalg.inv(np.array((Hk@covp@(Hk.T)+Vk),dtype=float)))
  
    #Kalman estimate for state vector       
    temp_stv=(temp_stv.T+Kf@(mk[i,:].T-Hk@(temp_stv.T))).T
    
    #Kalman estimate for filtered error covariance
    covp=(ide-Kf@Hk)@covp@((ide-Kf@Hk).T)+Kf@Vk@(Kf.T)
    
    return temp_stv


#***************************************************************
#Appending the new state vector value

def vector_updation(i,temp,z):    
    global stv
    stv[i][0],stv[i][1],stv[i][2],stv[i][3],stv[i][4]=temp
    xpp.append(stv[i][0])
    ypp.append(stv[i][1])
    zpp.append(z)
    

#***************************************************************

def Plots():
    
    #X coordinates

    #plt.plot(xm,zm)
    #plt.plot(xpp,zpp,linestyle='dashed',c='red') 
    #plt.scatter(xm,zm)
    #plt.scatter(xpp,zpp) 

    #plt.legend(("Simulated x",'Predicted x' ))
    #plt.xlabel('x - values') 
    #plt.ylabel('y - values') 
    #plt.title('Prediction vs Simulation for x coordinates (20 GeV)') 
    
    #**************************
    
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
# Main loop starts here


E=[]
E.append(E_inc)                #initial value of energy
zpp,xpp,ypp=[],[],[]           #data saved for plotting
ze=zm[0]                       #initial value for z for plotting
index=4

for iterations in range(index): # controls the number of iterations; 
    temp_stv=stv[0]
    for i in range(len(df1)):  #Main loop for 150 (air+iron+air) combo
        
        D=1
        x_e,y_e,tx_e,ty_e,qbp_e=temp_stv

        for j in range(58):    #Subloop for each combo            
                       
            if (j==0 or j==57):
                dz_e=20*D            #in mm for air gap between rpc and iron      
                material="Air"
            else:
                dz_e=1*D             #in mm for iron plate
                material="Iron"
            
        
            x_e = f_x(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating x
            y_e = f_y(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating y
            tx_e = f_tx(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)     #Updating tx
            ty_e = f_ty(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)     #Updating ty
            
            dl =dz_e*np.sqrt(1+tx_e**2+ty_e**2)          #differential arc length
        
            
            if (material=="Iron"):
                dEds=EnergylossIron(qbp_e)
            else:
                dEds=EnergylossAir(qbp_e)          #Assuming no energy loss in air       
            
            E_cal = E[-1]-(dEds*dl*10**-4)                #Updating Energy            
            E.append(E_cal)
          
            if ((E_cal**2-m**2)<0):
                qbp_e=qbp_e
            else:
                qbp_e = q/np.sqrt(E_cal**2-m**2)              #Updating q/p    
            
            Propagator(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)   #Updating propagator matrix
            Scattering(x_e,y_e,tx_e,ty_e,qbp_e,dz_e,dl,D)

            Covariance()                                  #Updating the Covariance matrix
    
            ze = ze - dz_e                                #Updating z
            temp_stv=x_e,y_e,tx_e,ty_e,qbp_e    
    
            #printing the propagator matrix for each j   
            #print('\n________________________________________________________________________________________________________________________________\n',i,j,'propf \t',prop_f[0][0],prop_f[0][1],prop_f[0][2],prop_f[0][3],prop_f[0][4],'\n','\t',prop_f[1][0],prop_f[1][1],prop_f[1][2],prop_f[1][3],prop_f[1][4],'\n','\t',prop_f[2][0],prop_f[2][1],prop_f[2][2],prop_f[2][3],prop_f[2][4],'\n','\t',prop_f[3][0],prop_f[3][1],prop_f[3][2],prop_f[3][3],prop_f[3][4],'\n','\t',prop_f[4][0],prop_f[4][1],prop_f[4][2],prop_f[4][3],prop_f[4][4],'\n')        

            #printing the random error matrix for each j   
            #print('\n________________________________________________________________________________________________________________________________\n',i,j,'frwd \t',Q_l[0][0],Q_l[0][1],Q_l[0][2],Q_l[0][3],Q_l[0][4],'\n','\t',Q_l[1][0],Q_l[1][1],Q_l[1][2],Q_l[1][3],Q_l[1][4],'\n','\t',Q_l[2][0],Q_l[2][1],Q_l[2][2],Q_l[2][3],Q_l[2][4],'\n','\t',Q_l[3][0],Q_l[3][1],Q_l[3][2],Q_l[3][3],Q_l[3][4],'\n','\t',Q_l[4][0],Q_l[4][1],Q_l[4][2],Q_l[4][3],Q_l[4][4],'\n')        
            

            #printing the covariance matrix for each j   
            #print('\n________________________________________________________________________________________________________________________________\n',i,j,'\t',covp[0][0],covp[0][1],covp[0][2],covp[0][3],covp[0][4],'\n','\t',covp[1][0],covp[1][1],covp[1][2],covp[1][3],covp[1][4],'\n','\t',covp[2][0],covp[2][1],covp[2][2],covp[2][3],covp[2][4],'\n','\t',covp[3][0],covp[3][1],covp[3][2],covp[3][3],covp[3][4],'\n','\t',covp[4][0],covp[4][1],covp[4][2],covp[4][3],covp[4][4],'\n')            
        
        temp_stv=Kalman(i,temp_stv) #Kalman filtering

        E.append(np.sqrt((q/temp_stv[4])**2+m**2))      #updating the energy with new q/p

        #print(iterations,i,temp_stv[0],temp_stv[1],temp_stv[2],temp_stv[3],temp_stv[4],'\t',E[-1])
        
#For Smooothing    
    for i in reversed(range(len(df1))): #-1, -1, -1)): #Main loop for 150 (air+iron+air) combo forward+backwards
        
        D=-1
        x_e,y_e,tx_e,ty_e,qbp_e=temp_stv

        for j in range(58):    #Subloop for each combo            
                       
            if (j==0 or j==57):
                dz_e=20*D            #in mm for air gap between rpc and iron      
                material="Air"
            else:
                dz_e=1*D             #in mm for iron plate
                material="Iron"
            
        
            x_e = f_x(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating x
            y_e = f_y(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)       #Updating y
            tx_e = f_tx(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)     #Updating tx
            ty_e = f_ty(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)     #Updating ty
            
            dl =dz_e*np.sqrt(1+tx_e**2+ty_e**2)          #differential arc length
           
            if (material=="Iron"):
                dEds=EnergylossIron(qbp_e)
            else:
                dEds=EnergylossAir(qbp_e)          #Assuming no energy loss in air       
        
            E_cal = E[-1]-(dEds*dl*10**-4)                #Updating Energy
            E.append(E_cal)
            if ((E_cal**2-m**2)<0):
                qbp_e=qbp_e
            else:
                qbp_e = q/np.sqrt(E_cal**2-m**2)              #Updating q/p    
            
            Propagator(x_e,y_e,tx_e,ty_e,qbp_e,dz_e)   #Updating propagator matrix
            Scattering(x_e,y_e,tx_e,ty_e,qbp_e,dz_e,dl,D)

            Covariance()                                  #Updating the Covariance matrix
    
            ze = ze - dz_e                                #Updating z
            temp_stv=x_e,y_e,tx_e,ty_e,qbp_e    
    
            #printing the propagator matrix for each j   
            #print('\n________________________________________________________________________________________________________________________________\n',i,j,'propf \t',prop_f[0][0],prop_f[0][1],prop_f[0][2],prop_f[0][3],prop_f[0][4],'\n','\t',prop_f[1][0],prop_f[1][1],prop_f[1][2],prop_f[1][3],prop_f[1][4],'\n','\t',prop_f[2][0],prop_f[2][1],prop_f[2][2],prop_f[2][3],prop_f[2][4],'\n','\t',prop_f[3][0],prop_f[3][1],prop_f[3][2],prop_f[3][3],prop_f[3][4],'\n','\t',prop_f[4][0],prop_f[4][1],prop_f[4][2],prop_f[4][3],prop_f[4][4],'\n')        

            #printing the random error matrix for each j   
            #print('\n________________________________________________________________________________________________________________________________\n',i,j,'bkwrd \t',Q_l[0][0],Q_l[0][1],Q_l[0][2],Q_l[0][3],Q_l[0][4],'\n','\t',Q_l[1][0],Q_l[1][1],Q_l[1][2],Q_l[1][3],Q_l[1][4],'\n','\t',Q_l[2][0],Q_l[2][1],Q_l[2][2],Q_l[2][3],Q_l[2][4],'\n','\t',Q_l[3][0],Q_l[3][1],Q_l[3][2],Q_l[3][3],Q_l[3][4],'\n','\t',Q_l[4][0],Q_l[4][1],Q_l[4][2],Q_l[4][3],Q_l[4][4],'\n')        
                    
            #printing the covariance matrix for each j   
            #print('\n________________________________________________________________________________________________________________________________\n',i,j,'\t',covp[0][0],covp[0][1],covp[0][2],covp[0][3],covp[0][4],'\n','\t',covp[1][0],covp[1][1],covp[1][2],covp[1][3],covp[1][4],'\n','\t',covp[2][0],covp[2][1],covp[2][2],covp[2][3],covp[2][4],'\n','\t',covp[3][0],covp[3][1],covp[3][2],covp[3][3],covp[3][4],'\n','\t',covp[4][0],covp[4][1],covp[4][2],covp[4][3],covp[4][4],'\n')            
        
        temp_stv=Kalman(i,temp_stv) #Kalman filtering 

        E.append(np.sqrt((q/temp_stv[4])**2+m**2)) #updating the energy with new q/p
        
        if iterations==(index-1):               
            vector_updation(i,temp_stv,ze)      #saves the state vector only at the last iteration
        
        #print(iterations,i,temp_stv[0],temp_stv[1],temp_stv[2],temp_stv[3],temp_stv[4],'\t',E[-1])

#Plots()
end = time.time()
print('Energy of the muon = \t',E[-1])
print(f"Runtime of the program is {end - start}")
