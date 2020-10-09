# ical-kalman-filter
 A python based implementation of a kalman filter based on the paper for the same by Kolahal Bhattacharya

An in-house developed software (Simple) which runs Geant4 in its background was utilized for generating the input data.

temp_stv: is a dummy variable which takes in the state vector for a given iteration(j)

#----------------------------------------------------------------------------
FUNCTIONS:
#----------------------------------------------------------------------------

Propagator(x_e,y_e,tx_e,ty_e,qp_e,dz_e,dl,b_x): (line 336)

Function used to define the propagator matrix F_k. 
I/P:Prior estimate of the state vector,step size, differential length and magnetic field along x
O/P:None (globally redefines the propagator matrix F_k with each iteration(j). )
#--------------------------------------
Random_Error(x_e,y_e,tx_e,ty_e,qp_e,dz_e,l,D,material,b_x): (line 444)

Function used to define the random error matrix Q_l.
I/P:Prior estimate of the state vector,step size, differential length , material and magnetic field along x
O/P: None (globally redefines the Random Error Matrix Q_l with each iteration(j). If material is air, random error is a zero matrix. )
#--------------------------------------
Covariance():(line 496)

Function used to define the Covariance matrix cov_C for prior estimates.
I/P: Prior estimate of the state vector
O/P: None ( globally redefines the Covariance Matrix cov_C for each iteration(j). )
#--------------------------------------
Kalman(i,X_k)

Function to implement the Kalman Filter algorithm.
I/P: Index of the plane, prior estimate of the state vector
O/P: Posterior Estimate( globally redines the Covariance matrix cov_C corresponding to the posterior estimate)
#--------------------------------------