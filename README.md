# Kalman Filter
 Welcome to the git repository for a python based implementation of kalman filter based on the works by Kolahal Bhattacharya (http://www.ino.tifr.res.in/ino/theses/2015/Thesis_Kolahal.pdf)

## Goal:
  Reconstructing the tracks of muons within the ICAL detector to estimate the incident energy of the same. 

## What is a Kalman Filter?
It is recursive fitting technique used to derive the best estimate of a system's state at a given point from the information collected at multiple observation points. Kalman filter is used generally for systems which are under the influence of random disturbances(process noise) during its evolution following an equation of motion. It is used to extract the track parameters called the state vector of the particle is the being estimated at every interaction point.


## What is state vector?
A state vector is a vector which defines the state of the particle at each point. It contains all possible information about the particle as it passes through the detector. The components of the state vectors described for the ICAL are the positional coordinates (x, y), their local slopes( t<sub>x</sub>, t<sub>y</sub>) and the charge by momentum ration (q/p). We seek to obtain the optimal estimates of the state vector at every interaction point, and are particulary interested at knowing the state vector at the vertex.

## Reading Measured Data
An in-house developed software (Simple) which runs Geant4 in its background was utilized for generating the input data. The code has been tested for anti-muons of energy ranging from 1-10 GeV. The data files are saved in the following format "1GeVmu+100eve.txt" within the dataGeV folder.

## Prediction Equations
Extrapolation formulae for tracing the track of a charged partiicle passing through a magnetic field has been used to predict the prior estimates of the first four components of the state vector. The Bethe Bloch Formula is used for the prediction of q/p.

## Code Specifics
The code is modularised in 6 classes,

### Data Handling
The input file is read, the number of events is set and the number of iterations of the loop is also set in this snippet. The measured data includes the positional coordinates of the RPC hit of the particle. These are then stored in an array $M_k$ to be used in the Kalman Filter module.

### Symbolic_Expressions
Symbolic functions which are used for the prediction equations and the error calculation are all mentioned within this class. These functions are then invoked in the beginning of main and lambdified to mathematical functions to save computational time.

### Loop_Initialisation
Defines the material, thickness(step size) and the magnetic field of for each iteration. A total of 58 iterations are considered between two RPCs(air+ 56\*iron+ air) 

### Bethe_Bloch
It is a subclass of Loop initialisation. Computes the energy loss due to ionisation for each iteration using the Bethe Bloch Formula.

### Prior_Predictions
It is a subclass of Bethe Bloch. Consists of the f_predictions which computes the prior estimates of the state vector and the corresponding Covariance and Propagator Matrices.
 
### Kalman_Filter
Contains the Kalman Function which computes the posterior estimates and the corresponding covariance matrix for the statevectors.





