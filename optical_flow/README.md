# OPTICAL FLOW
# Implementation Lukas-Kanade and Horn-Schunck methods.

## AUTOR
* [**VICENTE GILABERT MAÑO**](https://www.linkedin.com/in/vgilabert/)


## DESCRIPTION
Implementation of Dense **Lukas-Kanade** algorithm with least squares calculation and direct calculation. This is a Local metoth using a window sliding. Comparison of execution times (check report).

Implementation of the **Horn-Schunck** algorithm by direct computation and parameter checking (lambda and iterations). This is a global, dense, variational calculus method. Creation of a stopping criterion with the norm of the velocity vector U.


## FILES
* **optical_flow.py:** where the algorithms and representations have been implemented as functions.
* **main_optical_flow.py:** application to execute the algorithms on videos or consecutive images.
* **test_optical_flow.py:** Where all the tests have been carried out for the practice with only 2 consecutive frames. 


## STRUCTURE OF THE PROJECT
```
.
├── video1
├── video3
├── README.md
├── main_optical_flow.py
├── optical_flow.py
├── requirements.txt
├── test_optical_flow
```


## RESULTS
Read the full report (spanish) -> [PDF](docs/report.pdf)

#### LUKAS-KANADE
windows size = 25  

![image](https://user-images.githubusercontent.com/44602177/160464668-2620fc80-da63-4175-a9ff-4b65ced4e0c8.png)

#### HORN-SCHUNCK
lambda = 0.1 / Iterations = 25  

![image](https://user-images.githubusercontent.com/44602177/160464824-59551f79-9081-496d-b977-a51a167b1599.png)




