# Project Framework

## Process:
### Design Variables from Optimizer
- joint 1 -> fixed
- joint 2 -> rpy, xyz relative to j1
- joint 3 -> rpy, xyz relative to j2
- link 1 -> ID, wall-thickness
- link 2 -> ID, wall-thickness
- actuator 1 -> power, gear ratio
- actuator 2 -> power, gear ratio
- actuator 3 -> power, gear ratio

### Pre-processing 
- calculate cumulative transforms. J2->J1. J3->J1. 
- extract screw axes X
- construct jacobian X

### functions
- Toppra
    - JS spline fitting from waypoints
        - Inverse kinematics
        - forward kinematics 
- forward dynamics 
    - mass properties
        - Inertia calculator
    - motor internal inertia modeling 

## Next Steps
- Should test robot reachability optimization sample problem:
    - Input Variables: xyz, rpy for J2 and J3
    - Output: Magnitude of total error (norm of 6 vector).

## Open Questions
- What is better: specifying relative joint transforms or global ones? 
    - I think relative ones...