function y=dydx(t,x)
    global L g omega_0

    vx=x(1);
    vz=x(2);
    wy=x(3); 
    v1=x(4);
    v2=x(5);
    x1=x(6);
    x2=x(7);
    th=x(8);

    zcm=-(x1+x2)/4;   
    % Make zcm=0 and rigid-elastic couplings disappear for this problem.

    % Compute mass matrix
    E=[[   1    0              -zcm    0    0 0 0 0];
       [   0    1                 0 -1/4 -1/4 0 0 0];
       [-zcm    0 L^2+(x1^2+x2^2)/4  L/4 -L/4 0 0 0];
       [   0 -1/4               L/4  1/4    0 0 0 0];
       [   0 -1/4              -L/4    0  1/4 0 0 0];
       [   0    0                 0    0    0 1 0 0];
       [   0    0                 0    0    0 0 1 0];
       [   0    0                 0    0    0 0 0 1]];

   A=[[   0     wy       0      -wy/2     -wy/2             0             0 0];
       [ -wy      0 -wy*zcm         0         0             0             0 0];
       [   0 2*wy*zcm     0  .5*wy*x1  .5*wy*x2             0             0 0];
       [wy/4      0 -.25*wy*x1      0         0 (omega_0^2)/4             0 0];
       [wy/4      0 -.25*wy*x2      0         0             0 (omega_0^2)/4 0];
       [   0      0       0        -1         0             0             0 0];
       [   0      0       0         0        -1             0             0 0];
       [   0      0      -1         0         0             0             0 0]];

    b=g*[     -sin(th);
               cos(th);
           zcm*sin(th);
        -(1/4)*cos(th);
        -(1/4)*cos(th);
        0; 0; 0];
    y=E\(b-A*x);
end