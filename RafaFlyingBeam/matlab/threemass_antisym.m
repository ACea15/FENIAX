clear all, close all
global L g omega_0
L=1; g=10; omega_0=sqrt(12);

T=2*pi/(sqrt(2)*omega_0);
dt=T/100;

% Problem with 8 states (vx, vz, omega_y, dot{xi}_1, dot{xi}_2, xi_1, xi_2, theta)
% Define initial values
x0=zeros(8,1);
x0(1,1)=1;
x0(3,1)=omega_0/4;
x0(8,1)=0*pi/180;

% Loop through the initial velocities of the edge masses.
xi10_max=0.25*omega_0; N=5;
opts = odeset('RelTol',1e-5,'AbsTol',1e-7);
for xi10= 0:xi10_max/N:xi10_max
    x0(4,1)=xi10;
    x0(5,1)=-xi10;
    [t,x]=ode45(@threemass_dydx,[0:dt:2*T],x0,opts);

    % Displacements 
    figure(1)
    subplot(3,1,1)
        plot(t/T,x(:,3)/(omega_0/4), 'Color' , 0.7*(xi10/xi10_max)*[1 1 1], ...
            'LineWidth', 2), hold on
        ylabel('$$\omega_y/(\omega_0/4)$$','FontSize',16,'Interpreter','latex')
    subplot(3,1,2)
        plot(t/T,x(:,6), 'Color' , 0.7*(xi10/xi10_max)*[1 1 1], ...
            'LineWidth', 2), hold on
        ylabel('$$\xi_1/l$$','FontSize',16,'Interpreter','latex')
    subplot(3,1,3)
        plot(t/T,T*x(:,4), 'Color' , 0.7*(xi10/xi10_max)*[1 1 1], ...
            'LineWidth', 2), hold on
        ylabel('$$T\dot{\xi}_1/l$$','FontSize',16,'Interpreter','latex')
        xlabel('$$t/T$$','FontSize',16,'Interpreter','latex')

    % Compute velocities of CM in global frame
    figure(2)
       % Local components.
       vxcm=x(:,1)-(1/4)*x(:,3).*(x(:,6)+x(:,7));
       vzcm=x(:,2)-(1/4)*(x(:,4)+x(:,5));
       % Global components
       vx= vxcm.*cos(x(:,8))+vzcm.*sin(x(:,8));
       vz=-vxcm.*sin(x(:,8))+vzcm.*cos(x(:,8));
       plot(t,vx-cos(x0(8,1)),'r'), hold on
       plot(t,vz+sin(x0(8,1))-g*t,'b')


     figure(3)
      omega_star=x(:,3)+(x(:,4)-x(:,5))/(4*L);
      plot(t/T,omega_star/omega_star(1), ...
          'Color' , 0.7*(xi10/xi10_max)*[1 1 1], ...
            'LineWidth', 2), hold on
      xlabel('$$t/T$$','FontSize',16,'Interpreter','latex')
      ylabel('$$\omega_y^*/\omega_y^*(0)$$','FontSize',16,'Interpreter','latex')
end
figure(1)