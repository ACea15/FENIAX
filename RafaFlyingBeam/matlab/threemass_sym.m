clear all, close all
global L g omega_0
L=1; g=10; omega_0=2; 

V=1;    % Initial velocity.
T=2*pi/(sqrt(2)*omega_0);
dt=T/100;

% Problem with 8 states (vx, vz, omega_y, dot{xi}_1, dot{xi}_2, xi_1, xi_2, theta)
% Define initial values
x0=zeros(8,1);
x0(1,1)=V;
x0(4,1)=L*omega_0/100;
x0(5,1)=L*omega_0/100;
x0(8,1)=0*pi/180;

% Loop through the initial angular velocity.
wy0_max=omega_0/4;
opts = odeset('RelTol',1e-6,'AbsTol',1e-8);
for wy0= 0:wy0_max/5:wy0_max;
    x0(3,1)=wy0; 
    [t,x]=ode45(@threemass_dydx,[0:dt:2*T],x0,opts);

    % Displacements 
    figure(1)
      subplot(3,1,1)
          plot(t/T,x(:,8)*180/pi, 'Color' , 0.7*(wy0/wy0_max)*[1 1 1], ...
              'LineWidth', 2), hold on
          ylabel('$$\theta \textup{(deg)}$$','FontSize',16,'Interpreter','latex')
          yticks([0 45 90 135])    
     subplot(3,1,2)
         plot(t/T,x(:,6)+x(:,7), 'Color' , 0.7*(wy0/wy0_max)*[1 1 1], ...
             'LineWidth', 2), hold on
         ylabel('$$(\xi_1+\xi_2)/l$$','FontSize',16,'Interpreter','latex')
     subplot(3,1,3)
        plot(t/T,x(:,6)-x(:,7), 'Color' , 0.7*(wy0/wy0_max)*[1 1 1], ...
            'LineWidth', 2), hold on
        ylabel('$$(\xi_1-\xi_2)/l$$','FontSize',16,'Interpreter','latex')
        xlabel('$$t/T$$','FontSize',16,'Interpreter','latex')

    % Compute velocities of CM in global frame
    figure(2)
       % Vcm in body-attached components.
       vxcm=x(:,1)-(1/4)*x(:,3).*(x(:,6)+x(:,7));
       vzcm=x(:,2)-(1/4)*(x(:,4)+x(:,5));
       % Global components: Rotate with theta.
       vx= vxcm.*cos(x(:,8))+vzcm.*sin(x(:,8));
       vz=-vxcm.*sin(x(:,8))+vzcm.*cos(x(:,8));

       % Compare to analytical solution
       vxref= V*cos(x0(8,1)) - (1/4)*sin(x0(8,1))*(x0(4,1)+x0(5,1));
       vzref=-V*sin(x0(8,1)) - (1/4)*cos(x0(8,1))*(x0(4,1)+x0(5,1))+g*t;
       plot(t,vx-vxref,'r'), hold on
       plot(t,vz-vzref,'b')
end
figure
     subplot(2,1,1)
         plot(t(1:end)/T,x(:,3)/wy0_max, 'LineWidth', 2), hold on
         ylabel('$${\omega}_y/(\omega_{y0})$$','FontSize',16,'Interpreter','latex')
     % subplot(3,1,2)
     %     plot(t(1:end-1)/T,((x(2:end,3)-x(1:end-1,3))/dt)/(wy0_max^2), 'LineWidth', 2), hold on
     %     ylabel('$$\dot{\omega}_y/(omega_{y0})^2$$','FontSize',16,'Interpreter','latex')
     subplot(2,1,2)
        plot(t/T,x(:,6)-x(:,7),'LineWidth', 2), hold on
        ylabel('$$(\xi_1-\xi_2)/l$$','FontSize',16,'Interpreter','latex')
        xlabel('$$t/T$$','FontSize',16,'Interpreter','latex')


