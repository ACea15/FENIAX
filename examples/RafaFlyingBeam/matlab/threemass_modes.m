clear all, close all

N=100;
x=-1:1/N:1;
xmass=[-1 0 1];

phi0=zeros(size(x));
phi0mass=[0 0 0];

figure(1)
subplot(2,2,1)
    phi1=-ones(size(x));
    phi1mass=-ones(size(xmass));
    plot(x,phi0,'-','LineWidth',3,'Color',[0.5 0.5 0.5]), hold on
    plot(xmass,phi0mass,'o','MarkerSize',12,'MarkerFaceColor',[0.5 0.5 0.5])
    plot(x,phi1,'k-','LineWidth',3)
    plot(xmass,phi1mass,'o','MarkerSize',12,'MarkerFaceColor','k')
    axis([-1 1 -1.1 1.1])
    ylabel('$$\Phi_2$$','FontSize',16,'Interpreter','latex')

subplot(2,2,3)
    phi2=-x;
    phi2mass=[1 0 -1];
    plot(x,phi0,'-','LineWidth',3,'Color',[0.5 0.5 0.5]), hold on
    plot(xmass,phi0mass,'o','MarkerSize',12,'MarkerFaceColor',[0.5 0.5 0.5])
    plot(x,phi2,'k-','LineWidth',3)
    plot(xmass,phi2mass,'o','MarkerSize',12,'MarkerFaceColor','k')
    axis([-1 1 -1.1 1.1])
    ylabel('$$\Phi_3$$','FontSize',16,'Interpreter','latex')
    xlabel('$$x/l$$','FontSize',16,'Interpreter','latex')

subplot(2,2,2)
    phi3(1:N)=-0.5*(x(1:N).^3.*(3+x(1:N)));
    phi3(N+1:2*N+1)=0;
    phi3=phi3-1/4+x/4;
    phi3mass=[phi3(1) phi3(N+1) phi3(2*N+1)];
    plot(x,phi0,'-','LineWidth',3,'Color',[0.5 0.5 0.5]), hold on
    plot(xmass,phi0mass,'o','MarkerSize',12,'MarkerFaceColor',[0.5 0.5 0.5])
    plot(x,phi3,'k-','LineWidth',3)
    plot(xmass,phi3mass,'o','MarkerSize',12,'MarkerFaceColor','k')
    axis([-1 1 -1.1 1.1])
    ylabel('$$\Phi_4$$','FontSize',16,'Interpreter','latex')

subplot(2,2,4)
    phi4(N:2*N+1)=0.5*x(N:2*N+1).^3.*(3-x(N:2*N+1));
    phi4=phi4-1/4*(1+x);
    phi4mass=[phi4(1) phi4(N+1) phi4(2*N+1)];
    plot(x,phi0,'-','LineWidth',3,'Color',[0.5 0.5 0.5]), hold on
    plot(xmass,phi0mass,'o','MarkerSize',12,'MarkerFaceColor',[0.5 0.5 0.5])
    plot(x,phi4,'k-','LineWidth',3)
    plot(xmass,phi4mass,'o','MarkerSize',12,'MarkerFaceColor','k')
    axis([-1 1 -1.1 1.1])
    ylabel('$$\Phi_5$$','FontSize',16,'Interpreter','latex')
    xlabel('$$x/l$$','FontSize',16,'Interpreter','latex')


figure(2)
subplot(2,1,1)
   phisym=(phi3+phi4);
   phisym_mass=[phisym(1) phisym(N+1) phisym(2*N+1)]
    plot(x,phi0,'-','LineWidth',3,'Color',[0.5 0.5 0.5]), hold on
    plot(xmass,phi0mass,'o','MarkerSize',12,'MarkerFaceColor',[0.5 0.5 0.5])
    plot(x,phisym,'k-','LineWidth',3)
    plot(xmass,phisym_mass,'o','MarkerSize',12,'MarkerFaceColor','k')
    axis([-1 1 -1.1 1.1])
    ylabel('$$\Phi_4+\Phi_5$$','FontSize',16,'Interpreter','latex')
    xlabel('$$x/l$$','FontSize',16,'Interpreter','latex')

subplot(2,1,2)
   phiant=(phi3-phi4);
   phiant_mass=[phiant(1) phiant(N+1) phiant(2*N+1)]
    plot(x,phi0,'-','LineWidth',3,'Color',[0.5 0.5 0.5]), hold on
    plot(xmass,phi0mass,'o','MarkerSize',12,'MarkerFaceColor',[0.5 0.5 0.5])
    plot(x,phiant,'k-','LineWidth',3)
    plot(xmass,phiant_mass,'o','MarkerSize',12,'MarkerFaceColor','k')
    axis([-1 1 -1.1 1.1])
    ylabel('$$\Phi_4-\Phi_5$$','FontSize',16,'Interpreter','latex')
    xlabel('$$x/l$$','FontSize',16,'Interpreter','latex')