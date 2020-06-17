function [dy] = Mouvement(t,y,alpha,theta,ve,Mi)  % y = [R; V; M]
R = y(1:2);
V = y(3:4);
M = y(5);

% Constantes
mu = 3.986*10^14;
cx = 0.1;
rho0 = 1.225;
H = 7000;
Rt = 6378137;

% Calcul des forces
W = (-mu*M/norm(R)^3)*R;

rho = rho0*exp(-(norm(R)-Rt)/H);
D = -cx*rho*norm(V)*V;

gamma = asin(R'*V/(norm(R)*norm(V)));
u = [cos(gamma+theta)*(-R(2))+sin(gamma+theta)*R(1);
     cos(gamma+theta)*R(1)+sin(gamma+theta)*R(2)]/norm(R);
T = alpha*Mi*u;

% Calcul de R,V et M
dy = [V; (T+W+D)/M; -norm(T)/ve];
end



















% dydt(3) = (alpha*y(5)*(cos(gamma+theta))*(-y(2))+sin(gamma+theta)*y(1))/sqrt(y(1)^2+y(2)^2) - mu*y(5)/(y(1)^2+y(2)^2)*y(1) - cx*rho * sqrt(y(3)^2 + y(4)^2))*y(3))/y(5);
% dydt(4) = (alpha*y(5)*u(2) - mu*y(5)/norm(y(1))^3)*y(2) - cx*rho * sqrt(y(3)^2 + y(4)^2)*y(4)/y(5);

% dydt(5) = -sqrt((alpha*y(5)*cos(gamma+theta)*(-y(2))+sin(gamma+theta)/sqrt(y(1)^2+y(2)^2)*y(1))^2 + (alpha*y(5)*cos(gamma+theta)*y(1)+sin(gamma+theta)*y(2))^2/sqrt(y(1)^2+y(2)^2))/ve;


%     W = (-mu*M/norm(R)^3)*R;
%     W1 = -mu*y(5)/norm(y(1))^3)*y(1);
%     W2 = -mu*y(5)/norm(y(2))^3)*y(2);
%     rho = rho0*exp(-(norm(R)-Rt)/H);
%     D = -cx*rho*norm(V)*V;
%     D1 = -cx*rho * sqrt(y(3)^2 + y(4)^2))*y(3);
%     D2 = -cx*rho * sqrt(y(3)^2 + y(4)^2))*y(4);
%      T = alpha(j)*M*u;
%      T1 = alpha*y(5)*u1
%      T2 = alpha*y(5)*u2
%      
%      sqrt((alpha*y(5)*cos(gamma+theta(j))*(-y(2))+sin(gamma+theta(j))/sqrt(y(1)^2+y(2)^2)*y(1))^2 + (alpha*y(5)*cos(gamma+theta(j))*y(1)+sin(gamma+theta(j))*y(2))^2/sqrt(y(1)^2+y(2)^2))
%      
%      
%      dydt(3) = (alpha*y(5)*(cos(gamma+theta))*(-y(2))+sin(gamma+theta)*y(1))/sqrt(y(1)^2+y(2)^2) - mu*y(5)/norm(y(1))^3)*y(1) - cx*rho * sqrt(y(3)^2 + y(4)^2))*y(3))/y(5);
%      dydt(4) = (alpha*y(5)*u(2) - mu*y(5)/norm(y(1))^3)*y(2) - cx*rho * sqrt(y(3)^2 + y(4)^2))*y(4))/y(5);
%      
%      
%      u = [cos(gamma+theta(j))*(-y(2))+sin(gamma+theta(j))*y(1);
%          cos(gamma+theta(j))*y(1)+sin(gamma+theta(j))*y(2)]/sqrt(y(1)^2+y(2)^2);