function[R,V,M] = Simulateur(theta,me,mv,Hc,graphe)
% Constantes
alpha = [15;10;10];
k = [0.1;0.15;0.2];
ve = [2600;3000;4400];
Rt = 6378137;

% Initialisation
V = 100*[cos(theta(1)); sin(theta(1))];
R = [Rt;0];
M = mv+sum((1+k).*me);
tc = [0;0];

% Graphiques
if graphe==1
    close all
    Altitude = figure('Name','Altitude','NumberTitle','off');
    Vitesse = figure('Name','Vitesse','NumberTitle','off');
    Trajet = figure('Name','Trajectoire','NumberTitle','off');

    figure(Trajet)
    hold on
    viscircles([0,0],Rt,'Color','k','LineStyle','-','LineWidth',0.5);
    viscircles([0,0],Rt+Hc,'Color','b','LineStyle',':','LineWidth',0.5);
    %axis([6*10^6 8*10^6 0 10^6]);
end


% Simulation de la trajectoire
for j = 1:3
    tc(2) = me(j)*ve(j)/(alpha(j)*M) + tc(1);
    Mi = M;
    y = [R;V;M];
    [t,y] = ode45(@(t,y) Mouvement(t,y,alpha(j),theta(j+1),ve(j),Mi), [tc(1) tc(2)], y);
    R = y(length(y),1:2)';
    V = y(length(y),3:4)';
    M = y(length(y),5) - k(j)*me(j);
    tc(1) = tc(2);
    
    if(graphe==1)
        N = zeros(length(y),1);
        for i = 1:length(y)
            N(i) = norm(y(i,1:2));
        end
        figure(Altitude)
        hold on
        plot(t,N-Rt,'.')

        for i = 1:length(y)
            N(i) = norm(y(i,3:4));
        end
        figure(Vitesse)
        hold on
        plot(t,N,'.')

        figure(Trajet)
        plot(y(:,1),y(:,2),'.')
    end
end
if(graphe==1)
    figure(Altitude)
    title('Altitude du lanceur en fonction du temps')
    xlabel('Temps (s)') 
    ylabel('Altitude (km)')
    legend('Etage 1','Etage 2','Etage 3','Location','northwest')
    hold off
    figure(Vitesse)
    title('Vitesse du lanceur en fonction du temps')
    xlabel('Temps (s)') 
    ylabel('Vitesse (m/s)')
    legend('Etage 1','Etage 2','Etage 3','Location','northwest')
    hold off
    figure(Trajet)
    title('Trajectoire du lanceur par rapport à la Terre')
    xlabel('x (km)') 
    ylabel('y (km)')
    legend('Etage 1','Etage 2','Etage 3','Location','northeast')
    hold off
end
end