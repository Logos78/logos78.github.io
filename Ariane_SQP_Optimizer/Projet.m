clear
close all

warning('off','all')
%%% Valeurs a choisir
% Criteres d'arrets de l'algorithme d'optimisation
% Norme du gradient du lagrangien / Norme de f(X(k))-f(X(k-1)) / Norme de X(k) - X(k-1) 
eps = [0.001;0.000001;0.000001];

% Nombre d'iterations maximales pour l'optimiseur
kmax = 200;

% Choix de la formule de calcul de la hessienne
  % 0 pour la formule BFGS
  % 1 pour la formule SR1
v = 0;

% Coefficient
c1 = 0.1;


%%% Valeurs initiales
mv = 2000;
Hc = 300000;

% Constantes
mu = 3.986*10^14;
Rt = 6378137;
Rc = Rt+Hc;
Vc = sqrt(mu/Rc);
Vp = Vc;
dV = 0.2*Vc;

kf = 0;
while abs(dV/Vc) > 0.01 && kf < 50
    % Ajustement de la vitesse propulsive
    Vp = Vp + dV;
    
    % Probleme d'etagement
    Eta = @(X) Etagement(X,mv,Vp);
    X = [30000;10000;2000];
    Xinf = [20000;10000;5000];
    Xsup = [60000;30000;10000];
    h = X*10^-6;
    lambda = 0;

    [me,k0,tab0] = SQP(Eta,X,Xinf,Xsup,h,lambda,c1,eps,kmax,v,10);


    % Probleme de trajectoire
    Traj = @(X) Trajectoire(X,me,mv,Hc);
    X = [1;-1;5;9.92]*pi/180;
    Xinf = ([-1;-1;-1;-1]*1000)*pi/180;     % Pas de borne
    Xsup = ([1;1;1;1]*1000)*pi/180;
    h = X*10^-6;
    lambda = [0;0];
    [theta,k1,tab1] = SQP(Traj,X,Xinf,Xsup,h,lambda,c1,eps,kmax,v,25);
    
    [R,V] = Simulateur(theta,me,mv,Hc,0);
    Vr = norm(V);

    
    %%% Calcul de l'ecart entre le vitesse reelle et la vitesse cible
    dV = Vc - Vr;
    kf = kf+1;
    
    %%% Stockage des informations
    if kf==1
        tab2 = {'iterations','Vitesse de propulsion','Vitesse reelle','Ecart','Altitude';kf,Vp,Vr,dV,norm(R)-Rt};
        cRow = 3;
    else
        tab2(cRow,:) = {kf,Vp,Vr,dV,norm(R)-Rt};
        cRow = cRow+1;
    end
end

%%% Resultat final
[R,V] = Simulateur(theta,me,mv,Hc,1);
fprintf("Vitesse propulsive initiale Vp = %.3f m/s\n\n",Vp)
fprintf("Masse d'ergol des différents étages :\n")
fprintf("me(1) = %.3f kg\nme(2) = %.3f kg\nme(3) = %.3f kg\n\n",me(1),me(2),me(3))
fprintf("Angle pour chaque étape :\n")
fprintf("theta(0) = %.3f°\ntheta(1) = %.3f°\ntheta(2) = %.3f°\ntheta(3) = %.3f°\n\n",theta(1)*180/pi,theta(2)*180/pi,theta(3)*180/pi,theta(4)*180/pi)

fprintf("Vitesse finale : %.3f m/s\n",Vr)
fprintf("Ecart avec la vitesse cible : %.3f%% \n\n",100*abs(dV)/Vc)
fprintf("Altitude finale : %.3f km\n",(norm(R)-Rt)/1000)
fprintf("Ecart avec l'altitude cible : %.3f%%\n",100*abs(norm(R)-Rc)/Rc)