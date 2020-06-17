clear
close all

warning('off','all')
%%% Valeurs à choisir
% Critères d'arrêts de l'algorithme d'optimisation 
eps = [0.001;0.0000001;0.0000001];
%eps = [0.01;0;0]; % A essayer si GradL final est trop grand et k<kmax

% Nombre d'itérations maximales pour l'optimiseur
kmax = 200;

% Choix de la formule de calcul de la hessienne
  % 0 pour la formule BFGS
  % 1 pour la formule SR1
v = 1;

% Coefficient
c1 = 0.1;

% Cas test
  % 0 pour l'optimisation des thétas
  % 1 pour MH4WD
  % 2 pour Arianne 1
test = 0;


%%% Optimisation des thétas
if test==0
    me = [34353;16562;7300];
    mv = 2000;
    Hc = 300000;
    
    % Calcul des angles thetas
    Trajectoire = @(X) Trajectoire(X,me,mv,Hc);
    X = [1;-1;5;10]*pi/180;
    Xinf = ([-1;-1;-1;-1]*1000)*pi/180;     % Pas de borne
    Xsup = ([1;1;1;1]*1000)*pi/180;
    h = X*10^-6;
    lambda = [0;0];
    
    eps = [0.1;0;0];
    [X,k,tab] = SQP(Trajectoire,X,Xinf,Xsup,h,lambda,c1,eps,kmax,v,10);
    
    %%Calcul de la vitesse et de l'altitude
    mu = 3.986*10^14;
    Rt = 6378137;
    Rc = Hc+Rt;
    Vc = sqrt(mu/Rc);
    
    [R,V,M] = Simulateur(X,me,mv,Hc,1);
    Vr = norm(V);
    fprintf("Ecart avec la vitesse cible :  %.3f m/s\n",Vc-Vr)
    fprintf("Ecart avec l'altitude cible : %.3f km\n",(norm(R)-Rc)/1000)

    fprintf("\nSolution de l'optimisation des thetas :")
    X = X*180/pi
end
    

%%% Cas test MH4WD
if test==1
    X = [-1;2;1;-2;-2];
    Xinf = [-1;2;1;0;-2]-1;
    Xsup = [-1;2;1;0;-2]+1;
    %Xinf = [-1.3;2.3;1.1;-0.3;-1.7];
    %Xsup = [-1.1;2.5;1.3;-0.1;-1.5];
    h = X*10^-6;
    lambda = [0;0;0];

    [X,k,tab] = SQP(@MH4WD,X,Xinf,Xsup,h,lambda,c1,eps,kmax,v,10);
    fprintf("\nSolution attendue : ")
    Xf = [-1.2366; 2.4616; 1.1911; -0.2143; -1.6165]
    fprintf("\nSolution du cas test MH4WD :")
    X
end


%%% Cas test Ariane 1
if test==2
    X = [180000;50000;15000];
    Xinf = [100000;10000;5000];
    Xsup = [200000;50000;10000];
    h = X*10^-6;
    lambda = 0;

    [X,k,tab] = SQP(@Ariane1,X,Xinf,Xsup,h,lambda,c1,eps,kmax,v,0);
    fprintf("\nSolution attendue : ")
    Xf = [145349; 31315; 7900]
    fprintf("\nSolution du cas test Ariane 1 :")
    X
end

fprintf("\nNombre d'itérations = %.0d\n\n",k)


