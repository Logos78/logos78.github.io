clear
close all


%%% Cas test Ariane 1
% Constantes
K = [0.1101;0.1532;0.2154];
om = K./(1+K);
ve = [2647.2;2922.4;4344.3];
Vp = 11527;
mv = 1700;

% Parametres
eps = 0.0001;
max_iter = 100;

C = 0;
fC_Etagement = @(C) fC_Etagement(C,om,ve,Vp);
[C,Gstar,iter] = Newton(fC_Etagement,C,eps,max_iter);
X = (1-C./ve)./om;

% Calcul des masses d'ergols
me(3) = (X(3)-1)*mv/(1+K(3)-K(3)*X(3));
Mi3 = mv + (1+K(3))*me(3);
me(2) = (X(2)-1)*Mi3/(1+K(2)-K(2)*X(2));
Mi2 = Mi3 + (1+K(2))*me(2);
me(1) = (X(1)-1)*Mi2/(1+K(1)-K(1)*X(1));
fprintf('Cas test Ariane 1\n');
fprintf('Masse d ergol pour le premier etage : %.3f kg\n', me(1));
fprintf('Masse d ergol pour le deuxieme etage : %.3f kg\n', me(2));
fprintf('Masse d ergol pour le troisieme etage : %.3f kg\n', me(3));

% Calcul de la masse totale du lanceur
% Mi1 = mv + (1+K(1))*me(1) + (1+K(2))*me(2) + (1+K(3))*me(3);
Mi1 = Mi2 + (1+K(1))*me(1);
fprintf('Masse totale du lanceur : %.3f kg\n\n\n', Mi1);



%%% Notre probleme
clear
close all
% Données du problème
mv = 2000;
Hc = 300000;

% Constantes
K = [0.1;0.15;0.2];
om = K./(1+K);
ve = [2600;3000;4400];
mu = 3.986*10^14;%constante gravitationnelle
Rt = 6378137;
Rc = Rt + Hc;
Vc = sqrt(mu/Rc);
Vp = Vc*1.2;

% Paramètres
eps = 0.01;
max_iter = 200;

% Résolution par l'algorithme de Newton
C = 0;
fC_Etagement = @(C) fC_Etagement(C,om,ve,Vp);
[C,Gstar,iter] = Newton(fC_Etagement,C,eps,max_iter);
X = (1-C./ve)./om;

% Calcul des masses d'ergols
me(3) = (X(3)-1)*mv/(1+K(3)-K(3)*X(3));
Mi3 = mv + (1+K(3))*me(3);
me(2) = (X(2)-1)*Mi3/(1+K(2)-K(2)*X(2));
Mi2 = Mi3 + (1+K(2))*me(2);
me(1) = (X(1)-1)*Mi2/(1+K(1)-K(1)*X(1));
fprintf("Probleme d'etagement - resultats analytiques \n");
fprintf('Masse d ergol pour le premier etage : %.3f kg\n', me(1));
fprintf('Masse d ergol pour le deuxieme etage : %.3f kg\n', me(2));
fprintf('Masse d ergol pour le troisieme etage : %.3f kg\n', me(3));

% Calcul de la masse totale du lanceur
% Mi1 = mv + (1+K(1))*me(1) + (1+K(2))*me(2) + (1+K(3))*me(3);
Mi1 = Mi2 + (1+K(1))*me(1);
fprintf('Masse totale du lanceur : %.3f kg\n\n', Mi1);

Etagement = @(X) Etagement(X,mv,Vp);
h = me*10^-4;
[fme,Cme,Gradf,GradC] = Gradient(Etagement,me',h);
lambda = - fme/C;
fprintf('Le multiplicateur vaut : %f\n',lambda);
fprintf('La norme gradient du lagrangien vaut : %f\n', norm(Gradf + GradC*lambda));
fprintf('La contrainte est : %f\n\n\n', Cme);


% Resolution par l'algorithme SQP
fprintf('Comparaison des masse d ergol donnees par SQP\n')
X = [30000;10000;2000];
[X,k_sqp,tab] = SQP(Etagement,X,[20000;5000;1000],[50000;25000;10000],X*10^-4,0,0.1,[0.01;0.0001;0.0001],200,0,10);
fprintf('Masse d ergol pour le premier etage : %.3f kg\n', X(1));
fprintf('Masse d ergol pour le deuxieme etage : %.3f kg\n', X(2));
fprintf('Masse d ergol pour le troisieme etage : %.3f kg\n', X(3));
