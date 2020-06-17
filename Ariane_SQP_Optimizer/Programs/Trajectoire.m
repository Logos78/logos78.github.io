function[NV,C] = Trajectoire(theta,me,mv,Hc)
Rt = 6378137;
Rc = Hc+Rt;
mu = 3.986*10^14;
Vc = sqrt(mu/Rc);
[R,V] = Simulateur(theta,me,mv,Hc,0);

% Critère
NV = -norm(V)/Vc;

% Contraintes
C = [(norm(R) - Rc)/Rc; R'*V/(Rc*Vc)];
%C = norm(R) - Rc;
end