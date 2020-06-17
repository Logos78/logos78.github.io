function[fX,CX] = Arianne1(me)
% Constantes
k = [0.1101;0.1532;0.2154];
ve = [2647.2;2922.4;4344.3];
mv = 1700;
V = 11527;

% Critère
fX = sum((1+k).*me) + mv;

% Contraintes
Mf = [mv + (k(3)+1)*me(3) + (k(2)+1)*me(2) + k(1)*me(1);
      mv + (k(3)+1)*me(3) + k(2)*me(2);
      mv + k(3)*me(3)];
CX = sum(ve.*log(1+me./Mf)) - V;
end