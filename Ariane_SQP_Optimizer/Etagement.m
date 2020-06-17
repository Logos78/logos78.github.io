function[fX,CX] = Etagement(me,mv,V)
    % Constantes
    k = [0.1;0.15;0.2];
    ve = [2600;3000;4400];
    
    % Critère
    fX = sum((1+k).*me) + mv;

    % Contraintes
    Mf = [mv + (k(3)+1)*me(3) + (k(2)+1)*me(2) + k(1)*me(1);
          mv + (k(3)+1)*me(3) + k(2)*me(2);
          mv + k(3)*me(3)];
    CX = sum(ve.*log(1+me./Mf)) - V;
end