function[fX,CX] = MH4WD(X)
% Critère
fX = (X(1)-1)^2 + (X(1)-X(2))^2 + (X(2)-X(3))^3 + (X(3)-X(4))^4 + (X(4)-X(5))^4;

% Contrainte
CX = [X(1)+(X(2)^2)+(X(3)^2)-3*sqrt(2)-2
         X(2)-(X(3)^2)+X(4)-2*sqrt(2)+2
         X(1)*X(5)-2];
end