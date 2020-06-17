%Geoffrey Pouliquen
%Apolline El Baz

function[y] = Merite(fX,CX,p)
    y = fX + p*sum(abs(CX));
end