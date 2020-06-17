function[fX,CX,Gradf,GradC] = Gradient(Probleme,X,h)   % X en colonne
    n = length(X);
    [fX,CX] = Probleme(X);
    for i=1:n
        H = zeros(n,1);
        H(i) = h(i);
        [fH_val,CH_val] = Probleme(X+H);
        Gradf(i) = (fH_val - fX)/h(i);
        GradC(i,:) = (CH_val - CX)/h(i);
    end
    Gradf = Gradf';
    %GradC = GradC;
end
