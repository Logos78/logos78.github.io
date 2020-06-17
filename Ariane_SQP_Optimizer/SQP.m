function[X,k,tab] = SQP(P,X,Xinf,Xsup,h,lambda,c1,eps,kmax,v,ks_max)
    p = 1;
    kp = 0;
    s = 1;
    ks = 0;
    k = 0;
    X_old = zeros(length(X),1);
        
   
    [fX,CX,Gradf,GradC] = Gradient(P,X,h);
    fX_old = P(X_old);
    GradL = Gradf + GradC*lambda;

    H = eye(length(X));
    
    d_f = norm(fX-fX_old);
    d_X = norm(X-X_old);
    n_GradL = norm(GradL);
    
    tab = {'iteration', 'X', 'lambda', 'f(X)', 'C(X)', 'norm(GradL)', 'dX', 'df' ; k, X, lambda, fX, CX, n_GradL, d_X, d_f};
    cRow = 3;

    while n_GradL>eps(1) && k<kmax && d_f>eps(2) && d_X>eps(3)

        % Calcul du hessien
        if k>0
            H = Hessien(H,GradL,GradL_old,X,X_old,v);
        end

        % Modification du hessien
        V = eig(H);
        if min(V) <= 0
            tau = max(abs(V)) + 0.1;
            H = H + tau*eye(length(H(1,:)));
        end
        
        % Resolution du probleme quadratique
        lambda = -inv(GradC'*inv(H)*GradC)*(GradC'*inv(H)*Gradf - CX);
        d = -inv(H)*(GradC*lambda + Gradf);

        % Globalisation - Recherche lineaire
        s = 1;
        [fXsd,CXsd] = P(X+s*d);
        FXsd = Merite(fXsd,CXsd,p);
        FX = Merite(fX,CX,p);
        dFX = Gradf'*d - p*sum(abs(CX));

        if dFX >= 0
            % Reinitialisation du hessien
            H = eye(length(X));

            % Resolution du probleme quadratique
            lambda = -inv((GradC'/H)*GradC)*((GradC'/H)*Gradf - CX);
            d = -H\(GradC*lambda + Gradf);

            % Calcul de la direction de descente
            dFX = Gradf'*d - p*sum(abs(CX));

            while dFX >= 0 && kp < 10
                p = p*2; 
                kp = kp+1;

                % Calcul de la direction de descente
                dFX = Gradf'*d - p*sum(abs(CX));
            end
            kp = 0;
        end

        % Recherche lineaire
        if dFX < 0
            while (FXsd > (FX+c1*s*dFX)) && ks < ks_max
                s = s/2;
                [fXsd,CXsd] = P(X+s*d);
                FXsd = Merite(fXsd,CXsd,p);
                ks = ks+1;
            end
            ks = 0;
        end
        
        X_old = X;
        X = X + s*d;
        X = min(Xsup,X);
        X = max(Xinf,X);
        [fX,CX,Gradf,GradC] = Gradient(P,X,h);
        GradL_old = GradL;
        GradL = Gradf + GradC*lambda;
        k = k+1;
        
        n_GradL = norm(GradL);
        fX_old = P(X_old);
        d_f = norm(fX-fX_old);
        d_X = norm(X-X_old);
        tab(cRow,:)= {k, X, lambda, fX, CX, n_GradL, d_X, d_f};
        cRow = cRow+1;
    end

end
