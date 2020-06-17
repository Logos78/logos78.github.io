function [Xstar,Gstar,iter]=Newton(G,X0,eps,max_iter)
    X = X0;
    [Y,J] = G(X0);
    iter = 0;
    alpha = 0.4;
    while (abs(Y) >= eps) & (iter <= max_iter)
        [Y,J] = G(X);
        X = X - alpha*Y/J;
        iter = iter+1;
    end
    Xstar = X;
    Gstar = J;
end