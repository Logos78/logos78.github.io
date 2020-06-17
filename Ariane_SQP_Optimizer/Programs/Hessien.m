function[Hess] = hessien(H,GradL,GradL_old,X,X_old,v)
    d = X - X_old;
    y = GradL - GradL_old;
    if v==0
        if y'*d > 0
            Hess = H + (y*y')/(y'*d) - (H*d*d'*H)/(d'*H*d);
        else
            Hess = H;
        end
    end
    if v==1
        if d'*(y-H*d) ~= 0
            Hess = H + ((y-H*d)*(y-H*d)')/(d'*(y-H*d));
        else
            Hess = H;
        end
    end
end