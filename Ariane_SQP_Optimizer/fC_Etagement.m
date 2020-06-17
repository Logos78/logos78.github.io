function [y,g] = fC_Etagement(C,om,ve,Vp)
    y = sum(ve.*log((1-C./ve)./om)) - Vp;
    g = sum(ve./(C-ve));
end