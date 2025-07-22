function stop=outfun(x,optimValues,state)
    global hist searchdir
    stop=false;

    hist.fval=[hist.fval; optimValues.fval];
    hist.coef=[hist.coef x];
    hist.feas=[hist.feas;optimValues.constrviolation];
    hist.step=[hist.step optimValues.lssteplength];
    hist.opt=[hist.opt optimValues.firstorderopt];

    searchdir=[searchdir;optimValues.searchdirection'];
    
  end