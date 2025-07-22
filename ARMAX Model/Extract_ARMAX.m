%% Richard Foster and Cheng Ly
% The following code holds information on extracting the optimal ARMAX model for the candidate signal (participant 9, trial 5, interval 1)
% Optimal model orders are informed by AIC (from the AIC_ARMAX.m script)

clear
close all
clc

global hist searchdir

load AR_MA_Lags.mat % Load all significant AR and MA lag indices (informed by the PACF and ACF plots, respectively)
load AIC_ARMAX.mat % Load AIC search results for the candidate signal ARMAX model
load CandidateSignal.mat dataTrain dataTest stimTrain stimTest % Load training data of the candidate signal

% Optimal ARMAX model order extracted from the AIC vector, first entry = 1 AR term
[AR_Order,MA_Order]=find(AICmat==min(min(AICmat)));
AR_Order=AR_Order-1;
MA_Order=MA_Order-1;
temp_ARLags=AR_Lags(1:AR_Order); % AR terms only regress against signficant lags
temp_MALags=MA_Lags(1:MA_Order);

Mdl=arima('ARLags',temp_ARLags,'MALags',temp_MALags); % Initialize model
options=optimoptions(@fmincon,'Algorithm','sqp','MaxFunctionEvaluations',1e10,'StepTolerance',1e-10,'MaxIterations',2e3,'OutputFcn',@outfun);
try
    hist.fval=[];
    hist.coef=[];
    hist.feas=[];
    hist.step=[];
    hist.opt=[];
    searchdir=[];
    [EstMdl,EstParamCov,logL,info,Extra]=estimate_mod(Mdl,dataTrain(Mdl.P+1:end),'Y0',dataTrain(1:Mdl.P),'X',stimTrain,'Display','iter','Options',options);
    summarize_mod(EstMdl,Extra);
    history=hist;
    
    % Collect AR coefficients
    aCoefs=cell2mat(EstMdl.AR);
    aCoefs=[-aCoefs(end:-1:1) 1];
    coefs_roots=roots(aCoefs); % Find roots of the AR polynomial
    
    % Figure: Plots roots of the AR polynomial on the complex plane, roots should be outside the unit circle
    figure;
    hold on;
    theta=0:0.0001:2*pi;
    x=cos(theta);
    y=sin(theta);
    plot(real(coefs_roots),imag(coefs_roots),'.k');
    plot(x,y,'-r');
catch
    %Extract last stable iteration of the optimization procedure
    last_id=find(hist.feas==0,1,'last');
    Constant=hist.coef(1,last_id);
    Var=hist.coef(end,last_id);
    AR=num2cell(hist.coef(2:AR_Order+1,last_id)');
    MA=num2cell(hist.coef(AR_Order+2:AR_Order+MA_Order+1,last_id)');
    Beta=(hist.coef(end-1,last_id)');

    nonoptMdl=arima('ARLags',AR_Lags,'MALags',MA_Lags,'AR',AR,'MA',MA,'beta',Beta,'Variance',Var,'Constant',Constant);
    [EstMdl,EstParamCov,logL,info]=estimate(nonoptMdl,dataTrain(Mdl.P+1:end),'X',exog_input,'Y0',dataTrain(1:Mdl.P),'Display','iter','Options',options);

    % Collect AR coefficients
    aCoefs=cell2mat(EstMdl.AR);
    aCoefs=[-aCoefs(end:-1:1) 1];
    coefs_roots=roots(aCoefs); % Find roots of the AR polynomial
    
    % Figure: Plots roots of the AR polynomial on the complex plane, roots should be outside the unit circle
    figure;
    hold on;
    theta=0:0.0001:2*pi;
    x=cos(theta);
    y=sin(theta);
    plot(real(coefs_roots),imag(coefs_roots),'.k');
    plot(x,y,'-r');
end
save('Results_ARMAX.mat','dataTrain','dataTest','stimTrain','stimTest','EstMdl','EstParamCov','logL','info','coefs_roots');


