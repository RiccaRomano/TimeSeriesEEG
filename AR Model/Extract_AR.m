%% Richard Foster and Cheng Ly
% The following code holds information on extracting the optimal AR model for the candidate signal (participant 9, trial 5, interval 1)
% Optimal model orders are informed by AIC (from the AIC_AR.m script)

clear
close all
clc

load AR_MA_Lags.mat % Load all significant AR and MA lag indices (informed by the PACF and ACF plots, respectively)
load AIC_AR.mat % Load AIC search results for the candidate signal AR model
load CandidateSignal.mat dataTrain dataTest % Load training data of the candidate signal

% Optimal AR model order extracted from the AIC vector, first entry = 1 AR term
AR_Order=find(AICmat==min(AICmat));
temp_ARLags=AR_Lags(1:AR_Order); % AR terms only regress against signficant lags

Mdl=arima('ARLags',temp_ARLags); % Initialize model
options=optimoptions(@fmincon,'Algorithm','sqp','MaxFunctionEvaluations',1e10,'StepTolerance',1e-10,'MaxIterations',2e3);

[EstMdl,EstParamCov,logL,info]=estimate(Mdl,dataTrain(Mdl.P+1:end),'Y0',dataTrain(1:Mdl.P),'Display','iter','Options',options);
summarize(EstMdl);

% Collect AR coefficients
aCoefs=cell2mat(EstMdl.AR);
aCoefs=[-aCoefs(end:-1:1) 1];
coefs_roots=roots(aCoefs); % Find roots of the AR polynomial

% Figure: Plots roots of the AR polynomial on the complex plane
figure;
hold on;
theta=0:0.0001:2*pi;
x=cos(theta);
y=sin(theta);
plot(real(coefs_roots),imag(coefs_roots),'.k');
plot(x,y,'-r');

save('Results_AR.mat','dataTrain','dataTest','EstMdl','EstParamCov','logL','info','coefs_roots');


