%% Richard Foster and Cheng Ly
% The following code holds information on extracting the optimal ARX model for the candidate signal (participant 9, trial 5, interval 1)
% Optimal model orders are informed by AIC (from the AIC_ARX.m script)

clear
close all
clc

load AR_MA_Lags.mat % Load all significant AR and MA lag indices (informed by the PACF and ACF plots, respectively)
load AIC_ARX.mat % Load AIC search results for the candidate signal ARX model
load CandidateSignal.mat dataTrain dataTest stimTrain stimTest % Load training data of the candidate signal

% Optimal AR model order extracted from the AIC vector, first entry = 1 AR term
AR_Order=find(AICmat==min(AICmat));
temp_ARLags=AR_Lags(1:AR_Order); % AR terms only regress against signficant lags

Mdl=arima('ARLags',temp_ARLags); % Initialize model
options=optimoptions(@fmincon,'Algorithm','sqp','MaxFunctionEvaluations',1e10,'StepTolerance',1e-10,'MaxIterations',2e3);

[EstMdl,EstParamCov,logL,info]=estimate(Mdl,dataTrain(Mdl.P+1:end),'Y0',dataTrain(1:Mdl.P),'X',stimTrain,'Display','iter','Options',options);
summarize(EstMdl);

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

save('Results_ARX.mat','dataTrain','dataTest','stimTrain','stimTest','EstMdl','EstParamCov','logL','info','coefs_roots');


