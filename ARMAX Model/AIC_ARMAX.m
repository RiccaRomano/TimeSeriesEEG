%% Richard Foster and Cheng Ly
% The following code performs an AIC search on an ARMAX model of the candidate signal (participant 9, trial 5 , interval 1)

clear
close all
clc

load CandidateSignal.mat dataTrain stimTrain % Load the training period of the candidate signal 
load AR_MA_Lags.mat % Load all significant AR and MA lag indices (informed by the PACF and ACF plots, respectively)

maxARLags=60; % Maximum number of AR terms
maxMALags=60; % Maximum number of MA terms
AICmat=zeros(maxARLags+1,maxMALags+1); % Initialize the AIC matrix


for ii=0:maxARLags
    for jj=0:maxMALags

        if ii==0
            temp_MALags=MA_Lags(1:jj);
            Mdl=arima('MALags',temp_MALags); % Initialize model
        elseif jj==0
            temp_ARLags=AR_Lags(1:ii);
            Mdl=arima('ARLags',temp_ARLags); % Initialize model
        else
            temp_ARLags=AR_Lags(1:ii);
            temp_MALags=MA_Lags(1:jj);
            Mdl=arima('ARLags',temp_ARLags,'MALags',temp_MALags); % Initialize model
        end

        exitflag=0;
        optionset=1;
        % Set two catch loops, trying a different estimation algorithm if the first fails
        while exitflag==0                                                                                                                                                   
            if optionset==1
                try
                    options=optimoptions(@fmincon,'Algorithm','sqp','MaxFunctionEvaluations',1e10,'StepTolerance',1e-11,'MaxIterations',2e3,'ConstraintTolerance',1e-6);
                    [EstMdl,EstParamCov,logL,info,Extra]=estimate_mod(Mdl,dataTrain(Mdl.P+1:end),'Y0',dataTrain(1:Mdl.P),'X',stimTrain,'Display','full','Options',options);
                    SumMdl=summarize_mod(EstMdl,Extra);
                    AICmat(ii+1,jj+1)=SumMdl.AIC;
                    exitflag=1;
                catch
                    optionset=2;
                end
            elseif optionset==2
                try
                    options=optimoptions(@fmincon,'Algorithm','sqp-legacy','MaxFunctionEvaluations',1e10,'StepTolerance',1e-11,'MaxIterations',2e3,'ConstraintTolerance',1e-6);
                    [EstMdl,EstParamCov,logL,info,Extra]=estimate_mod(Mdl,dataTrain(Mdl.P+1:end),'Y0',dataTrain(1:Mdl.P),'X',stimTrain,'Display','full','Options',options);
                    SumMdl=summarize_mod(EstMdl,Extra);
                    AICmat(ii+1,jj+1)=SumMdl.AIC;
                    exitflag=1;
                catch
                    AICmat(ii+1,jj+1)=NaN;
                    formatSpec='Model with %4.0f AR lags could not converge, check to confirm \n';
                    fprintf(formatSpec,ii);
                    exitflag=1;
                end
            end
        end
    end
end

save('AIC_ARMAX.mat','AICmat');
