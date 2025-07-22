%% Richard Foster and Cheng Ly
% The following code simulates the in-sample SARMAX model fit against the candidate signal
% It also simulates individual forecasts and measures foreacsting accuracy according to metrics listed in the associated manuscript 

clear
close all
clc

load Results_SARMAX.mat % Load optimal SARMAX model of the candidate signal
Fs=256; % Frames per second
tspan=(EstMdl.P+1:length(dataTrain))/Fs; % Time span of the in-sample period
residual=infer(EstMdl,dataTrain(EstMdl.P+1:end),'Y0',dataTrain(1:EstMdl.P),'X',stimTrain); % Estimate the in-sample residuals
in_sample=dataTrain(EstMdl.P+1:end)+residual; % Find the in-sample fit

% In-sample fit figure
f1=figure;
hold on;
plot((1:length(dataTrain))/Fs,dataTrain,'-k');
plot(tspan,in_sample,'--r');
set(gca,'FontSize',12);
xlabel('Time (s)','FontSize',14,'FontWeight','bold');
ylabel('EEG Value (\muV)','FontSize',14,'FontWeight','bold');
legend('EEG Data','In-Sample Fit');

% Simulate individual SARMAX model forecasts
numobs=length(dataTest);
numpaths=1000;
[ysim,~]=simulate_mod_hist(EstMdl,numobs,'Y0',dataTrain,'X0',stimTrain,'XF',stimTest,'NumPaths',numpaths);

%% Forecasting Metrics
% Power spectrum estimate
Fs=256;
freqrange=0:0.1:60;
[pxx,fx]=pwelch(dataTest-mean(dataTest),[],[],freqrange,Fs);
[pyy,fy]=pwelch(ysim-mean(ysim,2),[],[],freqrange,Fs);  %same as ysim-repmat(mean(ysim),896,1)
mean_pyy=mean(pyy,2);
power_err=mean((mean_pyy-pxx').^2);

f2=figure;
hold on;
plot(fx,pxx,'-k');
plot(fy,mean_pyy,'-r');
set(gca,'FontSize',12);
ylabel('Power spectrum (\muV)^2s','FontSize',14,'FontWeight','bold');
xlabel('Frequency (Hz)','FontSize',14,'FontWeight','bold');
legend('EEG Data','Out-of-Sample Model');

% Histogram Metric
edges=-10:0.2:10;
save_counts=zeros(numpaths,length(edges)-1);
for pp=1:numpaths
    [N,edges]=histcounts(ysim(:,pp),edges);
    save_counts(pp,:)=N;
end
avg_hist=mean(save_counts);
data_hist=histcounts(dataTest,edges);
hist_err=mean((data_hist-avg_hist).^2);

f3=figure;
hold on;
histogram(dataTest,edges,'Facecolor','k');
histogram('BinCounts', avg_hist, 'BinEdges', edges,'Edgecolor','k','Edgecolor','r','DisplayName','Average Histogram Forecast');
set(gca,'FontSize',12);
xlabel('EEG Value (\muV)','FontSize',14,'FontWeight','bold');
ylabel('Counts');
legend('Out-of-sample Data','Average Model Histogram');

% Pearson's Correlation
corr=corrcoef(mean(ysim,2),dataTest);
pear_corr=corr(1,2);

save('Forecast_SARMAX.mat','power_err','hist_err','pear_corr');

saveas(f1,'InSampleFit_SARMAX.fig');
saveas(f1,'InSampleFit_SARMAX.svg');
saveas(f1,'InSampleFit_SARMAX.eps');

saveas(f2,'PowerSpec_SARMAX.fig');
saveas(f2,'PowerSpec_SARMAX.svg');
saveas(f2,'PowerSpec_SARMAX.eps');

saveas(f3,'Hist_SARMAX.fig');
saveas(f3,'Hist_SARMAX.svg');
saveas(f3,'Hist_SARMAX.eps');

