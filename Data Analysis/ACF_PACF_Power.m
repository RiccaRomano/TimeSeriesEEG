%% Richard Foster and Cheng Ly
% The following code takes the autocorrelation, partial autocorrelation, power spectrum estimate of the candidate signal

clear
close all
clc

load CandidateSignal.mat fulldata % Load the full trial length of the candidate signal

% Autocorrelation of the full candidate signal
[acf,acflags,acfbounds]=autocorr(fulldata,"NumLags",1000);
ACFCell=[acf acflags];
ACFBounds=acfbounds;

% Partial autocorrelation of the full candidate signal
[pacf,pacflags,pacfbounds]=parcorr(fulldata,'NumLags',1000);
PACFCell=[pacf pacflags];
PACFBounds=pacfbounds;

% Partial autocorrelation results inform on signficant AR lags
AR_Lags=(1:length(PACFCell(:,1))).*(abs(PACFCell(:,1)) > PACFBounds(1))';
AR_Lags(AR_Lags==0)=[];

% Autocorrelation results inform on signficant MA lags
MA_Lags=(1:length(ACFCell(:,1))).*(abs(ACFCell(:,1)) > ACFBounds(1))';
MA_Lags(MA_Lags==0)=[];

% Save information
save('AR_MA_Lags.mat');


%% Figures
clear
close all
clc

load AR_MA_Lags.mat

% Autocorrelation function figure
Fs=256;
tspan=acflags/Fs;
f1=figure;
plot(tspan,acf,'Color',[0.5 0.5 0.5],'LineStyle','-','LineWidth',2);
set(gca,'FontSize',12)
xlabel('Time Lag (s)','FontSize',14,'FontWeight','bold');
ylabel('Autocorrelation Function','FontSize',14,'FontWeight','bold');

% Partial autocorrelation function figure
f2=figure;
plot(tspan,pacf,'Color',[0.5 0.5 0.5],'LineStyle','-','LineWidth',2);
set(gca,'FontSize',12);
xlabel('Time Lag (s)','FontSize',14,'FontWeight','bold');
ylabel('Partial Autocorrelation Function','FontSize',14,'FontWeight','bold');

% Power spectrum estimate figure
freqrange=0:0.1:60;
[pxx,fx]=pwelch(fulldata-mean(fulldata),[],[],freqrange,Fs);

f3=figure;
plot(fx,pxx,'Color',[0.5 0.5 0.5],'LineStyle','-','LineWidth',2);
set(gca,'FontSize',12);
xlabel('Frequency (Hz)','FontSize',14,'FontWeight','bold');
ylabel('Power Spectrum (\mu V)^2s','FontSize',14,'FontWeight','bold');
xlim([0 50]);

saveas(f1,'ACFplot.fig');
saveas(f1,'ACFplot.eps');
saveas(f1,'ACFplot.svg');

saveas(f2,'PACFplot.fig');
saveas(f2,'PACFplot.eps');
saveas(f2,'PACFplot.svg');

saveas(f3,'pWelchplot.fig');
saveas(f3,'pWelchplot.eps');
saveas(f3,'pWelchplot.svg');
