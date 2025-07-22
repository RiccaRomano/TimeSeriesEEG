function [Y,E,V] = simulate_mod(Mdl, numObs, varargin)
%SIMULATE Simulate ARIMA model responses and conditional variances
%
% Syntax:
%
%   [Y,E,V] = simulate(Mdl,numObs)
%   [Y,E,V] = simulate(Mdl,numObs,param1,val1,param2,val2,...)
%
% Description:
%
%   Simulate sample paths of responses, innovations, and conditional 
%   variances of a univariate ARIMA process.
%
% Input Arguments:
%
%   Mdl - ARIMA model specification object, as produced by the ARIMA 
%     constructor or ARIMA/ESTIMATE method.
%
%   numObs - Positive integer indicating the number of observations (rows)
%     generated for each path of the outputs Y, E, and V.
%
% Optional Input Parameter Name/Value Pairs:
%
%   'NumPaths'   Positive integer indicating the number of sample paths 
%                (columns) generated for all simulated outputs. The default 
%                is 1.
%
%   'Y0'         Presample response data, providing initial values for the 
%                model. Y0 is a column vector or a matrix. If Y0 is a 
%                column vector, then it is applied to each simulated path. 
%                If Y0 is a matrix, then it must have at least NumPaths 
%                columns. Y0 may have any number of rows, provided at least 
%                Mdl.P observations exist to initialize the model. If the
%                number of rows exceeds Mdl.P, then only the most recent 
%                Mdl.P observations are used. If the number of columns 
%                exceeds NumPaths, then only the first NumPaths columns are 
%                used. If Y0 is unspecified, any necessary presample 
%                observations are set to the unconditional mean for stationary 
%                AR processes, and to zero if the process is non-stationary 
%                or contains a regression component. The last row contains 
%                the most recent observation.
%
%   'E0'         Mean-zero presample innovations, providing initial values 
%                for the model. E0 is a column vector or a matrix. If E0 is 
%                a column vector, then it is applied to each simulated path. 
%                If E0 is a matrix, then it must have at least NumPaths
%                columns. E0 may have any number of rows, provided sufficient 
%                observations exist to initialize the ARIMA model as well 
%                as any conditional variance model (the number of observations
%                required is at least Mdl.Q, but may be more if a conditional 
%                variance model is included). If the number of rows 
%                exceeds the number necessary, then only the most recent 
%                observations are used. If the number of columns exceeds 
%                NumPaths, then only the first NumPaths columns are used. If
%                no presample data is specified, any necessary observations 
%                are set to zero. The last row contains the most recent 
%                observation.
%
%   'V0'         Positive presample conditional variances, providing initial
%                values for any conditional variance model; if the variance 
%                of the model is constant, then V0 is unnecessary. V0 is a 
%                column vector or a matrix. If V0 is a column vector, then 
%                it is applied to each simulated path. If V0 is a matrix, 
%                then it must have at least NumPaths columns. V0 may have 
%                any number of rows, provided sufficient observations exist 
%                to initialize any conditional variance model. If the number 
%                of rows exceeds the number necessary, then only the most 
%                recent observations are used. If the number of columns 
%                exceeds NumPaths, then only the first NumPaths columns are 
%                used. If no presample variance data is specified, any 
%                necessary observations are set to the unconditional variance 
%                of the conditional variance process. The last row contains 
%                the most recent observation.
%
%   'X'          Matrix of predictor data used to include a regression 
%                component in the conditional mean. Each column of X is a
%                separate time series, and the last row of each contains
%                the most recent observation of each series. The number of
%                observations in X must equal or exceed numObs. When the 
%                number of observations in X exceeds numObs, only the most 
%                recent observations are used. If missing, the conditional 
%                mean will have no regression component regardless of the 
%                presence of any regression coefficients found in the model.
%
% Output Arguments:
%
%   Y - numObs-by-NumPaths matrix of simulated response data.
%
%   E - numObs-by-NumPaths matrix of simulated mean-zero innovations. 
%
%   V - numObs-by-NumPaths matrix of conditional variances of the 
%     innovations in E.
%
% Notes:
%
%   o Missing data values, indicated by NaNs, are removed from X by listwise 
%     deletion (i.e., any row in X with at least one NaN is removed), reducing
%     the effective sample size. The presample data Y0, E0, and V0 are merged
%     into a composite series, and any row of the combined series with at 
%     least one NaN is also removed by listwise deletion. The presample data 
%     is also synchronized such that the last (most recent) observation of 
%     each series occurs at the same time.
%
%  o  Regression models included in the conditional mean are based on the
%     presence of the predictor matrix X. Although each column of the output 
%     time series represents a different path of the corresponding univariate 
%     stochastic process, the regression matrix X represents as a single 
%     path of a (possibly) multivariate time series matrix in which each 
%     column is a different time series. When the conditional mean has a 
%     regression component, the entire predictor matrix X is applied to 
%     every column of the output time series. 
%
% References:
%
%   [1] Box, G. E. P., G. M. Jenkins, and G. C. Reinsel. Time Series
%       Analysis: Forecasting and Control. 3rd edition. Upper Saddle River,
%       NJ: Prentice-Hall, 1994.
%
%   [2] Enders, W. Applied Econometric Time Series. Hoboken, NJ: John Wiley
%       & Sons, 1995.
%
%   [3] Hamilton, J. D. Time Series Analysis. Princeton, NJ: Princeton
%       University Press, 1994.
%
% See also ARIMA, FORECAST, ESTIMATE, INFER, FILTER.

% Copyright 2018 The MathWorks, Inc.

if numel(Mdl) > 1
   error(message('econ:arima:simulate:NonScalarModel'))
end

%
% Check input parameters and set defaults.
%

if nargin < 2
   error(message('econ:arima:simulate:NonEnoughInputs'))
end

parser = inputParser;
parser.addRequired ('requiredNumObs',    @(x) validateattributes(x, {'double'}, {'scalar' 'integer' '>' 0}, '', 'number of observations'));
parser.addParameter('numPaths'      , 1, @(x) validateattributes(x, {'double'}, {'scalar' 'integer' '>' 0}, '', 'number of paths'));
parser.addParameter('Y0'            , 0, @(x) validateattributes(x, {'double'}, {}, '', 'presample responses'));
parser.addParameter('E0'            , 0, @(x) validateattributes(x, {'double'}, {}, '', 'presample residuals'));
parser.addParameter('V0'            , 0, @(x) validateattributes(x, {'double'}, {}, '', 'presample variances'));
parser.addParameter('XF'             ,[], @(x) validateattributes(x, {'double'}, {}, '', 'regression matrix'));
parser.addParameter('X0',[],@(x) validateattributes(x, {'double'}, {}, '', 'presample regression matrix'))
try
  parser.parse(numObs, varargin{:});
catch exception
  exception.throwAsCaller();
end

numObs   = parser.Results.requiredNumObs;
numPaths = parser.Results.numPaths;
Y0       = parser.Results.Y0;
E0       = parser.Results.E0;
V0       = parser.Results.V0;
X0       = parser.Results.X0;
XF        = parser.Results.XF;

%
% Get model parameters and extract lags associated with non-zero coefficients.
%
% In the code segment below, AR and MA represent the compound auto-regressive 
% and moving average polynomials, respectively, including the effects of 
% integration and seasonality.
%

constant = Mdl.Constant;       % Additive constant
variance = Mdl.Variance;       % Conditional variance

if any(strcmp(class(variance), {'garch' 'gjr' 'egarch'}))  % Allow for a conditional variance model
   P = max(Mdl.Variance.P, Mdl.P);
   Q = max(Mdl.Variance.Q, Mdl.Q);                  % Total number of lagged e(t) needed
else
   P = Mdl.P;                         % Total number of lagged y(t) needed
   Q = Mdl.Q;                         % Total number of lagged e(t) needed
end

AR     = Mdl.getLagOp('Compound AR');
MA     = Mdl.getLagOp('Compound MA');
LagsAR = AR.Lags;                     % Lags of non-zero AR coefficients
LagsAR = LagsAR(LagsAR > 0);
LagsMA = MA.Lags;                     % Lags of non-zero MA coefficients

isARstable = isStable(AR);            % Determine if the process is AR stable

if isempty(LagsAR)
   AR =  [];
else
   AR = AR.Coefficients;              % Lag Indexed Array
   AR = -[AR{LagsAR}];                % Non-zero AR coefficients (vector)
end

MA = MA.Coefficients;                 % Lag Indexed Array
MA = [MA{LagsMA}];                    % Non-zero MA coefficients (vector)

%
% Ensure coefficients are specified.
%

if any(isnan(constant))
   error(message('econ:arima:simulate:UnspecifiedConstant'))
end

if any(isnan(AR))
   error(message('econ:arima:simulate:UnspecifiedAR'))
end

if any(isnan(MA))
   error(message('econ:arima:simulate:UnspecifiedMA'))
end

isRegressionIncluded = ~any(strcmpi('XF', parser.UsingDefaults));

if isRegressionIncluded
   beta = Mdl.Beta;
   if isempty(beta) || any(isnan(beta))
      error(message('econ:arima:simulate:UnspecifiedBeta'))
   end
   if numel(beta) ~= size(XF,2)
      error(message('econ:arima:simulate:InconsistentRegression'))
   end
end

if ~any(strcmp(class(variance), {'garch' 'gjr' 'egarch'})) && any(isnan(variance))
   error(message('econ:arima:simulate:UnspecifiedVariance'))
end

%
% Remove missing observations (NaN's) via listwise deletion.
%

if any(isnan(Y0(:))) || any(isnan(E0(:))) || any(isnan(V0(:)))
   [Y0, E0, V0] = internal.econ.LagIndexableTimeSeries.listwiseDelete(Y0, E0, V0);
end

if any(isnan(XF(:)))
   XF = internal.econ.LagIndexableTimeSeries.listwiseDelete(XF);
end

%
% Compute the total number of observations generated for each path as the
% sum of the number of observations simulated and the number of presample 
% observations needed to initialize the recursion.
%

maxPQ = max([P Q]);     % Maximum presample lags needed   
T     = numObs + maxPQ; % Total number of observations generated

%
% Check any user-specified presample observations used for conditioning, or 
% generate any required observations automatically.
%

isY0specified = ~any(strcmpi('Y0', parser.UsingDefaults));
isE0specified = ~any(strcmpi('E0', parser.UsingDefaults));
isX0specified = ~any(strcmpi('X0', parser.UsingDefaults));
if isE0specified      % Did the user specify presample e(t) observations?

%
%  Check user-specified presample data for the residuals e(t). 
%
%  Notice that the following line of code saves the original E0 input to 
%  forecast the conditional variance model later (rather than overwriting
%  it with a stripped version of itself).
%

   e0 = internal.econ.LagIndexableTimeSeries.checkPresampleData(zeros(maxPQ,numPaths), 'E0', E0, Q);

%
%  Prepend the residuals with any user-specified presample observations and
%  transpose for efficiency.
%

   E            = zeros(numPaths,T);
   E(:,1:maxPQ) = e0';

else

%
%  The user did not specify presample e(t) observations. 
%

   if isY0specified && ( (size(Y0,1) >= (P + Q)) && ~isempty(Y0) )

      isE0Inferred = true;           % Initialize to true

      if isX0specified               % ARMAX model
%
%        Since the model has a regression component, whether or not sufficient 
%        observations of the input series y(t) required to infer initial values
%        of the residuals e(t) exist depends upon whether sufficient observations 
%        of the predictor data exist.
%
         if size(X0,1) >= (size(Y0,1) + Mdl.P)
%
%           When the number of presample observations of X0 equals or exceeds
%           the number of observations in Y0 plus Mdl.P, the INFER method is
%           called without specifying optional presample responses (see the 
%           Y0 input to INFER). When called in this manner, INFER automatically 
%           generates any required initial responses by backcasting, which
%           is consistent with the behavior in the absence of a regression
%           component (see just below).
%
            residuals = infer(Mdl, Y0, 'X', X0);  % Use backcasting

         else

            residuals = zeros(size(Y0,1),numPaths);          % Pre-allocate to correct size
%
%           When the number of presample observations of X0 is less than the
%           number of observations in Y0 plus Mdl.P, the INFER method must 
%           be called by specifying optional presample responses (see the
%           Y0 input to INFER). To do this, the optional presample responses 
%           passed to INFER are obtained by stripping the first Mdl.P 
%           observations of the presample historical input series (the Y0 
%           input to this FORECAST method), thereby reducing the effective 
%           sample size. 
%
%           When presample stripping occurs, we must again test to ensure
%           that sufficient observations remain.
%
            if size(Y0((Mdl.P + 1):end,:),1) >= (P + Q)
%
%              Sufficient presample observations of y(t) exist to infer
%              presample innovations e(t), so initialize the first few
%              observations to the unconditional standard deviation.
%
               if any(strcmp(class(variance), {'garch' 'gjr' 'egarch'}))
                  residuals(1:Mdl.P,:) = repmat(sqrt(Mdl.Variance.UnconditionalVariance), Mdl.P, numPaths);
               else
                  residuals(1:Mdl.P,:) = repmat(sqrt(Mdl.Variance), Mdl.P, numPaths);
               end

               temp_residuals = infer(Mdl, Y0((Mdl.P + 1):end,:), 'Y0', Y0(1:Mdl.P,:), 'X', X0);
               residuals((Mdl.P + 1):end,:)=repmat(temp_residuals,1,numPaths);
            else
               isE0Inferred = false;                % Insufficient observations
            end

         end

      else            % ARIMA model only
%
%        Since the model has no regression component, sufficient observations 
%        of the input series y(t) have been specified and so initial values 
%        of the innovations e(t) may be inferred.
%
         residuals = infer(Mdl, Y0);

      end

      E            = zeros(numPaths,T);
      E(:,1:maxPQ) = residuals((end - maxPQ + 1):end,:)';

   else
%
%    Insufficient observations of the input series y(t) have been specified,
%    so initialize any required presample observations with the unconditional 
%    mean of zero.

     E            = zeros(numPaths,T);   % Unconditional mean of e(t)
     isE0Inferred = false;

   end

end




% Did the user specify presample y(t) observations?
if ~any(strcmpi('Y0', parser.UsingDefaults))  
%
%  Check user-specified presample data for the responses y(t).
%
   Y0 = internal.econ.LagIndexableTimeSeries.checkPresampleData(ones(maxPQ,numPaths), 'Y0', Y0, Mdl.P);
%
%  Size the responses y(t) and initialize with specified data.
%
   Y            = zeros(numPaths,T);
   Y(:,1:maxPQ) = Y0';
else
%
%  The user did not specify presample y(t) observations. 
%
   if isARstable && (sum(AR) ~= 1) && ~isRegressionIncluded
%
%     The model is AR-stable and without a regression component, so compute 
%     the unconditional (i.e., long-run) mean of the y(t) process directly 
%     from the parameters of the model and use it to initialize any required 
%     presample observations.
%
      average = constant / (1 - sum(AR));
      Y       = repmat([average(ones(1,maxPQ)) zeros(1,numObs)], numPaths, 1);

   else
%
%     The model is not AR-stable, and so a long-run mean of the y(t) process 
%     cannot be calculated from the model. The following simply assumes zeros 
%     for any required presample observations for y(t).
%
      Y  = zeros(numPaths,T);

   end

end




%
% Check any user-specified regression data for sufficient observations.
%
if isRegressionIncluded
   try
     XF = internal.econ.LagIndexableTimeSeries.checkPresampleData(zeros(T,size(XF,2)), 'X', XF, numObs);
     XF = XF.';
   catch exception
     error(message('econ:arima:simulate:InsufficientXRows'))
   end
end





%
% Generate the innovations e(t).
%

if any(strcmp(class(variance), {'garch' 'gjr' 'egarch'}))    % Is there a conditional variance model?

   isV0specified = ~any(strcmpi('V0', parser.UsingDefaults));

   if isE0specified && isV0specified 
      [V,e] = simulate(variance, numObs, 'numPaths', numPaths, 'E0', E0, 'V0', V0);
   elseif isE0specified
      [V,e] = simulate(variance, numObs, 'numPaths', numPaths, 'E0', E0);
   elseif isV0specified
      [V,e] = simulate(variance, numObs, 'numPaths', numPaths, 'V0', V0);
   else
      [V,e] = simulate(variance, numObs, 'numPaths', numPaths);
   end

   E(:,(maxPQ + 1):end) = e';
   
else                             % Then it's a constant-variance model

%
%  Format the standardized disturbance processes z(t). These processes drive
%  the model innovations e(t) = sqrt(v(t)) * z(t).
%

   Z                    = internal.econ.simulateStandardizedVariates(Mdl, numPaths, numObs);
   E(:,(maxPQ + 1:end)) = Z * sqrt(variance);

   if nargout > 2
      V = variance(ones(numObs, numPaths));
   end

end






%
% Simulate the data y(t).
%

coefficients = [constant  AR  MA]';  % ARIMA coefficient vector
I            = ones(numPaths,1);

if isRegressionIncluded              % ARIMAX model

   for t = (maxPQ + 1):T
       data   = [I  Y(:,t - LagsAR)  E(:,t - LagsMA)];
       Y(:,t) = data * coefficients  +  beta * XF(:,t);
   end

else                                 % ARIMA model (no regression component)

   for t = (maxPQ + 1):T
       data   = [I  Y(:,t - LagsAR)  E(:,t - LagsMA)];
       Y(:,t) = data * coefficients;
   end

end

%
% Since max(P,Q) observations have been prepended to the output processes 
% to compensate for presample effects, strip the start-up, retain only the 
% last numObs observations, and transpose to a conventional time series
% format.
%

Y = Y(:,(maxPQ + 1):T)';
E = E(:,(maxPQ + 1):T)';

end