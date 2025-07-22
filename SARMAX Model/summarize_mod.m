function varargout = summarize_mod(Mdl,Extra)
%SUMMARIZE Summarize ARIMA model estimation results
%
% Syntax:
%
%   summarize(Mdl)
%   results = summarize(Mdl)
%
% Description:
%
%   Given an estimated ARIMA model, summarize estimation results including 
%   a table of model parameter values, standard errors, t statistics, and 
%   p-values. The summary also includes various statistics such as 
%   loglikelihood and AIC/BIC.
%
%   When called without an output (first syntax), summary information is 
%   printed to the MATLAB command window. When called with an output, no 
%   information is printed, but the summary is returned in the results
%   structure.
%
%   If the model is not estimated, then it is simply displayed to the command
%   window when called without an output; otherwise the input model is 
%   returned.
%
% Input Arguments:
%
%   Mdl - ARIMA model created by the ARIMA constructor or ARIMA/ESTIMATE 
%     method.
%
% Output Arguments:
%
%   results - For estimated models, a structure of estimation summary 
%     information with the following fields:
%
%     o Description            - Model summary description
%     o SampleSize             - Effective sample size
%     o NumEstimatedParameters - Number of estimated parameters
%     o LogLikelihood          - Loglikelihood value
%     o AIC                    - Akaike information criterion
%     o BIC                    - Bayesian information criterion
%     o Table                  - Table of parameter values, standard errors, t 
%                                statistics (value divided by standard error), 
%                                and p-values (assuming normality)
%     o VarianceTable          - Table of parameter values, standard errors, t 
%                                statistics (value divided by standard error), 
%                                and p-values (assuming normality) of the
%                                ARIMA model variance.
%
%     If the model is not estimated, then the input model (Mdl) is returned.
%
% Note:
%
%  If the ARIMA model variance is constant, then VarianceTable is the Variance
%  row of Table; if the variance is a conditional variance model (e.g., GARCH)
%  then VarianceTable contains the parameter information of the conditional 
%  variance model.
%
% See also ESTIMATE.

% Copyright 2019 The MathWorks, Inc.

if numel(Mdl) > 1
   error(message('econ:arima:summarize:NonScalarModel'))
end

%
% If the model has not been estimated, then simply call the display method
% or return the input model.
%

% if isempty(Mdl.FitInformation)
%    if nargout > 0
%       varargout = {Mdl};
%    else
%       displayScalarObject(Mdl)
%    end
%    return
% end

%
% Initialize some flags.
%

isDistributionT      =  strcmpi(Mdl.Distribution.Name, 'T');   % Is it a t distribution?
isVarianceConstant   = ~any(strcmp(class(Mdl.Variance), {'garch' 'gjr' 'egarch'})); % Is it a constant variance model?
isRegressionIncluded = ~isempty(Mdl.Beta);                     % Is there a regression component?

%
% Compute the actual number of parameters included in the ARIMA model, which 
% determines number of rows in the parameter Table. If the variance is
% constant, then it is included in the ARIMA parameter count; if the variance
% is a conditional variance model, then it is NOT counted but rather included 
% the contained conditional variance model.
%

nParameters = 1 + sum(cell2mat(Extra.FitInformation.IsEstimated.AR))  + ...
                  sum(cell2mat(Extra.FitInformation.IsEstimated.SAR)) + ...
                  sum(cell2mat(Extra.FitInformation.IsEstimated.MA))  + ...
                  sum(cell2mat(Extra.FitInformation.IsEstimated.SMA)) + ...
                  size(Mdl.Beta,2) + isDistributionT + isVarianceConstant;
%
% Create the row names and column information of the summary table.
%

rowNames      = cell(nParameters,1);
Value         = zeros(nParameters,1);
StandardError = zeros(nParameters,1);
iRow          = 1;

rowNames{iRow}      = 'Constant';
Value(iRow)         = Mdl.Constant;
StandardError(iRow) = Extra.FitInformation.StandardErrors.Constant;
iRow                = iRow + 1;

Lags = 1:Extra.LagOpLHS{1}.Degree;
Lags = Lags(cell2mat(Extra.FitInformation.IsEstimated.AR));
for i = Lags            % AR terms
    rowNames{iRow}      = sprintf('AR{%d}', i);
    Value(iRow)         = Mdl.AR{i};
    StandardError(iRow) = Extra.FitInformation.StandardErrors.AR{i};
    iRow                = iRow + 1;
end

Lags = 1:Extra.LagOpLHS{2}.Degree;
Lags = Lags(cell2mat(Extra.FitInformation.IsEstimated.SAR));
for i = Lags            % SAR terms
    rowNames{iRow}      = sprintf('SAR{%d}', i);
    Value(iRow)         = Mdl.SAR{i};
    StandardError(iRow) = Extra.FitInformation.StandardErrors.SAR{i};
    iRow                = iRow + 1;
end

Lags = 1:Extra.LagOpRHS{1}.Degree;
Lags = Lags(cell2mat(Extra.FitInformation.IsEstimated.MA));
for i = Lags            % MA terms
    rowNames{iRow}      = sprintf('MA{%d}', i);
    Value(iRow)         = Mdl.MA{i};
    StandardError(iRow) = Extra.FitInformation.StandardErrors.MA{i};
    iRow                = iRow + 1;
end

Lags = 1:Extra.LagOpRHS{2}.Degree;
Lags = Lags(cell2mat(Extra.FitInformation.IsEstimated.SMA));
for i = Lags            % SMA terms
    rowNames{iRow}      = sprintf('SMA{%d}', i);
    Value(iRow)         = Mdl.SMA{i};
    StandardError(iRow) = Extra.FitInformation.StandardErrors.SMA{i};
    iRow                = iRow + 1;
end

if isRegressionIncluded
   for i = 1:numel(Mdl.Beta)
       rowNames{iRow}      = sprintf('Beta(%d)', i);
       Value(iRow)         = Mdl.Beta(i);
       StandardError(iRow) = Extra.FitInformation.StandardErrors.Beta(i);
       iRow                = iRow + 1;
   end
end

if isDistributionT     % Degrees-of-freedom
   rowNames{iRow}      = 'DoF';
   Value(iRow)         = Mdl.Distribution.DoF;
   StandardError(iRow) = Extra.FitInformation.StandardErrors.DoF;
   iRow                = iRow + 1;
end

if isVarianceConstant  % Constant Variance
   rowNames{iRow}      = 'Variance';
   Value(iRow)         = Mdl.Variance;
   StandardError(iRow) = Extra.FitInformation.StandardErrors.Variance;
else
% 
%  The ARIMA model contains a conditional variance model, so call the SUMMARIZE 
%  method of the contained conditional variance model.
%
   VarianceTable = summarize(Mdl.Variance);
end

%
% Compute ARIMA model t-statistics & p-values and create the summary table.
%

tStatistic = Value ./ StandardError;
pValue     = 2 * (normcdf(-abs(tStatistic)));
Table      = table(Value, StandardError, tStatistic, pValue, 'RowNames', rowNames);
Table.Properties.VariableNames = {'Value' 'StandardError' 'TStatistic' 'PValue'};

%
% Compute additional statistics.
%

if Extra.FitInformation.NumEstimatedParameters > 0
  [AIC,BIC] = aicbic(Extra.FitInformation.LogLikelihood,Extra.FitInformation.NumEstimatedParameters, Extra.FitInformation.SampleSize);
else
   AIC = -2 * Extra.FitInformation.LogLikelihood;
   BIC = -2 * Extra.FitInformation.LogLikelihood;
end

%
% Now print or return the summary information.
%

if nargout > 0
   output.Description            = Mdl.Description;
   output.SampleSize             = Extra.FitInformation.SampleSize;
   output.NumEstimatedParameters = Extra.FitInformation.NumEstimatedParameters;
   output.LogLikelihood          = Extra.FitInformation.LogLikelihood;
   output.AIC                    = AIC;
   output.BIC                    = BIC;
   output.Table                  = Table;
%
%  Pack the statistics associated with the variance model into a separate table.
%
   if isVarianceConstant        % Constant variance model
%
%     If the ARIMA model has a constant variance, then simply pack the variance 
%     row of the ARIMA table into it so users can rely on its existence.
%
      output.VarianceTable = Table('Variance',:);
   else                         % Conditional variance model
%
%     If the ARIMA model has a conditional variance model, then pack the parameters 
%     into a separate table as returned by the SUMMARIZE method of the contained
%     conditional variance model.
%
      output.VarianceTable = VarianceTable;
   end
   varargout = {output};
else
   disp(' ')
   fprintf("   " + '<strong>' + Mdl.Description + '</strong>\n');
   disp(' ')
   fprintf('    Effective Sample Size: %d\n', Extra.FitInformation.SampleSize)
   fprintf('    Number of Estimated Parameters: %d\n', Extra.FitInformation.NumEstimatedParameters)
   fprintf('    LogLikelihood: %g\n', Extra.FitInformation.LogLikelihood)
   fprintf('    AIC: %g\n', AIC)
   fprintf('    BIC: %g\n', BIC)
   disp(' ')
   disp(Table)
   disp(' ')
   disp(' ')
   if ~isVarianceConstant      % Conditional Variance
      fprintf("   " + '<strong>' + Mdl.Variance.Description + '</strong>\n');
      disp(' ')
      disp(VarianceTable)
      disp(' ')
   end
end

end               