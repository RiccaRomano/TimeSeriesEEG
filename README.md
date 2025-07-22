# Time-series models can forecast long periods of human temporal EEG responses to randomly alternating visual stimuli <br />

## Richard R. Foster $^1$ , Connor Delaney $^2$ , Dean J. Krusienski $^2$ , Cheng Ly $^{1,3}$ <br />
### $^1$ _Department of Mathematics and Applied Mathematics, Virginia Commonwealth University, Richmond VA 23284_ <br />
### $^2$ _Department of Biomedical Engineering, Virginia Commonwealth University, Richmond VA 23284 <br />_
### $^3$ _Department of Statistics, Virginia Commonwealth University, Richmond VA 23284 <br />_

## Abstract <br />

Visual stimuli changing at constant temporal frequencies induces sharp peaks in the power spectrum of the electroencephalogram (EEG) over the visual cortex at the driving frequency and its harmonics, known as steady-state visual evoked potentials (SSVEPs). Visual stimuli that alternate according to randomized temporal patterns can also result in predictable EEG patterns. While such EEG responses are robust and predictable, the underlying biophysical mechanisms that shape these responses are not fully understood. To better understand the relationship between the stimuli and associated EEG responses, and ultimately inform a biophysical model, we examine these relationships using EEG data from a controlled experiment. We model the EEG using several statistical time series models with components that loosely mimic biophysical mechanics: an autoregressive (AR) model, with an exogenous input (ARX), adding moving average terms (ARMAX), and a seasonality term (SARMAX). We fit these models using the Box-Jenkins methodology and assess EEG forecast performance for a relatively long period of several seconds out-of-sample. We find in-sample fits are good in all models despite the complexities of the visual pathway, and that all models can capture aspects of out-of-sample EEG, including the distribution of values (point-wise in time), the point-wise Pearson's correlation of EEG and model, and the frequency content. Surprisingly, we find little variation in the performance of all models, with the most sophisticated and detailed model (SARMAX) performing comparatively poorly in some instances. Taken together, our results suggest the simplest AR model is valuable because it is easy to understand and can perform just as well as more complicated models. Since these models are relatively simple and more transparent than contemporary predictive models with numerous parameters, our study may provide insights into the biological mechanisms of the temporal dynamics of human EEG response that could generalize to other visual stimuli.

## Description

This work implements a suite of autoregressive time-series models to describe and predict EEG responses to randomly alternating visual stimuli. 

## Code Architecture

Preliminary data analysis of one candidate signal is contained in the

## Dependencies

Program prerequisites include MATLAB version 2024b (or newer)

Richard Foster: fosterrr@vcu.edu
Cheng Ly: cly@vcu.edu (corresponding author)


