clear all
close all
clc

addpath(genpath('/home/dcardenasp/Documents/MATLAB/spm8/'))
addpath(genpath('/home/dcardenasp/Documents/MATLAB/SliceBrowser/'))

HP=spm_vol('/media/dcardenasp/VERBATIM/DataBases/brainweb/phantom.nii');
HX=spm_vol('/media/dcardenasp/VERBATIM/DataBases/brainweb/t1_pn3_rf40/t1_pn3_rf40.nii');

X = spm_read_vols(HX);
L = spm_read_vols(HP);

mask = (L<4 & L>0) | L==8;
se = strel(ones(5,5,5));
mask = imdilate(mask,se);

X = X.*mask;

% HX.fname = '/media/dcardenasp/VERBATIM/DataBases/brainweb/t1_pn3_rf40/t1_pn3_rf40_brain.nii';
% spm_write_vol(HX,X);

HX.fname = '/media/dcardenasp/VERBATIM/DataBases/brainweb/t1_pn3_rf40/t1_pn3_rf40_mask.nii';
spm_write_vol(HX,mask);