%%comparison figure ground maps ED and FG agaist the ground truth.
%%Computing the SSIM and MSE with the figure ground map.


addpath('phasebar', 'phasemap', 'phasewrap');
clc,clear all,close all;

path_gt='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/GTfigureground/'; %73
path_FBFG='/home/giuliadangelo/figure-ground-organisation/FG_RNN/output/ori/';%306
path_EDFG='/home/giuliadangelo/figure-ground-organisation/Berkleyresults/results/ori/';%197


names_str = dir(path_gt);
names_str=names_str(3:end,:);
len_names=length(names_str);
disp(len_names);
