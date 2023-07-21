clc,clear all,close all;
addpath('BSDS300-human','fgcode','GroundTruth','segbench', 'fgsegresults')


% imagefiles = dir('./GroundTruth/*.fg');

dirinfo = dir('./BSDS300-human/BSDS300/human/color/');
dirinfo(~[dirinfo.isdir]) = [];
names={};
for K = 3 : length(dirinfo)
  thisdir=dirinfo(K).name
  subdirinfo = dir(strcat('./BSDS300-human/BSDS300/human/color/',thisdir));
  for I = 3 : length(subdirinfo)
    names=[names,strcat(thisdir,'-',subdirinfo(I).name)]
  end
end
save('names.mat','names');