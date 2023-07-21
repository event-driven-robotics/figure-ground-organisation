
clc,clear all,close all;
addpath('BSDS300-human','fgcode','GroundTruth')

MAXDIM = 21;

imagefiles = dir('./GroundTruth/*.fg');
path_segfile='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/BSDS300-human/BSDS300/human/color/1105/113016.seg';
path_fgfile='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/GroundTruth/113016-105.fg';


%% load segmentation from the BDSD dataset
seg = readSeg(path_segfile);
bmap = seg2bmap(seg);
[cmap,cid2sid] = seg2cmap(seg,bmap);


ha1 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
axes(ha1(1)); imagesc(bmap); caxis([0 1]);
colormap(jet(64));
colorbar;
set(ha1(1),'XTick',[]);
set(ha1(1),'YTick',[])

%% load FG ground truth from the BDSD dataset
FG=readbin(path_fgfile);
FGmap=FG{1};

f1 = figure;
ha2 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
axes(ha2(1)); imagesc(FGmap); caxis([-1 1]);
colormap(parula(5));
% exportgraphics(f1,'FGgroundtruth.png','Resolution',300);
colorbar;
set(ha2(1),'XTick',[]);
set(ha2(1),'YTick',[]);
exportgraphics(f1,'FGgroundtruthlegend.png','Resolution',300);


%%load csvresult 
f2 = figure;
M=csvread('/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/csvresults/113016.csv');
ha3 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
axes(ha3(1)); imagesc(M);caxis([-200 200]);
colormap(parula(5));
colorbar;
set(ha3(1),'XTick',[]);
set(ha3(1),'YTick',[]);
exportgraphics(f2,'FGgroundtruthangles.png','Resolution',300);




print('end')
