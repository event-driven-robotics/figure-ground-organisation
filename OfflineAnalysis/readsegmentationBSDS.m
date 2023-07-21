
clc,clear all,close all;
addpath('BSDS300-human','fgcode','GroundTruth','segbench', 'fgsegresults','BSDS300','csvresults')


MAXDIM = 21;

names_mat=load("names.mat");
names=names_mat.names.';

% path_segfile='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/BSDS300-human/BSDS300/human/color/1105/113016.seg';

path_segfile='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/BSDS300-human/BSDS300/human/color/';
path_fgfile='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/GroundTruth/';
path_csv_res='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/csvresults/';
allreults_path='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/';

for idx=1:size(names,1)
    
    %% load segmentation from the BDSD dataset
    name=names(idx);
    img=extractBetween(name,"-",".");
    sbj=extractBefore(name,"-");
    namedir=strcat(allreults_path,sbj,'-',img);
    reults_path=strcat(namedir{1},"/");


    %% load FG ground truth from the BDSD dataset
    filename=strcat(path_fgfile,img,"-",extractAfter(sbj,1),".fg");
    if isfile(filename)
        mkdir(namedir{1});

        seg_path=strcat(path_segfile,sbj,"/",img,".seg");
        seg = readSeg(seg_path);
        bmap = seg2bmap(seg);
        [cmap,cid2sid] = seg2cmap(seg,bmap);
        
        f = figure;
        imagesc(bmap); caxis([0 1]);
        colormap(jet(64));
        name_res_seg=strcat(reults_path,"resseg-",img,"-",sbj,'.png');
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        exportgraphics(f,name_res_seg,'Resolution',300);
        colorbar;


        FG=readbin(filename);
        FGmap=FG{1};
        
        f1 = figure;
        imagesc(FGmap); caxis([-1 1]);
        colormap(parula(5));
        name_res_fg=strcat(reults_path,"resfg-",img,"-",sbj,'.png');
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        exportgraphics(f1,name_res_fg,'Resolution',300);
        colorbar;
        % exportgraphics(f1,'FGgroundtruthlegend.png','Resolution',300);

        %%load csvresult 
        f2 = figure;
        name_csv=strcat(path_csv_res,img,'.csv');
        name_res_csv=strcat(reults_path,"rescsv-",img,"-",sbj,'.png');
        M=csvread(name_csv{1});
        imagesc(M);caxis([-200 200]);
        colormap(parula(5));
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        exportgraphics(f2,name_res_csv,'Resolution',300);
        colorbar;
        % exportgraphics(f2,'FGgroundtruthangles.png','Resolution',300);
    else
        fprintf('file do not exist')
    end



end


print('end')
