%%comparison figure ground maps ED and FG agaist the ground truth.
%%Computing the SSIM and MSE with the figure ground map.
addpath('phasebar', 'phasemap', 'phasewrap');
clc,clear all,close all;

%% generate images to be compared FB-GT-ED
% imgs_generation();

%% analysis on the FB-GT-ED computing SSIM & MSE
ssim_mse();

%% PLOT analysis on the FB-GT-ED computing SSIM & MSE




function ssim_mse()
    namefigures=load("namefigures.mat");
    namefigures=namefigures.namefigures;
    len_names=length(namefigures);
    
    labelsfigures=load('labelsfigures.mat');
    labelsfigures=labelsfigures.labelsfigures;
    
    path_gt='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/nolegend/GTangleinfo/'; %73
    path_FBFG='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/nolegend/FBangleinfo/';%306
    path_EDFG='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/nolegend/EDangleinfo/';%197
    
    FBGTssimvals=zeros(1,len_names);
    FBGTssimmaps=cell(len_names,1);
    FBGTmserrors=zeros(1,len_names);
    
    EDGTssimvals=zeros(1,len_names);
    EDGTssimmaps=cell(len_names,1);
    EDGTmserrors=zeros(1,len_names);
    
    EDFBssimvals=zeros(1,len_names);
    EDFBssimmaps=cell(len_names,1);
    EDFBmserrors=zeros(1,len_names);
    
    for idx=1:len_names
        disp(idx)
        name=namefigures(idx);
        img_gt=imread(strcat(path_gt,name,'.png'));
        img_fbfg=imread(strcat(path_FBFG,name,'.png'));
        img_edfg=imread(strcat(path_EDFG,name,'.png'));
    
        imshow(img_gt);
        imshow(img_fbfg);
        imshow(img_edfg);
    
        %% FB-GT
        % ssim
        [FBGTssimval,FBGTssimmap] = ssim(img_fbfg,img_gt);
        FBGTssimvals(1,idx)=FBGTssimval;
        FBGTssimmaps{idx}=FBGTssimmap;
        % mse
        FBGTerr = immse(img_fbfg,img_gt);
        FBGTmserrors(1,idx)=FBGTerr;
    
        %% ED-GT
        %ssim
        [EDGTssimval,EDGTssimmap] = ssim(img_edfg,img_gt);
        EDGTssimvals(1,idx)=EDGTssimval;
        EDGTssimmaps{idx}=EDGTssimmap;
    
        %mse
        EDGTerr = immse(img_edfg,img_gt);
        EDGTmserrors(1,idx)=EDGTerr;
    
        %% ED-FB
        %ssim
        [ssimval,ssimmap] = ssim(img_edfg,img_fbfg);
        EDFBssimvals(1,idx)=ssimval;
        EDFBssimmaps{idx}=ssimmap;
    
        %mse
        EDFBerr = immse(img_edfg,img_fbfg);
        EDFBmserrors(1,idx)=EDFBerr;
    end
    
    save('EDGTssimvals.mat','EDGTssimvals');
    save('EDGTssimmaps.mat','EDGTssimmaps');
    save('EDGTmserrors.mat','EDGTmserrors');

    save('FBGTssimvals.mat','FBGTssimvals');
    save('FBGTssimmaps.mat','FBGTssimmaps');
    save('FBGTmserrors.mat','FBGTmserrors');

    save('EDFBssimvals.mat','EDFBssimvals');
    save('EDFBssimmaps.mat','EDFBssimmaps');
    save('EDFBmserrors.mat','EDFBmserrors');
end




function imgs_generation()
    path_gt='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/GTfigureground/'; %73
    path_FBFG='/home/giuliadangelo/figure-ground-organisation/FG_RNN/output/ori/';%306
    path_EDFG='/home/giuliadangelo/figure-ground-organisation/Berkleyresults/results/ori/';%197
    
    names_str = dir(path_gt);
    names_str=names_str(3:end,:);
    len_names=length(names_str);
    disp(len_names);
    namefigures=string((len_names));
    labelsfigures=string((len_names));
    cnt = 1;
    for idx=1:len_names
        disp(idx)
        name = extractBetween(names_str(idx).name , '-' , '-' );
        namefigures(cnt)=name;
        labelsfigures(cnt)=strcat('# ',int2str(idx));
        img_gt=rgb2gray(imread(strcat(path_gt,names_str(idx).name)));
        img_fbfg=rgb2gray(imread(strcat(path_FBFG,name{1},'.jpg')));
        img_edfg=rgb2gray(imread(strcat(path_EDFG,'FG_',name{1},'.png')));
    
        [rows,cols]=size(img_fbfg);
        img_gt = imresize(img_gt,[rows cols]);
        img_fbfg = imresize(img_fbfg,[rows cols]);
        img_edfg = imresize(img_edfg,[rows cols]);
        % generate images with legend or not, be careful to change it inside
        generate_imgsGT(img_gt,img_fbfg,img_edfg, name{1});
        cnt=cnt+1;
    end
    save('namefigures.mat','namefigures');
    save('labelsfigures.mat','labelsfigures');
end





function generate_imgsGT(img_gt,img_fbfg,img_edfg, name)

            fgt=figure('visible','off');
%             subplot(1,3,1);
            img_gt(img_gt>180)=img_gt(img_gt>180)-180;
            imagesc(img_gt);
            colormap('hsv');
%             phasebar('location','se');
            caxis([0,180]);
            axis off;
            labelGT=strcat('/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/nolegend/GTangleinfo/', name, '.png');
            saveas(fgt, labelGT);
        
            ffb=figure('visible','off');
%             subplot(1,3,2);
            img_fbfg(img_fbfg>180)=img_fbfg(img_fbfg>180)-180;
            imagesc(img_fbfg);
            colormap('hsv');
%             phasebar('location','se');
            caxis([0,180]);
            axis off;
            labelFB=strcat('/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/nolegend/FBangleinfo/', name, '.png');
            saveas(ffb, labelFB);
        
            fed= figure('visible','off');
%             subplot(1,3,3);
            img_edfg(img_edfg>180)=img_edfg(img_edfg>180)-180;
            imagesc(img_edfg);
            colormap('hsv');
%             phasebar('location','se');
            caxis([0,180]);
            axis off;
            labelED=strcat('/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/nolegend/EDangleinfo/', name, '.png');
            saveas(fed, labelED);

end