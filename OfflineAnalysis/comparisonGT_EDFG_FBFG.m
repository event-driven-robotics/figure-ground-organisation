%%comparison figure ground maps ED and FG agaist the ground truth.
%%Computing the SSIM and MSE with the figure ground map.
addpath('phasebar', 'phasemap', 'phasewrap');
clc,clear all,close all;

analysisFLAG=1;
SSIM_MSE_FLAG=0;
saveFLAG=1;

path_gt='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/GTfigureground/'; %73
path_FBFG='/home/giuliadangelo/figure-ground-organisation/FG_RNN/output/ori/';%306
path_EDFG='/home/giuliadangelo/figure-ground-organisation/Berkleyresults/results/ori/';%197

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%computation of ssim and mse: fg vs ed & gt vs fg vs ed 
if analysisFLAG
    ssim_mseFGED(path_gt,path_FBFG,path_EDFG,saveFLAG)
    ssim_mseGTFGED(path_gt,path_FBFG,path_EDFG,saveFLAG)
end

if SSIM_MSE_FLAG
    %% load the SSIM and MSE between the FBFG and EDFG
    [FBEDFGssimvals,FBEDFGssimvals_mean,FBEDFGssimvals_var,labelnames,labelnamesval]=valsEDvsFB();
    [EDFBvalsort,FBEDindx] = sort(FBEDFGssimvals,'descend');
    EDFBlabelnames=labelnames(FBEDindx);
    
    EDFBmserrors=load('EDFBmserrors.mat');
    EDFBmserrors=EDFBmserrors.EDFBmserrors;
    EDFBmserrors=EDFBmserrors(FBEDindx);
    
    figure;
    subplot(3,1,1);
    
    %%bar
    x=categorical(EDFBlabelnames);
    x = reordercats(x,string(x));
    y=EDFBvalsort(1,:);
    ax=gca;
    bar(x,y,0.8,'FaceColor',[0.9290 0.6940 0.1250],'EdgeColor',[0 0 0],'LineWidth',1.5);
    hold on;
    % curve1 = FBEDFGssimvals_mean + FBEDFGssimvals_var;
    % curve2 = FBEDFGssimvals_mean - FBEDFGssimvals_var;
    % x2 = [x, fliplr(x)];
    % inBetween = [curve1, fliplr(curve2)];
    % f=fill(x2, inBetween, 'yellow');
    % alpha(f,.5)
    % hold on;
    
    %%mean
    plot(x,FBEDFGssimvals_mean,'LineWidth',3);
    
    legend('EDvsFB','mean',FontSize=20);
    xtickangle(45)
    ax.XAxis.FontSize = 15;
    ax.YAxis.FontSize = 26;
    x0=10;
    y0=10;
    width=10000;
    height=1000;
    set(gcf,'position',[x0,y0,width,height]);
    set(gcf,'units','points','position',[x0,y0,width,height]);
    hold off;

    
    %% load the SSIM and MSE between the FBFG against the GT
    [FBGTssimvals,FBGTssimvals_mean,FBGTssimvals_var]=valsFBvsGT();
    [FBGTvalsort,FBGTindx] = sort(FBGTssimvals,'descend');
    FBGTlabelnames=labelnames(FBGTindx);
    
    FBGTmserrors=load('FBGTmserrors.mat');
    FBGTmserrors=FBGTmserrors.FBGTmserrors;
    FBGTmserrors=FBGTmserrors(FBGTindx);
    
    
    subplot(3,1,2);
    x=categorical(FBGTlabelnames);
    x = reordercats(x,string(x));
    
    y=FBGTvalsort(1,:);
    ax=gca;
    bar(x,y,0.8,'FaceColor',[0.4660 0.6740 0.1880],'EdgeColor',[0 0 0],'LineWidth',1.5);
    hold on;
    % curve1 = FBGTssimvals_mean + FBGTssimvals_var;
    % curve2 = FBGTssimvals_mean - FBGTssimvals_var;
    % x2 = [x, fliplr(x)];
    % inBetween = [curve1, fliplr(curve2)];
    % f=fill(x2, inBetween, 'yellow');
    % alpha(f,.5)
    % hold on;
    plot(x,FBGTssimvals_mean,'LineWidth',3);
    legend('FBvsGT','mean',FontSize=20)
    xtickangle(45)
    ax.XAxis.FontSize = 15;
    ax.YAxis.FontSize = 26;
    x0=10;
    y0=10;
    width=10000;
    height=1000;
    set(gcf,'position',[x0,y0,width,height]);
    set(gcf,'units','points','position',[x0,y0,width,height]);
    
    
    
    
    
    %% load the SSIM and MSE between the EDFG against the GT
    
    [EDGTssimvals,EDGTssimvals_mean,EDGTssimvals_var]=valsEDvsGT();
    [EDGTvalsort,EDGTindx] = sort(EDGTssimvals,'descend');
    EDGTlabelnames=labelnames(EDGTindx);
    
    EDGTmserrors=load('EDGTmserrors.mat');
    EDGTmserrors=EDGTmserrors.EDGTmserrors;
    EDGTmserrors=EDGTmserrors(EDGTindx);
    
    
    subplot(3,1,3);
    x=categorical(EDGTlabelnames);
    x = reordercats(x,string(x));
    
    y=EDGTvalsort(1,:);
    ax=gca;
    bar(x,y,0.8,'FaceColor',[0.3010 0.7450 0.9330],'EdgeColor',[0 0 0],'LineWidth',1.5);
    hold on;
    % curve1 = EDGTssimvals_mean + EDGTssimvals_var;
    % curve2 = EDGTssimvals_mean - EDGTssimvals_var;
    % x2 = [x, fliplr(x)];
    % inBetween = [curve1, fliplr(curve2)];
    % f=fill(x2, inBetween, 'yellow');
    % alpha(f,.5)
    % hold on;
    plot(x,EDGTssimvals_mean,'LineWidth',3);
    legend('EDvsGT','mean',FontSize=20)
    xtickangle(45)
    ax.XAxis.FontSize = 15;
    ax.YAxis.FontSize = 26;
    x0=10;
    y0=10;
    width=10000;
    height=1000;
    set(gcf,'position',[x0,y0,width,height]);
    set(gcf,'units','points','position',[x0,y0,width,height]);
    hold off;
    
    
    
    figure;
    subplot(3,1,1);
    %%MSE
    plot(x,EDFBmserrors,'LineWidth',3);
    legend('EDvsFB',FontSize=20)
    %%MSE
    subplot(3,1,2);
    plot(x,FBGTmserrors,'LineWidth',3);
    legend('FBvsGT',FontSize=20)
    %%MSE
    subplot(3,1,3);
    plot(x,EDGTmserrors,'LineWidth',3);
    legend('EDvsGT',FontSize=20)
end


disp('end')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% FUNCTIONS
function [FBGTssimvals,FBGTssimvals_mean,FBGTssimvals_var]=valsFBvsGT()

    FBGTssimvals= load('FBGTssimvals.mat');
    FBGTssimvals=FBGTssimvals.FBGTssimvals;
    FBGTssimvals_mean=mean(FBGTssimvals);
    FBGTssimvals_var=var(FBGTssimvals);
    FBGTssimvals_mean= ones(1,length(FBGTssimvals))*FBGTssimvals_mean;
    

end

function [EDGTssimvals,EDGTssimvals_mean,EDGTssimvals_var]=valsEDvsGT()

    EDGTssimvals= load('EDGTssimvals.mat');
    EDGTssimvals=EDGTssimvals.EDGTssimvals;
    EDGTssimvals_mean=mean(EDGTssimvals);
    EDGTssimvals_var=var(EDGTssimvals);
    EDGTssimvals_mean= ones(1,length(EDGTssimvals))*EDGTssimvals_mean;
    

end

function [FBEDFGssimvals,FBEDFGssimvals_mean,FBEDFGssimvals_var,labelnames,labelnamesval]=valsEDvsFB()

    FBEDFGssimvals= load('EDFBssimvals.mat');
    FBEDFGssimvals=FBEDFGssimvals.EDFBssimvals;
    FBEDFGssimvals_mean=mean(FBEDFGssimvals);
    FBEDFGssimvals_var=var(FBEDFGssimvals);
    
    labelnames=load('EDFBnamesimages.mat');
    labelnames=labelnames.EDFBnamesimages;

    labelnamesval=load('EDFBlabelsimages.mat');
    labelnamesval=labelnamesval.EDFBlabelsimages;
    
    FBEDFGssimvals_mean= ones(1,length(FBEDFGssimvals))*FBEDFGssimvals_mean;
    

end   


function ssim_mseFGED(path_gt,path_FBFG,path_EDFG, saveFLAG)
    
    names_str = dir(path_gt);
    names_str=names_str(3:end,:);
    len_names=length(names_str);
    disp(len_names);
    
    EDFBssimvals=zeros(1,len_names);
    EDFBssimmaps=cell(len_names,1);
    EDFBmserrors=zeros(1,len_names);
    EDFBnamesimages=string([len_names]);
    EDFBlabelsimages=string([len_names]);
    cnt=1;
    
        for idx=1:len_names
            disp(idx)
            name = extractBetween(names_str(idx).name , '-' , '-' );
            EDFBnamesimages(cnt)=name;
            EDFBlabelsimages(cnt)=strcat('# ',int2str(idx));
            img_fbfg=rgb2gray(imread(strcat(path_FBFG,name{1},'.jpg')));
            img_edfg=rgb2gray(imread(strcat(path_EDFG,'FG_',name{1},'.png')));
            [rows,cols]=size(img_fbfg);
            img_fbfg = imresize(img_fbfg,[rows cols]);
            img_edfg = imresize(img_edfg,[rows cols]);
       
        
            %% ssim fb-ed
%             show_fbed(img_fbfg,img_edfg, name{1});

            [ssimval,ssimmap] = ssim(img_edfg,img_fbfg);
            EDFBssimvals(1,cnt)=ssimval;
            EDFBssimmaps{cnt}=ssimmap;

            %show ssim map
%             ha = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
%             axes(ha(1)); 
%             imagesc(ssimmap); %caxis([0 1]);
%             imshow(ssimmap,[])
%             title("Local SSIM Map with Global SSIM Value: "+num2str(ssimval))
    
            %% mse fb-ed
            err = immse(img_edfg,img_fbfg);
            EDFBmserrors(1,cnt)=err;
            cnt=cnt+1;
        end
        if saveFLAG
            save('EDFBssimvals.mat','EDFBssimvals');
            save('EDFBssimmaps.mat','EDFBssimmaps');
            save('EDFBmserrors.mat','EDFBmserrors');
            save('EDFBnamesimages.mat','EDFBnamesimages');
            save('EDFBlabelsimages.mat','EDFBlabelsimages');
        end
    
    end
    
    
    
    function ssim_mseGTFGED(path_gt,path_FBFG,path_EDFG, saveFLAG)
        
        names_str = dir(path_gt);
        names_str=names_str(3:end,:);
        len_names=length(names_str);
        disp(len_names);
        
        FBGTssimvals=zeros(1,len_names);
        FBGTssimvals=zeros(1,len_names);
        FBGTssimmaps=cell(len_names,1);
        FBGTmserrors=zeros(1,len_names);
    
        EDGTssimvals=zeros(1,len_names);
        EDGTssimvals=zeros(1,len_names);
        EDGTssimmaps=cell(len_names,1);
        EDGTmserrors=zeros(1,len_names);
    
        namefigures=string([len_names]);
        labelsfigures=string([len_names]);
    
       cnt=1;
        
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

%             figGT = figure();
%             imagesc(img_gt);
%             colormap(jet(64));
%             colorbar;
%             axis off;
%             caxis([0,350]);
%             labelGT=strcat('/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/GTfigureground/', name, '.png');
%             saveas(figGT, labelGT);
        
            %% ssim fb-gt
%             show_gtfbed(img_gt,img_fbfg,img_edfg, name{1});

            [FBGTssimval,FBGTssimmap] = ssim(img_fbfg,img_gt);
            FBGTssimvals(1,cnt)=FBGTssimval;
            FBGTssimmaps{cnt}=FBGTssimmap;

%             ha = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
%             axes(ha(1)); 
%             imagesc(FBGTssimmap); %caxis([0 1]);
%             colormap(jet(64));
%             colorbar;
%             set(ha(1),'XTick',[]);
%             set(ha(1),'YTick',[]);
%             saving_path=fullfile('comparisonFBGT/',sprintf('%s.png',name{1})),'png';
%             saveas(ha,saving_path);
            
%             imshow(ssimmap,[])
%             title("Local SSIM Map with Global SSIM Value: "+num2str(ssimval))
    
            %% ssim ed-gt
            [EDGTssimval,EDGTssimmap] = ssim(img_edfg,img_gt);
            EDGTssimvals(1,cnt)=EDGTssimval;
            EDGTssimmaps{cnt}=EDGTssimmap;


%             ha = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
%             axes(ha(1)); 
%             imagesc(EDGTssimmap); %caxis([0 1]);
%             colormap(jet(64));
%             colorbar;
%             set(ha(1),'XTick',[]);
%             set(ha(1),'YTick',[]);
%             saving_path=fullfile('comparisonEDGT/',sprintf('%s.png',name{1})),'png';
%             saveas(ha,saving_path);
            
%             imshow(ssimmap,[])
%             title("Local SSIM Map with Global SSIM Value: "+num2str(ssimval))


            %% mse fb-gt
            FBGTerr = immse(img_fbfg,img_gt);
            FBGTmserrors(1,cnt)=FBGTerr;

            %% mse ed-gt
            EDGTerr = immse(img_edfg,img_gt);
            EDGTmserrors(1,cnt)=EDGTerr;




            cnt=cnt+1;
        end
    if saveFLAG

        save('EDGTssimvals.mat','EDGTssimvals');
        save('EDGTssimmaps.mat','EDGTssimmaps');
        save('EDGTmserrors.mat','EDGTmserrors');
    
        save('FBGTssimvals.mat','FBGTssimvals');
        save('FBGTssimmaps.mat','FBGTssimmaps');
        save('FBGTmserrors.mat','FBGTmserrors');
    
        save('namefigures.mat','namefigures');
        save('labelsfigures.mat','labelsfigures');
    end

end





function show_fbed(img_fbfg,img_edfg, name)

%         subplot(1,2,1);
        figFB = figure();
        imagesc(img_fbfg);
        colormap(jet(64));
        colorbar;
        axis off;
        caxis([0, 360]);
        labelFB=strcat('/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/FBangleinfo/', name, '.png');
        saveas(figFB, labelFB);
    %     title('FB figure-ground',FontSize=20);
    
%         subplot(1,2,2);
        figED = figure();
        imagesc(img_edfg);
        colormap(jet(64));
        colorbar;
        axis off;
        caxis([0, 360]);
        labelED=strcat('/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/EDangleinfo/', name, '.png');
        saveas(figED, labelED);
  %     title('ED figure-ground',FontSize=20);
end



function show_gtfbed(img_gt,img_fbfg,img_edfg, name)

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

function normA = normalise(A)
    normA = A - min(A(:));
    normA = normA ./ max(normA(:));
end




