%%comparison figure ground maps ED and FG agaist the ground truth
clc,clear all,close all;

path_gt='/home/giuliadangelo/figure-ground-organisation/OfflineAnalysis/fgsegresults/paper/';
path_FBFG='/home/giuliadangelo/figure-ground-organisation/FG_RNN/output/ori/';
path_EDFG='/home/giuliadangelo/figure-ground-organisation/Berkleyresults/results/ori/';

name_img={'12003', '12074', '22090','28075','35070','35091','41004','43070','112082','113016','135069', '156079'}; 

ssim_mse(name_img,path_gt,path_FBFG,path_EDFG)



%% FUNCTIONS

function ssim_mse(name_img,path_gt,path_FBFG,path_EDFG)
    
    num_imgs = length(name_img);
    ssimvals=zeros(1,num_imgs);
    ssimmaps=zeros(321,481,num_imgs);
    mserrors=zeros(1,num_imgs);
    cnt=1;
    
        for name = name_img
            img_gt=rgb2gray(imread(strcat(path_gt,name{1},'.png')));
            img_fbfg=rgb2gray(imread(strcat(path_FBFG,name{1},'.jpg')));
            img_edfg=rgb2gray(imread(strcat(path_EDFG,'FG_',name{1},'.png')));
            [rows,cols]=size(img_fbfg);
            img_gt = imresize(img_gt,[rows cols]);
            img_fbfg = imresize(img_fbfg,[rows cols]);
            img_edfg = imresize(img_edfg,[rows cols]);
       
        %     figure;
        %     subplot(1,2,1);
        %     imshow(img_fbfg);
        %     subplot(1,2,2);
        %     imshow(img_edfg);
        
            %% ssim fb-ed
        %     show_imgs(img_gt,img_fbfg,img_edfg);
            [ssimval,ssimmap] = ssim(img_edfg,img_fbfg);
            ssimvals(1,cnt)=ssimval;
            ssimmaps(:,:,cnt)=ssimmap;
        %     imshow(ssimmap,[])
        %     title("Local SSIM Map with Global SSIM Value: "+num2str(ssimval))
        
        
            %% mse fb-ed
            err = immse(img_edfg,img_fbfg);
            mserrors(1,cnt)=err;
            cnt=cnt+1;
        end
    save('ssimvals.mat','ssimvals');
    save('ssimmaps.mat','ssimmaps');
    save('mserrors.mat','mserrors');

end



function show_imgs(img_gt,img_fbfg,img_edfg)

    figure;
    subplot(1,3,1);
    imagesc(img_gt);
    colormap(jet(64));
    colorbar;
    axis off;
    caxis([0, 360]);
%     title('Ground Truth',FontSize=20);

    subplot(1,3,2);
    imagesc(img_fbfg);
    colormap(jet(64));
    colorbar;
    axis off;
    caxis([0, 360]);
%     title('FB figure-ground',FontSize=20);

    subplot(1,3,3);
    imagesc(img_edfg);
    colormap(jet(64));
    colorbar;
    axis off;
    caxis([0, 360]);
%     title('ED figure-ground',FontSize=20);

end

function normA = normalise(A)
    normA = A - min(A(:));
    normA = normA ./ max(normA(:));
end




