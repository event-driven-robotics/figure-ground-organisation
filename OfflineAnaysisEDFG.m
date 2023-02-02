% Load all the informations necessary for doing an offline analysis of the
% model: events frame, event pos, event neg, grouping, figure-ground,
% orientation matrix and edge maps at different orientations

clc,clear all,close all;
w=304;
h=240;
sft=10;
scale=4;

SUFFIX = '_test';
ORIENTATIONS = 0;
NUMORI = 16;

orienslist = 0:22.5:337.5;



data_path='/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/results/';
saving_path='/home/giuliadangelo/figure-ground-organisation/EDFG_RNN_results/results/';

flipFLAG=1;
for  name_str={'heart'}%, 'cat', 'footprint'}
% flipFLAG=0;
% for  name_str={'tv','key_mouse_flip','square_sasso','calib_circles', 'cilinder_cup_bottle'}
    name=name_str{1};
    PATH = strcat(data_path, name);
    
    oriens = load(strcat(PATH,'/oriens',SUFFIX,'.csv'));
    image = load(strcat(PATH,'/frame',SUFFIX,'.csv'));
    
    image_pos = load(strcat(PATH,'/img_pos',SUFFIX,'.csv'));
    image_neg = load(strcat(PATH,'/img_neg',SUFFIX,'.csv'));
    grouping = load(strcat(PATH,'/grouping',SUFFIX,'.csv'));
    
    % ORI
    % ha = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    % oriens=oriens(1:240,1:304);
    % imagesc(rad2deg(oriens)),colormap gray,colorbar,title('Oriens Matrix'),drawnow;
    % set(ha(1),'XTick',[]);
    % set(ha(1),'YTick',[]);
    
    % EVENTS FRAME
    figure
    ha1 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    image=image(sft:h-sft,sft:w-sft);
    if flipFLAG
        image = flipdim(image ,1);
        image = flipdim(image ,2);
    end
    image=imresize(image,scale);
    imshow(image);
    set(ha1(1),'XTick',[]);
    set(ha1(1),'YTick',[]);
    imwrite(image,fullfile(saving_path,'events',sprintf('%s.png',name)),'png');
    
    % GROUPING
    figure
    ha2 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    grouping=grouping(sft:h-sft,sft:w-sft);
    if flipFLAG
        grouping = flipdim(grouping ,1);
        grouping = flipdim(grouping ,2);
    end
    grouping=imresize(grouping,scale);
    grouping=grouping-min(grouping(:));
    grouping=grouping./max(grouping(:));
    axes(ha2(1)); imagesc(grouping); 
    colormap(jet(64));
    colorbar;
    set(ha2(1),'XTick',[]);
    set(ha2(1),'YTick',[]);
    saving_path_grouping=fullfile(saving_path,'grouping',sprintf('%s.png',name)),'png';
    saveas(ha2,saving_path_grouping);
    
    
    % FIGURE-GROUND
    figure
    ha3 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    X = load(strcat(PATH,'/X',SUFFIX,'.csv'));
    Y = load(strcat(PATH,'/Y',SUFFIX,'.csv'));
    occ_map = vfcolor(X,Y);
    occ_map=occ_map(sft:h-sft,sft:w-sft, :);
    if flipFLAG
        occ_map = flipdim(occ_map ,1);
        occ_map = flipdim(occ_map ,2);
    end
    occ_map=imresize(occ_map,scale);
    % occ_map = flipdim(occ_map ,1);
    % occ_map=occ_map(end-h:end,1:w, :);
    legendpath='/home/giuliadangelo/figure-ground-organisation/Wheel.png';
    lgd=im2double(imread(legendpath));
    lgd = imresize(lgd, 0.8);
    sza= size(lgd);
    shift=10;
    % im(1:sza(1),end-(sza(2))+1:end,:)=lgd;
    occ_map(shift:shift+sza(1)-1,end-(sza(2)+shift)+1:end-shift,:)=lgd;
    imagesc(occ_map);
    set(ha3(1),'XTick',[]);
    set(ha3(1),'YTick',[]);
    saving_path_figure=fullfile(saving_path,'ori',sprintf('%s.png',strcat('FG_',name))),'png';
    saveas(ha3,saving_path_figure);

end



clc,clear all,close all;