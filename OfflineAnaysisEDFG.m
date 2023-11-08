% Load all the informations necessary for doing an offline analysis of the
% model: events frame, event pos, event neg, grouping, figure-ground,
% orientation matrix and edge maps at different orientations

clc,clear all,close all;

sft=0;
scale=4;
flipFLAG=0;
SUFFIX = '_test';
ORIENTATIONS = 0;
NUMORI = 16;

orienslist = 0:22.5:337.5;

% {
% heart=160
% tv=80
% }


saving_path='/home/giuliadangelo/figure-ground-organisation/Berkleyresults/results/';
% saving_path='/home/giuliadangelo/figure-ground-organisation/EDFG_RNN_results/results_r08/';

saveFLAG=1;
patternFLAG = 3;
if patternFLAG ==0
     data_path='/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/resultspattericubreal/results_r08/';
     name_str={'heart', 'cat', 'footprint'};
     w=304; %321
     h=240;  %481
elseif patternFLAG ==1
     data_path='/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/resultspattericubreal/results_r08/';
     name_str={'cilinder_cup_bottle'};%'calib_circles', 'tv', 'square_sasso', 'cilinder_cup_bottle', 'key_mouse_flip','calib_circles'};   
     w=304; %321
     h=240;  %481
else
    data_path='/home/giuliadangelo/figure-ground-organisation/Berkleyresults/data/';
    name_str={'35058'}%,'12074','22090','28075','35008','35070','105053','159091'};
     w=481;
     h=321;
end

%%%%%%% experiments on patterns or icub-real %%%%%%%%%

for  name_str = name_str
    name=name_str{1};
    PATH = strcat(data_path, name);
    
    oriens = load(strcat(PATH,'/oriens',SUFFIX,'.csv'));
    image = load(strcat(PATH,'/frame',SUFFIX,'.csv'));
    
    image_pos = load(strcat(PATH,'/img_pos',SUFFIX,'.csv'));
    image_neg = load(strcat(PATH,'/img_neg',SUFFIX,'.csv'));
    grouping = load(strcat(PATH,'/grouping',SUFFIX,'.csv'));
    
    sizeim=size(image);
    if patternFLAG ==0
         rows=[round(sizeim(1)-h+(sft/2)),round(sizeim(1)-(sft/2))];
         cols=[round(sizeim(2)-w+(sft/2)),round(sizeim(2)-(sft/2))];
    elseif patternFLAG ==1 
         rows=[round(1+(sft/2)),round(h-(sft/2))];
         cols=[round(1+(sft/2)),round(w-(sft/2))];
    else
        rows=[1,h];
        cols=[1,w];
    end



    % ORI
    % ha = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    % oriens=oriens(1:240,1:304);
    % imagesc(rad2deg(oriens)),colormap gray,colorbar,title('Oriens Matrix'),drawnow;
    % set(ha(1),'XTick',[]);
    % set(ha(1),'YTick',[]);
    
    % EVENTS FRAME
    figure
    ha1 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    image=image(rows(1):rows(2),cols(1):cols(2));
    if flipFLAG
        image = flipdim(image ,1);
        image = flipdim(image ,2);
    end
    image=imresize(image,scale);
%     imshow(image);
    axes(ha1(1)); imshow(image);
    set(ha1(1),'XTick',[]);
    set(ha1(1),'YTick',[]);
    if saveFLAG ==0
        saving_path_events=fullfile(saving_path,'events',sprintf('%s.jpg',strcat('Frame_',name))),'jpg';
        saveas(ha1,saving_path_events);
    end
    
    % GROUPING
    figure
    ha2 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    grouping=grouping(rows(1):rows(2),cols(1):cols(2));
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
    if saveFLAG ==0
        saving_path_grouping=fullfile(saving_path,'grouping',sprintf('%s.png',strcat('Grouping_',name))),'png';
        saveas(ha2,saving_path_grouping);
    end
    
    
    % FIGURE-GROUND
    figure
    ha3 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    X = load(strcat(PATH,'/X',SUFFIX,'.csv'));
    Y = load(strcat(PATH,'/Y',SUFFIX,'.csv'));
    occ_map = vfcolor(X,Y); 
    occ_map=occ_map(rows(1):rows(2),cols(1):cols(2),:);
    if flipFLAG
        occ_map = flipdim(occ_map ,1);
        occ_map = flipdim(occ_map ,2);
    end
    occ_map=imresize(occ_map,scale);
    legendpath='/home/giuliadangelo/figure-ground-organisation/FG_RNN/Wheel.png';
    lgd=im2double(imread(legendpath));
    lgd = imresize(lgd, 2);
    sza= size(lgd);
    shift=10;
    occ_map(shift:shift+sza(1)-1,end-(sza(2)+shift):end-shift-1, :)=lgd;
    imagesc(occ_map);
    set(ha3(1),'XTick',[]);
    set(ha3(1),'YTick',[]);
    if saveFLAG ==1
        saving_path_figure=fullfile(saving_path,'ori',sprintf('%s.png',strcat('FG_',name))),'png';
        saveas(ha3,saving_path_figure);
    end

end



clc,clear all,close all;