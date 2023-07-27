% Load all the informations necessary for doing an offline analysis of the
% model: events frame, event pos, event neg, grouping, figure-ground,
% orientation matrix and edge maps at different orientations

clc,clear all,close all;
w=321;
h=481;
sft=0;
scale=4;
SUFFIX = '_test';
ORIENTATIONS = 0;
NUMORI = 16;

orienslist = 0:22.5:337.5;

data_path='/home/giuliadangelo/figure-ground-organisation/Berkleyresults/data/';
saving_path='/home/giuliadangelo/figure-ground-organisation/Berkleyresults/results/';
name_str=dir(data_path);
name_str=name_str(3:end,:);
len_names=length(name_str);
disp(len_names);


for idx=177:len_names
    disp(idx);
    name=name_str(idx).name;
    PATH = strcat(data_path, name);
    
    oriens = load(strcat(PATH,'/oriens',SUFFIX,'.csv'));
    image = load(strcat(PATH,'/frame',SUFFIX,'.csv'));
    
    image_pos = load(strcat(PATH,'/img_pos',SUFFIX,'.csv'));
    image_neg = load(strcat(PATH,'/img_neg',SUFFIX,'.csv'));
    grouping = load(strcat(PATH,'/grouping',SUFFIX,'.csv'));
    
    sizeim=size(image);
    rows=[round(1+(sft/2)),round(h-(sft/2))];
    cols=[round(1+(sft/2)),round(w-(sft/2))];


    % ORI
    % ha = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    % oriens=oriens(1:240,1:304);
    % imagesc(rad2deg(oriens)),colormap gray,colorbar,title('Oriens Matrix'),drawnow;
    % set(ha(1),'XTick',[]);
    % set(ha(1),'YTick',[]);
    
    % EVENTS FRAME
    figure
    ha1 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    image=imresize(image,scale);
    imshow(image);
    axes(ha1(1)); imshow(image);
    set(ha1(1),'XTick',[]);
    set(ha1(1),'YTick',[]);
    saving_path_events=fullfile(saving_path,'events',sprintf('%s.jpg',strcat('Frame_',name))),'jpg';
    saveas(ha1,saving_path_events);
    
    % GROUPING
    figure
    ha2 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    grouping=imresize(grouping,scale);
    imshow(grouping);
    grouping=grouping-min(grouping(:));
    grouping=grouping./max(grouping(:));
    axes(ha2(1)); imagesc(grouping); 
    colormap(jet(64));
    colorbar;
    set(ha2(1),'XTick',[]);
    set(ha2(1),'YTick',[]);
    saving_path_grouping=fullfile(saving_path,'grouping',sprintf('%s.png',strcat('Grouping_',name))),'png';
    saveas(ha2,saving_path_grouping);
    
    
    % FIGURE-GROUND
    figure
    ha3 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    X = load(strcat(PATH,'/X',SUFFIX,'.csv'));
    Y = load(strcat(PATH,'/Y',SUFFIX,'.csv'));
    occ_map = vfcolor(X,Y); 
    occ_map=imresize(occ_map,scale);
%     legendpath='/home/giuliadangelo/figure-ground-organisation/Wheel.png';
%     lgd=im2double(imread(legendpath));
%     lgd = imresize(lgd, 0.8);
%     sza= size(lgd);
%     shift=10;
%     occ_map(shift:shift+sza(1)-1,end-(sza(2)+shift):end-shift-1, :)=lgd;
    imagesc(occ_map);
    set(ha3(1),'XTick',[]);
    set(ha3(1),'YTick',[]);
    saving_path_figure=fullfile(saving_path,'ori',sprintf('%s.png',strcat('FG_',name))),'png';
    saveas(ha3,saving_path_figure);

end



clc,clear all,close all;