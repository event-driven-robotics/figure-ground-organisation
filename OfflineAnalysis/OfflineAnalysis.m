% Load all the informations necessary for doing an offline analysis of the
% model: events frame, event pos, event neg, grouping, figure-ground,
% orientation matrix and edge maps at different orientations

clc,clear all,close all;
PATH = '/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/results/key_mouse_flip';


%%%%%%%% HERE find the results for the remaining images 


SUFFIX = '_test';
ORIENTATIONS = 0;
NUMORI = 16;

orienslist = 0:22.5:337.5;


oriens = load(strcat(PATH,'/oriens',SUFFIX,'.csv'));
image = load(strcat(PATH,'/frame',SUFFIX,'.csv'));

image_pos = load(strcat(PATH,'/img_pos',SUFFIX,'.csv'));
image_neg = load(strcat(PATH,'/img_neg',SUFFIX,'.csv'));
grouping = load(strcat(PATH,'/grouping',SUFFIX,'.csv'));
% grouping = load(strcat('./Results/',IMAGE,'/grouping',SUFFIX,IMAGE,'.csv'));

figure,imagesc(rad2deg(oriens)),colormap gray,colorbar,title('Oriens Matrix'),drawnow;
figure,imshow(image),title('Events frame'),drawnow;
figure,imshow(image_pos),title('Image pos'),drawnow;
figure,imshow(image_neg),title('Image neg'),drawnow;
figure,imagesc(grouping),colormap jet,colorbar;

if ORIENTATIONS
    
    figure;

    for i=1:NUMORI
        
        subplot(4,4,i)
        temp = load(strcat(PATH,'/resp',string(orienslist(i)),SUFFIX,IMAGE,'.csv'));
%         temp = load(strcat('./Results/',IMAGE,'/resp',string(orienslist(i)),SUFFIX,IMAGE,'.csv'));
        
        imshow(temp),title(strcat(string(orienslist(i)),'°'))
        
    end
       
end

wheel = imread('/home/giuliadangelo/figure-ground-organisation/Wheel.png');

X = load(strcat(PATH,'/X',SUFFIX,'.csv'));
Y = load(strcat(PATH,'/Y',SUFFIX,'.csv'));

occ_map = vfcolor(X,Y);

figure
subplot(1,2,1)

imshow(occ_map),title('Figure-Ground')

subplot(1,2,2)
imshow(wheel)