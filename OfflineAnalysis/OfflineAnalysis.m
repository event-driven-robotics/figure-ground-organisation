% Load all the informations necessary for doing an offline analysis of the
% model: events frame, event pos, event neg, grouping, figure-ground,
% orientatiosn amtrix and edge maps at different orientations

clc,clear all,close all;

IMAGE = '135069';
SUFFIX = '_';
ORIENTATIONS = 0;
NUMORI = 16;

orienslist = 0:22.5:337.5;


oriens = load(strcat('./Results/',IMAGE,'/oriens',SUFFIX,IMAGE,'.csv'));
image = load(strcat('./Results/',IMAGE,'/frame',SUFFIX,IMAGE,'.csv'));

image_pos = load(strcat('./Results/',IMAGE,'/img_pos',SUFFIX,IMAGE,'.csv'));
image_neg = load(strcat('./Results/',IMAGE,'/img_neg',SUFFIX,IMAGE,'.csv'));

% grouping = load(strcat('./Results/',IMAGE,'/grouping',SUFFIX,IMAGE,'.csv'));

figure,imagesc(rad2deg(oriens)),colormap gray,colorbar,title('Oriens Matrix'),drawnow;
figure,imshow(image),title('Events frame'),drawnow;
figure,imshow(image_pos),title('Image pos'),drawnow;
figure,imshow(image_neg),title('Image neg'),drawnow;
% figure,imagesc(grouping),colormap jet,colorbar;

if ORIENTATIONS
    
    figure

    for i=1:NUMORI
        
        subplot(4,4,i)
        temp = load(strcat('./Results/',IMAGE,'/resp',string(orienslist(i)),SUFFIX,IMAGE,'.csv'));
        
        imshow(temp),title(strcat(string(orienslist(i)),'°'))
        
    end
       
end

wheel = imread('Wheel.png');

X = load(strcat('./Results/',IMAGE,'/X',SUFFIX,IMAGE,'.csv'));
Y = load(strcat('./Results/',IMAGE,'/Y',SUFFIX,IMAGE,'.csv'));

occ_map = vfcolor(X,Y);

figure
subplot(1,2,1)

imshow(occ_map),title('Figure-Ground')

subplot(1,2,2)
imshow(wheel)