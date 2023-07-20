clc, clear all, close all;

%path
im_path='/home/giuliadangelo/figure-ground-organisation/FG_RNN/output/ori/'
% name='FG_tv_RGB';
% im_path = 'Circles';
% im_path = 'footprint';
% im_path = 'gatto';
% im_path = 'heart';
% im_path = 'square_sasso';
% im_path = 'tv';

name='159091.jpg';
legendpath='/home/giuliadangelo/figure-ground-organisation/FG_RNN/Wheel.png';
im=im2double(imread(fullfile(im_path,name)));
lgd=im2double(imread(legendpath));
lgd=imresize(lgd,0.5);
sza= size(lgd);
shift=10;
% im(1:sza(1),end-(sza(2))+1:end,:)=lgd;
im(shift:shift+sza(1)-1,end-(sza(2)+shift)+1:end-shift,:)=lgd;
imshow(im);
imwrite(im,fullfile('output/ori_lgd/',sprintf('%s.png',name)),'png')


fprintf('\nDone\n')

