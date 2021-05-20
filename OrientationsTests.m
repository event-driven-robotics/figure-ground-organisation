%% DoG standard
clc,clear all,close all

img = im2double(imread('42049.bmp'));

sigma = 60;
sigmaRatio = 0.5;
sz = size(img);

g1 = fspecial('gaussian',sz,sigma);
g2 = fspecial('gaussian',sz,sigma*sigmaRatio);

G = g1 - g2;
surf(G)

result = conv2(G,img,'same');

figure,imshow(img)
figure,imshow(result)

%% Gradient
clc,clear all,close all

% img = im2double(imread('test.png'));

img = load('events.csv');
[BW,threshOut,Gv,Gh] = edge(img,'Sobel');

theta = atan2(Gh,Gv);

vertical = abs(Gv);

vertical(vertical < 0.2) = 0;

figure,imshow(img),title('Original'),drawnow
figure,imshow(vertical),title('Result'),drawnow

%% Gabor
clc,clear all,close all

% img = im2double(imread('42049.bmp'));
img = load('events.csv');


wavelength = 4;
orientation = 0;
[mag,phase] = imgaborfilt(img,wavelength,orientation);

% mag(mag < 1) = 0;

figure,imshow(img),title('Original'),drawnow
figure,imshow(mag),title('Result'),drawnow

%% Ellisse
clc,clear all,close all


img = im2double(imread('42049.bmp'));
% theta = linspace(0, 2*pi, 200);
% l = 1;
% lambda_u = 2;
% lambda_v = 2;
% r1 = l/sqrt(lambda_u);
% r2 = l/sqrt(lambda_v);
% u = r1 * cos(theta); 
% v = r2 * sin(theta);
% plot(u, v, '-');

% l = 1;
% lambda_u = 2;
% lambda_v = 2;
% r1 = sqrt(l/lambda_u);
% r2 = sqrt(l/lambda_v);

u1 = 0;
u2 = 0;
sigma1 = 2;
sigma2 = 1;
r1 = sigma1*sqrt(2);
r2 = sigma2*sqrt(2);

[X_1,Y_1,Z] = ellipsoid(u1,u2,0,r1,r2,0);

u1 = 0;
u2 = 0;
sigma1 = 1;
sigma2 = 0.5;
r1 = sigma1*sqrt(2);
r2 = sigma2*sqrt(2);

[X_2,Y_2,Z] = ellipsoid(u1,u2,0,r1,r2,0);

% G_x = X_1-X_2;
% G_y = Y_1-X_2;

% result = conv2(G,img,'same');

%% Multivariate Gaussian

clc,clear all,close all

% img = im2double(imread('42049.bmp'));

img = load('events.csv');
mu = [0 0];

% Horizontal
% Sigma = [0.8 0; 0 0.3];

% Vertical
Sigma = [0.4 0; 0 0.8];

x1 = -3:0.2:3;
x2 = -3:0.2:3;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
y = mvnpdf(X,mu,Sigma);
y_1 = reshape(y,length(x2),length(x1));

figure,surf(x1,x2,y_1),title('Y1'),drawnow;

mu = [0 0];

% Horizontal
% Sigma = [1 0; 0 0.2];

% Vertical
Sigma = [0.3 0; 0 0.8];

x1 = -3:0.2:3;
x2 = -3:0.2:3;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
y = mvnpdf(X,mu,Sigma);
y_2 = reshape(y,length(x2),length(x1));

figure,surf(x1,x2,y_2),title('Y2'),drawnow;

G = y_2-y_1;

figure,surf(x1,x2,G),title('G'),drawnow;

result = conv2(img,G,'same');

%result(result < 0.6) = 0;

figure,imshow(img);
figure,imshow(result),title('Result'),drawnow;


%% Simple Filter
clc,clear all,close all

img = im2double(imread('42049.bmp'));

ver_edge_filter = ones(2,2);
ver_edge_filter(:,2) = -1;

result = imfilter(img,ver_edge_filter);

%result(result < 0.5) = 0;

figure,imshow(img),title('Original'),drawnow
figure,imshow(result),title('Result'),drawnow

%% Test

clc,clear all,close all

img = im2double(imread('42049.bmp'));
% r = img(:,:,1);
% g = img(:,:,2);
% b = img(:,:,3);
% img = (r+b+g)/3;

orienslist = 0:22.5:337.5;

mu = [0 0];

% Vertical
Sigma = [0.4 0; 0 0.8];

x1 = -3:0.2:3;
x2 = -3:0.2:3;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
y = mvnpdf(X,mu,Sigma);
y_1 = reshape(y,length(x2),length(x1));

mu = [0 0];

% Vertical
Sigma = [0.3 0; 0 0.8];

x1 = -3:0.2:3;
x2 = -3:0.2:3;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
y = mvnpdf(X,mu,Sigma);
y_2 = reshape(y,length(x2),length(x1));

G = y_2-y_1;

% result = conv2(img,imrotate(G,orienslist(1)),'same');
% 
% result(result < 0.2) = 0;
% imshow(result)

for i=1:numel(orienslist)
    temp = conv2(img,imrotate(G,orienslist(i)),'same');
%     temp(temp < 0.02) = 0;
    temp(temp < 0.08) = 0;
    result(:,:,i) = temp;
end

[corfresponse, oriensMatrix] = calc_viewimage(result,1:numel(orienslist), orienslist*pi/180);
imshow(oriensMatrix)