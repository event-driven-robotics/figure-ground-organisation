
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% run setup before %%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% path
BW_flag=false;

% name_str={'Circles', 'tv', 'heart', 'square_sasso', 'cilinder_cup_bottle', 'key_mouse_flip','calib_circles'};  
name_str={'12003', '12074', '22090','28075','35070','35091','41004','43070','112082','113016','135069', '156079'}; 

% name_str={'28075'}; 

for  name_str = name_str
    im_path=strcat(name_str{1},'.jpg');
    
    %figure-ground model
    [e, o, g] = runProtoGroup(im_path, BW_flag);
    
    
    %INPUT
    ha = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    % for grayscale and color images on same axis
    cmap = [gray(64); jet(64)]; % new colormap
    colormap(cmap)
    axes(ha(1)); imagesc(imread(im_path));
    set(ha(1),'XTick',[]);
    set(ha(1),'YTick',[]);
    set(gcf,'Color','w');
    
    
%     EDGE
    ha1 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    % for grayscale and color images on same axis
    cmap = [gray(64); jet(64)]; % new colormap
    colormap(cmap)
    axes(ha1(1)); imagesc(1-e-1/64); caxis([0 2]);
    set(ha1(1),'XTick',[]);
    set(ha1(1),'YTick',[]);
    set(gcf,'Color','w');
    
    
    %ORIENTATION MATRIX
    ha2 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    % for grayscale and color images on same axis
    cmap = [gray(64); jet(64)]; % new colormap
    colormap(cmap)
    axes(ha2(1)); imagesc(o);
    set(ha2(1),'XTick',[]);
    set(ha2(1),'YTick',[]);
    set(gcf,'Color','w');
    
    
    %GROUPING
    ha3 = tight_subplot(1,1,0.01,[0.01 0.01],[0.01 0.01]);
    axes(ha3(1)); imagesc(g); caxis([0 1]);
    colormap(jet(64));
    colorbar;
    set(ha3(1),'XTick',[]);
    set(ha3(1),'YTick',[]);
    saving_path=fullfile('output/groupjetmap/',sprintf('%s.png',im_path(1:end-4))),'png';
    saveas(ha3,saving_path);
end
