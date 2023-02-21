clc,clear all;

% File with all image names
list_images = load('list_images.txt');

res_comp = zeros(size(list_images,1),6);

res_comp(:,1) = list_images;

% Simulator thresholds
th = {'010','015','020','030','040'};

for ind_th=1:size(th,2)
    
    classified = {};

    % For each image do a comparison with the ground truth
    for i=1:size(list_images,1)
        
        image_name = list_images(i);

        FG = deg2rad(load(strcat('./FG/',string(image_name),'.csv')));
        X = load(strcat('./',string(th{ind_th}),'/','X_',string(image_name),'.csv'));
        Y = load(strcat('./',string(th{ind_th}),'/','Y_',string(image_name),'.csv'));
        frame = load(strcat('./',string(th{ind_th}),'/','frame_',string(image_name),'.csv'));

        % Put 200 (i.e. background) in the result where there nothing in the events frame
        result = atan2(Y,X);
        result(frame==0) = 200;
        total_class = 0;
        total_noclass = 0;
        pixel = 1;
        differenze = {};

        for row=1:size(result,1)
            for col=1:size(result,2)
                
                % If there is something in the result and in the ground
                % truth, make an angle difference
                if ((FG(row,col)~=deg2rad(200)) && (result(row,col)~=200))
                    differenze{1,pixel} = angdiff(FG(row,col),result(row,col));                
                    pixel = pixel + 1;
                    total_class = total_class +1;
                else
                    total_noclass = total_noclass + 1;
                end
            end
        end
        classified{1,i} = total_class;
        differenze = cell2mat(differenze);
        media = rad2deg(mean(differenze));
        varianza = var(differenze);

        res_comp(i,ind_th+1) = media;

    end

end

writematrix(res_comp,'final.csv');