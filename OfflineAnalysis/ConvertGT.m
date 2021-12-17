% Convert the ground truth by looking to the line that fits pixel with
% value 1 inside a window 5x5 centered on a pixel of value 1. Depending on the slope of this line
% consider the pixel that are on its perpendicular line. For this pixel
% ,see whic have the value -1; this will be the pixel used to compute the
% angle for the central pixel of the window

clc,clear all,close all;

MAXDIM = 21;

imagefiles = dir('./color/*.fg');

% For each image
for ii=1:2:size(imagefiles,1)

        names = {imagefiles(ii).name,imagefiles(ii+1).name};
        image_name = string(names(1,randi([1 2])));
    
        FG = readbin(strcat('/home/simone/Scaricati/fgcode/color/',image_name));
        FG = FG{1,1};

        new_FG = zeros(size(FG));
        new_FG(:,:) = 200;

        ms = [-1/3, 1/3, 3, -3];
        count = 0;

        [row,col] = find(FG==1);
        
        % For each pixel 1 in the image
        for i=1:size(row,1)

            FLAG = 0;
            FLAGMAXDIM = 0;
            FLAGBORDER = 0;
            up = 0;
            down = 0;

            dim_window = 5;
            xFit = linspace(1, dim_window, dim_window);
            shift_window = floor(dim_window/2);

            while (FLAG == 0)
                
                % If the dimension of the window is less than 21x21
                if (FLAGMAXDIM == 0)
                    
                    % Check if i(x,y) is a border pixel
                    if (row(i)-shift_window<1) || (row(i)+shift_window>size(FG,1)) || (col(i)-shift_window<1) || (col(i)+shift_window>size(FG,2))
                        FLAGBORDER = 1;
                        break;
                    end

                    window = FG(row(i)-shift_window:row(i)+shift_window,col(i)-shift_window:col(i)+shift_window);

                    [y_ones,x_ones] = find(window==1);

                    linearCoefficients = polyfit(x_ones,y_ones,1);
                    yFit = polyval(linearCoefficients, xFit);

                    temp_x = 0;

                    m = linearCoefficients(1);
                    slope_perp = -1/m;
                    
                    % New window of 3x3 centered on i(x,y)
                    new_window = FG(row(i)-1:row(i)+1,col(i)-1:col(i)+1);
                    px = [0 0];

                else

                    [new_y_ones,new_x_ones] = find(window==1);

                end
                
                % -1/3<m<1/3
                if((m>=ms(1)) && (m<ms(2)))
                    
                    if (FLAGMAXDIM == 1)
                        
                        for index=1:size(new_y_ones,1)

                            if ((new_y_ones(index)+1<1) || (new_y_ones(index)+1>MAXDIM) || (new_x_ones(index)<1) || (new_x_ones(index)>MAXDIM))
                                down = down +1;
                            elseif ((new_y_ones(index)-1<1) || (new_y_ones(index)-1>MAXDIM) || (new_x_ones(index)<1) || (new_x_ones(index)>MAXDIM))
                                up = up +1;
                            else

                                if(window(new_y_ones(index)+1,new_x_ones(index)) == -1)
                                    down = down +1;
                                elseif(window(new_y_ones(index)-1,new_x_ones(index)) == -1)
                                    up = up +1;
                                end

                            end
                        end

                        if(up>=down)
                            temp_x = 2+sign(m);
                            FLAG = 1;
                        else
                            temp_x = 2-sign(m);
                            FLAG = 1;
                        end

                    else    

                        px = [new_window(1,2), new_window(3,2)];
                        a = find(px==-1);

                        if (size(a,2)>1)
                            dim_window = dim_window + 2;
                            if (dim_window > MAXDIM)
                                FLAGMAXDIM = 1;
                            else
                                shift_window = floor(dim_window/2);
                            end
                            
                         else
                            if (a==1)
                                up = 1;
                                FLAG = 1;
                                temp_x = 2+sign(m);

                            elseif (a==2)
                                down = 1;
                                FLAG = 1;
                                temp_x = 2-sign(m);
                            else
                                dim_window = dim_window + 2;
                                if (dim_window > MAXDIM)
                                    FLAGMAXDIM = 1;
                                else
                                    shift_window = floor(dim_window/2);
                                end

                            end
                        end
                    end
                % 1/3<m<3
                elseif ((m>=ms(2)) && (m<ms(3)))

                    if (FLAGMAXDIM == 1)

                        for index=1:size(new_y_ones,1)

                            if ((new_y_ones(index)+1<1) || (new_y_ones(index)+1>MAXDIM) || (new_x_ones(index)-1<1) || (new_x_ones(index)-1>MAXDIM))
                                down = down +1;
                            elseif ((new_y_ones(index)-1<1) || (new_y_ones(index)-1>MAXDIM) || (new_x_ones(index)+1<1) || (new_x_ones(index)+1>MAXDIM))
                                up = up +1;
                            else

                                if(window(new_y_ones(index)+1,new_x_ones(index)-1) == -1)
                                    down = down +1;
                                elseif(window(new_y_ones(index)-1,new_x_ones(index)+1) == -1)
                                    up = up +1;
                                end

                            end
                        end

                        if(up>=down)
                            temp_x = 3;
                            FLAG = 1;
                        else
                            temp_x = 1;
                            FLAG = 1;
                        end
                    else

                        px = [new_window(3,1),new_window(1,3)];
                        a = find(px==-1);

                        if (size(a,2)>1)
                            dim_window = dim_window + 2;
                            if (dim_window > MAXDIM)
                                FLAGMAXDIM = 1;
                            else
                                shift_window = floor(dim_window/2);
                            end
                        else
                            if (a==1)
                                FLAG = 1;
                                temp_x = 1;

                            elseif (a==2)
                                FLAG = 1;
                                temp_x = 3;
                            else
                                dim_window = dim_window + 2;
                                if (dim_window > MAXDIM)
                                    FLAGMAXDIM = 1;
                                else
                                    shift_window = floor(dim_window/2);
                                end
                                
                            end
                         end
                    end
                 % m>3 or m<-3
                elseif ((m>=ms(3)) || (m<=ms(4)))
                    
                    if (FLAGMAXDIM == 1)

                        for index=1:size(new_y_ones,1)

                            if ((new_y_ones(index)<1) || (new_y_ones(index)>MAXDIM) || (new_x_ones(index)-1<1) || (new_x_ones(index)-1>MAXDIM))
                                down = down +1;
                            elseif ((new_y_ones(index)<1) || (new_y_ones(index)>MAXDIM) || (new_x_ones(index)+1<1) || (new_x_ones(index)+1>MAXDIM))
                                up = up +1;
                            else

                                if(window(new_y_ones(index),new_x_ones(index)-1) == -1)
                                    down = down +1;
                                elseif(window(new_y_ones(index),new_x_ones(index)+1) == -1)
                                    up = up +1;
                                end
                            end
                        end

                        if(up>= down)
                            temp_x = 3;
                            FLAG = 1;
                        else
                            temp_x = 1;
                            FLAG = 1;
                        end
                    else 

                        px = [new_window(2,1), new_window(2,3)];
                        a = find(px==-1);

                        if (size(a)>1)
                            dim_window = dim_window + 2;
                            if (dim_window > MAXDIM)
                                FLAGMAXDIM = 1;
                            else
                                shift_window = floor(dim_window/2);
                            end
                        end

                        if (a==1)
                            FLAG = 1;
                            temp_x = 1;
                        elseif (a==2)
                            FLAG = 1;
                            temp_x = 3;
                        else
                           dim_window = dim_window + 2;
                           if (dim_window > MAXDIM)
                                FLAGMAXDIM = 1;
                           else
                               shift_window = floor(dim_window/2);
                           end
                        end

                    end
                 
                % -3<m<-1/3
                elseif ((m>ms(4)) && (m<ms(1)))

                    if (FLAGMAXDIM == 1)

                        for index=1:size(new_y_ones,1)

                            if ((new_y_ones(index)-1<1) || (new_y_ones(index)-1>MAXDIM) || (new_x_ones(index)-1<1) || (new_x_ones(index)-1>MAXDIM))
                                up = up +1;
                            elseif ((new_y_ones(index)+1<1) || (new_y_ones(index)+1>MAXDIM) || (new_x_ones(index)+1<1) || (new_x_ones(index)+1>MAXDIM))
                                down = down +1;
                            else

                                if(window(new_y_ones(index)-1,new_x_ones(index)-1) == -1)
                                    up = up +1;
                                elseif(window(new_y_ones(index)+1,new_x_ones(index)+1) == -1)
                                    down = down +1;
                                end
                            end
                        end

                        if(up>= down)
                            temp_x = 1;
                            FLAG = 1;
                        else
                            temp_x = 3;
                            FLAG = 1;
                        end
                    else 

                        px = [new_window(1,1), new_window(3,3)];
                        a = find(px==-1);

                        if (size(a,2)>1)
                            dim_window = dim_window + 2;
                            if (dim_window > MAXDIM)
                                FLAGMAXDIM = 1;
                            else
                                shift_window = floor(dim_window/2);
                            end
                        else
                            if (a==1)
                                FLAG = 1;
                                temp_x = 1;
                            elseif (a==2)
                                FLAG = 1;
                                temp_x = 3;
                            else
                                dim_window = dim_window + 2;
                                if (dim_window > MAXDIM)
                                    FLAGMAXDIM = 1;
                                else
                                    shift_window = floor(dim_window/2);
                                end
                            end
                        end

                    end

                else
                    
                    disp('else');
                    
                end

                    
            end

            if (FLAGBORDER == 0)
                
                half_dim_window = ceil(3/2);
                y_perp = slope_perp * (temp_x-half_dim_window);
                
                if (~isinf(slope_perp))

                    angle = rad2deg(atan2(y_perp,half_dim_window-temp_x))/22.5;
                    new_FG(row(i),col(i)) = round(angle)*22.5;
                else
                    
                    if(up>=down)
                        new_FG(row(i),col(i)) = -90;
                    elseif(down>up)
                        new_FG(row(i),col(i)) = 90;
                    end
                    
                end

            end

        end

        stringa = split(image_name,'.');
        stringa = split(stringa{1,1},'-');
        writematrix(new_FG,strcat(stringa{1,1},'.csv'));
    
    

end