% This script will generate a 3D volume of how the XY pattern should appear
% in the tissue

% Load the pattern
[x_start_mm, x_end_mm, y_start_mm, y_end_mm, z_start_end_mm] = ...
    generateXYPattern(false);

%% Script Inputs

% Define the bounding box to generate 3D simulation for
x_grid_mm = -0.25:1e-3:0.25;
y_grid_mm = -0.25:1e-3:0.25;
z_grid_mm =  0.00:5e-3:0.20;

% Physical parameters 
NA = 0.8; % 10x use 0.2, for 40x use 0.8
lambda_mm = 900e-9*1e3; % Wavelength in m
n = 1.4; % Medium index of refraction
photobleach_intensity = 40; % Can be any number >0

% Simulation output
output_tiff_file = 'out.tiff';

%% Configurable Parameters
% Gausian base waist
w0_mm = 1/pi*lambda_mm*n/NA;
zR_mm = pi*w0_mm^2/lambda_mm;

%% Create a gread
[xx_mm, yy_mm] = meshgrid(x_grid_mm,y_grid_mm);

% Loop for each plane in z
isFirstLoop=true;
for z=z_grid_mm
    c_all = ones(size(xx_mm));

    % Loop over all line
    for lineI=1:length(x_start_mm)
        c = ones(size(xx_mm))*(photobleach_intensity+1); % Create canvace 

        if y_start_mm(lineI) == y_end_mm(lineI)
            % Horizontal line  
            yI = abs(yy_mm-y_start_mm(lineI))<pixel_size_um/1e3;
            xI = xx_mm >= min(x_start_mm(lineI),x_end_mm(lineI)) & ...
                 xx_mm <= max(x_start_mm(lineI),x_end_mm(lineI));

        elseif x_start_mm(lineI) == x_end_mm(lineI)
            % Vertical line
            xI = abs(xx_mm-x_start_mm(lineI))<pixel_size_um/1e3;
            yI = yy_mm >= min(y_start_mm(lineI),y_end_mm(lineI)) & ...
                 yy_mm <= max(y_start_mm(lineI),y_end_mm(lineI));
        end

        % Create the base line
        c(yI&xI) = 0; 

        % Gausian waist at the depth we are
        wz_mm = w0_mm*sqrt(1+( ...
            (z_start_end_mm(lineI)-z) /zR_mm)^2);

        % Gausian filt
        c = imgaussfilt(c, wz_mm/sqrt(2)*1e3/pixel_size_um)-photobleach_intensity;
        c(c<0)=0;

        c_all = c_all .* c;
    end
    
    if isFirstLoop
        imwrite(c_all,output_tiff_file,...
            'Description',' ' ... Description contains min & max values
            );
        isFirstLoop = false;
    else
        imwrite(c_all,output_tiff_file,'writeMode','append');     
    end
    
    figure(27);
    imagesc(x_grid_mm,y_grid_mm,c_all)
    axis equal
    caxis([0,1])
    colormap gray
    pause(0.1);
end
