% This script will generate a 3D volume of how the XY pattern should appear
% in the tissue
clearvars
% Load the pattern
[x_start_mm, x_end_mm, y_start_mm, y_end_mm, z_start_end_mm] = ...
    generateXYPattern(false);

%% Script Inputs

% Define the bounding box to generate 3D simulation for
x_grid_mm = -0.40:2e-3:0.40;
y_grid_mm = -0.40:2e-3:0.40;
z_grid_mm =  0.00:2e-3:0.40;

% Physical parameters 
NA = 0.35; % Match NA to observed photobleach pattern. For 40x use 0.35 (though lens NA is 0.8)
lambda_mm = 900e-9*1e3; % Wavelength in m
n = 1.4; % Medium index of refraction
photobleach_intensity = 40/(diff(y_grid_mm(1:2))/1e-3); % Can be any number >0

% Plot OCT volume on top?
oct_scan_mm = [-0.25 0.25]; % OCT x-y scan size
% If you don't want to plot oct_scan_mm, un-comment:
% oct_scan_mm = nan;

%% Configurable Parameters
% Gausian base waist
w0_mm = 1/pi*lambda_mm*n/NA;
zR_mm = pi*w0_mm^2/lambda_mm;

%% Create a gread
[xx_mm, yy_mm] = meshgrid(x_grid_mm,y_grid_mm);
pixel_size_mm = diff(x_grid_mm(1:2));
disp(pixel_size_mm);

% Loop for each plane in z
isFirstLoop=false;
for z=fliplr(z_grid_mm) % Start at the bottom for ImageJ orientation
    c_all = ones(size(xx_mm));

    % Loop over all lines
    for lineI=1:length(x_start_mm)

        if y_start_mm(lineI) == y_end_mm(lineI)
            % Horizontal line  
            yI = abs(yy_mm-y_start_mm(lineI)) < pixel_size_mm;
            xI = xx_mm >= min(x_start_mm(lineI),x_end_mm(lineI)) & ...
                 xx_mm <= max(x_start_mm(lineI),x_end_mm(lineI));

        elseif x_start_mm(lineI) == x_end_mm(lineI)
            % Vertical line
            xI = abs(xx_mm-x_start_mm(lineI))<pixel_size_mm;
            yI = yy_mm >= min(y_start_mm(lineI),y_end_mm(lineI)) & ...
                 yy_mm <= max(y_start_mm(lineI),y_end_mm(lineI));
        end


        % Gausian waist at the depth we are
        wz_mm = w0_mm*sqrt(1+( ...
            (z_start_end_mm(lineI)-z) /zR_mm)^2);

        % Add the line
        c=createPhotobleachArea(yI&xI,photobleach_intensity,wz_mm/pixel_size_mm);
        c_all = c_all .* c;
    end

    % Add center point
    c=createPhotobleachArea(...
        xx_mm==min(abs(x_grid_mm)) & yy_mm==min(abs(y_grid_mm)),...
        photobleach_intensity,4);
    c_all = c_all .* c;

    % Add the OCT scan
    if all(~isnan(oct_scan_mm))
        c_all = addOCTScanRectangle(c_all, xx_mm, yy_mm, oct_scan_mm,pixel_size_mm);
    end
    
    c_all = rgb2gray(insertText(c_all, [size(c_all,2)/2 0], ...
        sprintf('z=%.1fum', z*1e3), ...
        'FontSize', 20, 'AnchorPoint', 'CenterTop'));
    
    % Convert to uint8 if the image is in double format
    if isa(c_all, 'double')
        c_all_uint8 = im2uint8(c_all); % Convert to uint8
    else
        c_all_uint8 = uint8(c_all); % Ensure it's uint8
    end

    % Simulation output
    output_filepath = 'out_xy.tif'
    output_tiff_file = Tiff(output_filepath, 'a');
    
    % Set TIFF tags
    tagstruct1.ImageLength = size(c_all_uint8, 1);
    tagstruct1.ImageWidth = size(c_all_uint8, 2);
    tagstruct1.Photometric = Tiff.Photometric.MinIsBlack; % Grayscale setting
    tagstruct1.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct1.RowsPerStrip = 16;
    tagstruct1.BitsPerSample = 8; % 8 bits per sample
    tagstruct1.SamplesPerPixel = 1; % Grayscale image
    tagstruct1.ResolutionUnit = Tiff.ResolutionUnit.Centimeter;
    tagstruct1.XResolution = 10/pixel_size_mm;
    tagstruct1.YResolution = 10/pixel_size_mm;
    % Write the tags and image to the TIFF file
    output_tiff_file.setTag(tagstruct1); % Set TIFF metadata
    output_tiff_file.write(c_all_uint8); % Write uint8 image data to file
    output_tiff_file.close(); % Close the TIFF file
    
    % Present to user
    figure(27);
    imagesc(x_grid_mm,y_grid_mm,c_all)
    axis equal
    caxis([0,1])
    colormap gray
    pause(0.1);
end

%% Finally, print to screen a description of how this pattern was created
describePattern(x_start_mm, x_end_mm, y_start_mm, y_end_mm, z_start_end_mm)

function c_all = addOCTScanRectangle(c_all, xx_mm, yy_mm, oct_scan_mm,pixel_size_mm)
r = zeros(size(c_all));

% Left and right borders
r(yy_mm >= oct_scan_mm(1) & yy_mm <= oct_scan_mm(2) & abs(xx_mm-oct_scan_mm(1)) < pixel_size_mm) = 1;
r(yy_mm >= oct_scan_mm(1) & yy_mm <= oct_scan_mm(2) & abs(xx_mm-oct_scan_mm(2)) < pixel_size_mm) = 1;

% Top and bottom borders 
r(xx_mm >= oct_scan_mm(1) & xx_mm <= oct_scan_mm(2) & abs(yy_mm-oct_scan_mm(1)) < pixel_size_mm) = 1;
r(xx_mm >= oct_scan_mm(1) & xx_mm <= oct_scan_mm(2) & abs(yy_mm-oct_scan_mm(2)) < pixel_size_mm) = 1;

c_all(r==1) = 0;

end

function describePattern(x_start_mm, x_end_mm, y_start_mm, y_end_mm, z_mm)
z = unique(z_mm);

fprintf('Draw Lines from (x,y) to (x,y) at the following depths. Units: mm\n');
for i=1:length(z)
    fprintf('Depth: %.3f\n',z(i))
    ii = find(z_mm == z(i));
    for j=ii
        fprintf('   (%+.3f,%+.3f) -> (%+.3f,%+.3f)\n',...
            x_start_mm(j),y_start_mm(j),...
            x_end_mm(j),y_end_mm(j)...
            );
    end
end
end

function c=createPhotobleachArea(whereToPhotobleach,photobleach_intensity,w)
% whereToPhotobleach - bolean mask = 1 if photobleach center / seed, 0
%   otherwise.
% w - gausian waist size in pixels

% Create canvas 
c = ones(size(whereToPhotobleach))*(photobleach_intensity+1); 

% Create the photobleach area
c(whereToPhotobleach) = 0; 

% Gausian filt
c = imgaussfilt(c, w/sqrt(2))-photobleach_intensity;

% Clean uo
c(c<0)=0;

end