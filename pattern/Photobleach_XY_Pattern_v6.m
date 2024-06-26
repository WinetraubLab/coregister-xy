                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    % Run this demo to use Thorlabs system to photobleach a pattern 

% Before running this script, make sure myOCT folder is in path for example
% by running: addpath(genpath('L:\Emilie\myOCT-master'))
% for online: addpath(genpath('myOCT-master'))


%% Inputs

% When set to true the stage will not move and we will not
% photobleach. Use "true" when you would like to see the output without
% physcaily running the test.
skipHardware = true;

% Photobleach pattern configuration
octProbePath = yOCTGetProbeIniPath('40x','OCTP900'); % Select lens magnification

% Pattern to photobleach. System will photobleach n lines from 
% (x_start(i), y_start(i)) to (x_end(i), y_end(i)) at height z
x_start_mm = []; x_end_mm=[]; y_start_mm=[]; y_end_mm=[]; z_mm=[];
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = create1BlockPattern(270e-3, 0, false);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = create1BlockPattern(-270e-3, 0, false);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = create1BlockPattern(0, 270e-3, true);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = create1BlockPattern(0, -270e-3, true);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];

line_exposure_sec_mm = 5; % sec/mm. For gel use 0.5
exposure_sec_mm = line_exposure_sec_mm*ones(size(x_start_mm));

% Guidelines - these help to find the photobleach area
if true % set to true to have guidelines
    x_start_mm = [x_start_mm [ 1.0  0    -0.3   -0.3   -0.3   0.34]];
    x_end_mm =   [x_end_mm   [ 0.6  0    -0.3   -0.24  -0.3   0.22]];
    y_start_mm = [y_start_mm [ 0   -1.5  0.2  0.3      -0.34  0.3]];
    y_end_mm =   [y_end_mm   [ 0   -0.6  0.3   0.3     -0.22  0.3]];
    if ~isempty(z_mm)
        L_depth_mm = min(z_mm)-0.5*mean(diff(z_mm));
    else
        L_depth_mm = 0;
    end
    z_mm = [z_mm ones(1,6)*L_depth_mm];
    guide_line_exposure_sec_mm = line_exposure_sec_mm*5; % sec/mm
    exposure_sec_mm = [exposure_sec_mm guide_line_exposure_sec_mm*ones(1,6)];
end

% Photobleach configurations
nPasses = 2; % Keep as low as possible. If galvo gets stuck, increase number

%% Plot Figure With Pattern, color is by z-depth

% Organize colors acordign to height
uz_mm = unique(z_mm);
colors = num2cell(jet(length(uz_mm)),2);

% Figure out what is the FOV in order to draw it
json = yOCTReadProbeIniToStruct(octProbePath);

% Plot all
figure(22)
for subplotI = 1:2
    subplot(1,2,subplotI);
    for plotI = 1:length(x_start_mm)
        if (exposure_sec_mm(plotI) == line_exposure_sec_mm)
            c = colors{uz_mm==z_mm(plotI)}; % This is a line, use regular color
            lw = 0.5;
        else
            %c = [0, 0, 0]; % This is a guide line, use black
            c = colors{uz_mm==z_mm(plotI)}; % This is a line, use regular color
            lw = 1.5;
        end
        plot([x_start_mm(plotI) x_end_mm(plotI)],[y_start_mm(plotI) y_end_mm(plotI)],'Color',c,'LineWidth',lw);
        if (plotI == 1)
           hold on;
        end
    end
    plot(json.RangeMaxX/2*[-1 1 1 -1 -1],json.RangeMaxY/2*[-1 -1 1 1 -1],'--k')
    hold off;
    axis equal;
    axis ij;
    grid on;
    xlabel('x[mm]');
    ylabel('y[mm]');
    
    if subplotI == 1
        xlim([min([x_start_mm x_end_mm])-100e-3, max([x_start_mm x_end_mm])+100e-3])
        ylim([min([y_start_mm y_end_mm])-100e-3, max([y_start_mm y_end_mm])+100e-3])
        title('Overview');
    else
        xlim(mean([x_start_mm1 x_end_mm1])+2*std([x_start_mm1 x_end_mm1])*[-1 1])
        ylim(mean([y_start_mm1 y_end_mm1])+2*std([y_start_mm1 y_end_mm1])*[-1 1])
        title('Pattern Zoom In');
    end
end
pause(0.1);

%% Plot Pattern
% Perform photobleach
for i=1:length(uz_mm)
    ii = find(z_mm==uz_mm(i));
    fprintf('Photobleaching Depth (%d): %.3f mm\n',i,z_mm(ii(1)));
    yOCTPhotobleachTile(...
        [x_start_mm(ii); y_start_mm(ii)], ...
        [x_end_mm(ii); y_end_mm(ii)], ...
        'octProbePath',octProbePath, ...
        'exposure',exposure_sec_mm(ii(1)),'nPasses',nPasses,...
        'z',z_mm(ii(1)),'skipHardware',skipHardware ...
        ...,'laserToggleMethod','LaserPowerSwitch' ... Comment this section out to use laser
        );  
    pause(0.5);
end

disp('Done Patterning')

function [x_start_mm, x_end_mm, y_start_mm, y_end_mm, z_mm] = create1BlockPattern(offset_x_mm, offset_y_mm, isflip)
xyscale = 240e-3; %mm
xp_start_mm = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,  2]/3*xyscale + 10e-3 -xyscale/2;
xp_end_mm   = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2,  3]/3*xyscale - 10e-3 -xyscale/2;
yp_start_mm = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,  3]/3*xyscale         -xyscale/2;
yp_end_mm   = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,  3]/3*xyscale        -xyscale/2;
z_mm       = [0     1     2    5      4     3     6     7     8     11   10     9]*50e-3;

if ~isflip
    x_start_mm = xp_start_mm + offset_x_mm;
    x_end_mm =   xp_end_mm   + offset_x_mm;
    y_start_mm = yp_start_mm + offset_y_mm;
    y_end_mm =   yp_end_mm   + offset_y_mm;
else
    x_start_mm = yp_start_mm + offset_x_mm;
    x_end_mm =   yp_end_mm   + offset_x_mm;
    y_start_mm = xp_start_mm + offset_y_mm;
    y_end_mm =   xp_end_mm   + offset_y_mm;
end 
end
