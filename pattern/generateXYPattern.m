function [...
    x_start_mm, x_end_mm, ...
    y_start_mm, y_end_mm, ...
    z_mm] = generateXYPattern(verbose, patternCenter_mm)
% This function generates the instructions (lines) for XY lines
% The lines are from (x_start_mm(i), y_start_mm(i)) (to x_end_mm(i), y_end_mm(i)) 
% at depth z
% INPUT: 
%   verbose (default: false). If set to true, function will output
%   line information as well as image.
%   patternCenter_mm - list of pattern centers [x0,y0,z0; x1,y1,z1; ...].
%       default: [0,0,0]

%% Inputs check
if ~exist('verbose','var')
    verbose = false;
end

if ~exist('patternCenter_mm','var')
    patternCenter_mm = [0,0,0];
end

%% Create pattern
x_start_mm = []; x_end_mm=[]; y_start_mm=[]; y_end_mm=[]; z_mm=[];
for i=1:size(patternCenter_mm,1)
    [x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = ...
        CreateFOVPattern( ...
            patternCenter_mm(i,1), ... x
            patternCenter_mm(i,2)  ... y
        );
    x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];
end

%% Sort
[~,i] = sort(z_mm);
x_start_mm = x_start_mm(i);
x_end_mm = x_end_mm(i);
y_start_mm = y_start_mm(i);
y_end_mm = y_end_mm(i);
z_mm = z_mm(i);

%% Plot
if verbose
    % Organize colors acordign to height
    uz_mm = unique(z_mm);
    colors = num2cell(jet(length(uz_mm)),2);

    % Plot all
    figure(22)
    for subplotI = 1:2
        subplot(1,2,subplotI);
        for plotI = 1:length(x_start_mm)
            c = colors{uz_mm==z_mm(plotI)};
            lw = 0.5;
            plot([x_start_mm(plotI) x_end_mm(plotI)],[y_start_mm(plotI) y_end_mm(plotI)],'Color',c,'LineWidth',lw);
            if (plotI == 1)
               hold on;
            end
        end
        lens_fov = 0.5; %mm, lens FOV
        plot(lens_fov/2*[-1 1 1 -1 -1],lens_fov/2*[-1 -1 1 1 -1],'--k')
        hold off;
        axis equal;
        axis ij;
        grid on;
        xlabel('x[mm]');
        ylabel('y[mm]');

        if subplotI == 1
            xlim([min([x_start_mm x_end_mm])-100e-3, max([x_start_mm x_end_mm])+100e-3])
            ylim([min([y_start_mm y_end_mm])-100e-3, max([y_start_mm y_end_mm])+100e-3])
            title(sprintf('Overview\n(Blue z=%.0f\\mum, Red z=%.0f\\mum)',min(z_mm)*1e3,max(z_mm)*1e3));
        else
            xlim(mean([x_start_mm1 x_end_mm1])+2*std([x_start_mm1 x_end_mm1])*[-1 1])
            ylim(mean([y_start_mm1 y_end_mm1])+2*std([y_start_mm1 y_end_mm1])*[-1 1])
            title('Pattern Zoom In');
        end
    end
    pause(0.1);
end
end

function [x_start_mm, x_end_mm, y_start_mm, y_end_mm, z_mm] = CreateFOVPattern(centerX, centerY)
%% Bulk of the pattern
% Pattern to photobleach. System will photobleach n lines from 
x_start_mm = []; x_end_mm=[]; y_start_mm=[]; y_end_mm=[]; z_mm=[];

% Right
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = Create1BlockPattern(185e-3, 0, false, false, centerX, centerY);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];

% Left
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = Create1BlockPattern(-185e-3, 0, false, false, centerX, centerY);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];

% Down
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = Create1BlockPattern(0, 185e-3, true, false, centerX, centerY);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];

% Top
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = Create1BlockPattern(0, -185e-3, true, false, centerX, centerY);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];

%% Alignment markers (L shape)
nGridLines = 0;

% Small L
if true
    x_start_mm = [x_start_mm ([-0.30 -0.30 -0.30 0.10 0.22] + centerX)];
    x_end_mm =   [x_end_mm   ([-0.30 -0.30 -0.20 0.17 0.30] + centerX)];
    y_start_mm = [y_start_mm ([-0.20  0.10  0.30 0.30 0.30] + centerY)];
    y_end_mm =   [y_end_mm   ([-0.30  0.30  0.30 0.30 0.30] + centerY)];
    nGridLines = nGridLines + 5;
end 

% Big L
if false
    x_start_mm = [x_start_mm [1.00   0.00]];
    x_end_mm =   [x_end_mm   [0.40   0.00]];
    y_start_mm = [y_start_mm [0.00  -1.50]];
    y_end_mm =   [y_end_mm   [0.00  -0.40]];
    nGridLines = nGridLines + 2;
end


%% Markers depths    
% Set Z
if ~isempty(z_mm)
    L_depth_mm = min(z_mm) + 3*mean(diff(unique(z_mm)));
else
    L_depth_mm = 0;
end
z_mm = [z_mm ones(1,nGridLines)*L_depth_mm];

end

%% Small Pattern
function [x_start_mm, x_end_mm, y_start_mm, y_end_mm, z_mm] = Create1BlockPattern( ...
    offset_x_mm, offset_y_mm, is_flip_xy, is_flip_z, centerX, centerY)
scale1 = 150e-3; %mm
scale2 = 180e-3;
xp_start_mm = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,  2]/3*scale1 + 10e-3 -scale1/2;
xp_end_mm   = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2,  3]/3*scale1 - 10e-3 -scale1/2;
yp_start_mm = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,  3]/3*scale2         -scale2/2;
yp_end_mm   = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,  3]/3*scale2         -scale2/2;
z_mm        = [0,1,2,5,4,3,6,7,8,11,10,9]*20e-3;

if is_flip_z
    z_mm = z_mm(end:-1:1);
end

if ~is_flip_xy
    x_start_mm = xp_start_mm + offset_x_mm + centerX;
    x_end_mm =   xp_end_mm   + offset_x_mm + centerX;
    y_start_mm = yp_start_mm + offset_y_mm + centerY;
    y_end_mm =   yp_end_mm   + offset_y_mm + centerY;
else
    x_start_mm = yp_start_mm + offset_x_mm + centerX;
    x_end_mm =   yp_end_mm   + offset_x_mm + centerX;
    y_start_mm = xp_start_mm + offset_y_mm + centerY;
    y_end_mm =   xp_end_mm   + offset_y_mm + centerY;
end 
end
