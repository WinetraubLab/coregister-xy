function [...
    x_start_mm, x_end_mm, ...
    y_start_mm, y_end_mm, ...
    z_mm] = generateXYPattern(verbose)
% This function generates the instructions (lines) for XY lines
% The lines are from (x_start_mm(i), y_start_mm(i)) (to x_end_mm(i), y_end_mm(i)) 
% at depth z
% INPUT: verbose (default: false). If set to true, function will output
% line information as well as image 

%% Inputs check
if ~exist('verbose','var')
    verbose = false;
end

%% Bulk of the pattern
% Pattern to photobleach. System will photobleach n lines from 
x_start_mm = []; x_end_mm=[]; y_start_mm=[]; y_end_mm=[]; z_mm=[];
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = Create1BlockPattern(180e-3, 0, false);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = Create1BlockPattern(-180e-3, 0, false);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = Create1BlockPattern(0, 180e-3, true);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];
[x_start_mm1, x_end_mm1, y_start_mm1, y_end_mm1, z_mm1] = Create1BlockPattern(0, -180e-3, true);
x_start_mm = [x_start_mm x_start_mm1]; x_end_mm=[x_end_mm x_end_mm1]; y_start_mm=[y_start_mm y_start_mm1]; y_end_mm=[y_end_mm y_end_mm1]; z_mm=[z_mm z_mm1];

%% Alignment markers (L shape)
nGridLines = 0;

x_start_mm = [x_start_mm [-0.15 -0.15 -0.15 0.1  0.16]];
x_end_mm =   [x_end_mm   [-0.15 -0.15 -0.1  0.13 0.19]];
y_start_mm = [y_start_mm [-0.15  0.1   0.15 0.15 0.15]];
y_end_mm =   [y_end_mm   [-0.1   0.15  0.15 0.15 0.15]];
nGridLines = nGridLines + 5;


%% Markers depths    
% Set Z
if ~isempty(z_mm)
    L_depth_mm = min(z_mm) + 3*mean(diff(unique(z_mm)));
else
    L_depth_mm = 0;
end
z_mm = [z_mm ones(1,nGridLines)*L_depth_mm];

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

%% Small Pattern
function [x_start_mm, x_end_mm, y_start_mm, y_end_mm, z_mm] = Create1BlockPattern(offset_x_mm, offset_y_mm, isflip)
scale1 = 100e-3; %mm
scale2 = 170e-3;
xp_start_mm = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,  2]/3*scale1 + 10e-3 -scale1/2;
xp_end_mm   = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2,  3]/3*scale1 - 10e-3 -scale1/2;
yp_start_mm = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,  3]/3*scale2         -scale2/2;
yp_end_mm   = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,  3]/3*scale2         -scale2/2;
z_mm        = [0,1,2,5,4,3,6,7,8,11,10,9]*20e-3;

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