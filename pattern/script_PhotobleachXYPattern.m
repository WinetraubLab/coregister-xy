% Run this demo to use Thorlabs system to photobleach a pattern 

% Before running this script, make sure myOCT folder is in path for example
% by running: addpath(genpath('F:\Jenkins\Scan OCTHist Dev\workspace\'))

%% Inputs

% When set to true the stage will not move and we will not
% photobleach. Use "true" when you would like to see the output without
% physcaily running the test.
skipHardware = false;

% OCT probe
octProbePath = yOCTGetProbeIniPath('40x','OCTP900'); % Select lens magnification

% Pattern to photobleach. System will photobleach n lines from 
% (x_start(i), y_start(i)) to (x_end(i), y_end(i)) at height z
% Load the pattern
[x_start_mm, x_end_mm, y_start_mm, y_end_mm, z_mm] = ...
    generateXYPattern(true);

% Photobleach configurations
nPasses = 1; % Keep as low as possible. If galvo gets stuck, increase number
line_exposure_sec_mm = 0.5; % sec/mm. For gel use 0.5

%% Perform photobleach of pattern
uz_mm = unique(z_mm);
for i=1:length(uz_mm)
    ii = find(z_mm==uz_mm(i));
    fprintf('Photobleaching Depth (%d): %.3f mm\n',i,z_mm(ii(1)));
    yOCTPhotobleachTile(...
        [x_start_mm(ii); y_start_mm(ii)], ...
        [x_end_mm(ii); y_end_mm(ii)], ...
        'octProbePath',octProbePath, ...
        'exposure',line_exposure_sec_mm,'nPasses',nPasses,...
        'z',z_mm(ii(1)),'skipHardware',skipHardware ...
        ...,'laserToggleMethod','LaserPowerSwitch' ... Comment this section out to use laser
        );  
    pause(0.5);
end

disp('Done Patterning')
