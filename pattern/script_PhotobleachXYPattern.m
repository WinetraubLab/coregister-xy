% Run this demo to use Thorlabs system to photobleach a pattern 

% Before running this script, make sure myOCT folder is in path for example
% by running: addpath(genpath('F:\Jenkins\Scan OCTHist Dev\workspace\'))

%% Inputs

% When set to true the stage will not move and we will not
% photobleach. Use "true" when you would like to see the output without
% physcaily running the test.
skipHardware = false;

% Input coordinates in mm for each pattern center
patternCenter_mm = [0,0,0;
                    0,1,0;
                    1,0,0;
                    -1,0,0;
                    0,-1,0];

% OCT probe
octProbePath = yOCTGetProbeIniPath('40x','OCTP900'); % Select lens magnification

% Pattern to photobleach. System will photobleach n lines from 
% (x_start(i), y_start(i)) to (x_end(i), y_end(i)) at height z
% Load the pattern
[x_start_mm, x_end_mm, y_start_mm, y_end_mm, z_mm] = ...
    generateXYPattern(true, patternCenter_mm);

% Photobleach configurations
nPasses = 2; % For gel use 1 pass. For brain tissue use 2. Keep as low as possible. If galvo gets stuck, increase number. 
line_exposure_sec_mm = 5; % sec/mm. For gel use 0.5 sec/mm. For brain tissue use 5 sec/mm

%% Perform photobleach of pattern
yOCTPhotobleachTile(...
    [x_start_mm; y_start_mm], ...
    [x_end_mm; y_end_mm], ...
    'octProbePath',octProbePath, ...
    'exposure',line_exposure_sec_mm,'nPasses',nPasses,...
    'z',z_mm,'skipHardware',skipHardware ...
    ...,'laserToggleMethod','LaserPowerSwitch' ... Comment this section out to use laser
    );  

disp('Done Patterning')
