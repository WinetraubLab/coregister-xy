classdef test_generateXYPattern < matlab.unittest.TestCase
    % Test generating a pattern
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each testa
    end
    
    methods(Test)
        function testZPosition(testCase)
            [~,~,~,~,z_mm] = generateXYPattern();

            minZ = min(z_mm(:));
            
            % Test that minZ is 0
            if minZ ~= 0
                testCase.verifyFail('Min photobleach depth of pattern should be 0');
            end
        end
   
    end
end