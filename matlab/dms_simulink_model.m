%% DMS Simulink Model Description
%% Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
%%
%% This script describes the Simulink model structure for the DMS.
%% It creates a programmatic description of the block diagram.

clear; clc;

%% ===================================================================
%% MODEL STRUCTURE DESCRIPTION
%% ===================================================================
fprintf('=== DMS Simulink Model Structure ===\n\n');

%% Top-Level Block Diagram
fprintf('Top-Level System: DMS_System\n');
fprintf('================================\n');
fprintf('Inputs:\n');
fprintf('  - Camera_Frame (Video Source)\n');
fprintf('  - Vehicle_Speed (CAN Bus)\n');
fprintf('\n');
fprintf('Subsystems:\n');
fprintf('  1. Face_Detection_Subsystem\n');
fprintf('     - Input: Camera_Frame\n');
fprintf('     - Output: Face_ROI, Eye_ROI, Landmarks\n');
fprintf('     - Method: MediaPipe FaceMesh / Haar Cascade\n');
fprintf('\n');
fprintf('  2. Feature_Extraction_Subsystem\n');
fprintf('     - Input: Face_ROI, Eye_ROI, Landmarks\n');
fprintf('     - Output: EAR, MAR, Head_Angles\n');
fprintf('     - Blocks: EAR_Calculator, MAR_Calculator, PnP_Solver\n');
fprintf('\n');
fprintf('  3. Behavioral_Analysis_Subsystem\n');
fprintf('     - Input: EAR, MAR\n');
fprintf('     - Output: PERCLOS, Blink_Rate, Drowsiness_Score\n');
fprintf('     - Blocks: Sliding_Window, PERCLOS_Calculator\n');
fprintf('\n');
fprintf('  4. CNN_Inference_Subsystem\n');
fprintf('     - Input: Eye_ROI\n');
fprintf('     - Output: CNN_Class, CNN_Confidence\n');
fprintf('     - Blocks: Preprocessing, CNN_Model, Postprocessing\n');
fprintf('\n');
fprintf('  5. ECU_Decision_Subsystem (Stateflow)\n');
fprintf('     - Input: PERCLOS, Head_Angles, Face_Detected, CNN_Class\n');
fprintf('     - Output: Driver_State, Warning_Level, Action_Commands\n');
fprintf('     - Logic: Finite State Machine with priority rules\n');
fprintf('\n');
fprintf('  6. Actuator_Interface_Subsystem\n');
fprintf('     - Input: Action_Commands\n');
fprintf('     - Output: Audio_PWM, Vibration_PWM, Brake_Signal, Hazard_Signal\n');
fprintf('\n');

%% Signal Routing
fprintf('Signal Routing:\n');
fprintf('  Camera_Frame --> Face_Detection --> Feature_Extraction\n');
fprintf('  Feature_Extraction --> Behavioral_Analysis\n');
fprintf('  Feature_Extraction --> CNN_Inference\n');
fprintf('  [Behavioral_Analysis, CNN_Inference, Feature_Extraction] --> ECU_Decision\n');
fprintf('  ECU_Decision --> Actuator_Interface\n');
fprintf('\n');

%% Bus Definitions
fprintf('Bus Definitions:\n');
fprintf('  DetectionBus: {face_bbox, eye_roi, landmarks, face_detected}\n');
fprintf('  FeatureBus: {ear, mar, yaw, pitch, roll}\n');
fprintf('  AnalysisBus: {perclos, blink_rate, drowsiness_score}\n');
fprintf('  DecisionBus: {state, warning_level, action_commands}\n');
fprintf('  ActionBus: {audio, vibration_seat, vibration_steering, brake, hazard}\n');

fprintf('\nModel description complete.\n');
