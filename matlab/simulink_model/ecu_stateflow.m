%% ECU Stateflow State Machine
%% Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
%%
%% Stateflow-like state machine for ECU decision logic.
%% Implements states: Attentive, Warning, Fatigue, Distracted, Emergency
%% with guard conditions and timer-based transitions.

classdef ecu_stateflow < handle
    %ECU_STATEFLOW Stateflow-like state machine for DMS ECU.
    %   Implements driver state monitoring with priority-based transitions
    %   and warning escalation.

    properties
        current_state       % Current state name (string)
        warning_level       % Warning level (0-3)
        no_face_timer       % Timer for face absence (seconds)
        head_turn_timer     % Timer for head turn (seconds)
        warning_timer       % Timer for warning escalation (seconds)
        dt                  % Time step (seconds)
    end

    properties (Constant)
        % Thresholds
        NO_FACE_TIMEOUT = 3.0
        HEAD_TURN_TIMEOUT = 6.0
        PERCLOS_WARNING = 0.20
        PERCLOS_FATIGUE = 0.35
        EAR_WARNING = 0.25
        MAR_YAWN = 0.6
        ESCALATION_TIMEOUT = 5.0

        % State names
        STATE_ATTENTIVE = 'Attentive'
        STATE_WARNING = 'Warning'
        STATE_FATIGUE = 'Fatigue'
        STATE_DISTRACTED = 'Distracted'
        STATE_EMERGENCY = 'Emergency'
    end

    methods
        function obj = ecu_stateflow(dt)
            %ECU_STATEFLOW Constructor.
            %   obj = ecu_stateflow(dt) creates state machine with time step dt.
            if nargin < 1
                dt = 1/30;  % Default 30 FPS
            end
            obj.dt = dt;
            obj.current_state = obj.STATE_ATTENTIVE;
            obj.warning_level = 0;
            obj.no_face_timer = 0;
            obj.head_turn_timer = 0;
            obj.warning_timer = 0;
        end

        function [state, level, actions] = step(obj, indicators)
            %STEP Process one frame of indicators.
            %   [state, level, actions] = step(obj, indicators)
            %   indicators is a struct with fields:
            %     face_detected, head_distracted, perclos, ear, mar

            % Update timers
            if ~indicators.face_detected
                obj.no_face_timer = obj.no_face_timer + obj.dt;
            else
                obj.no_face_timer = 0;
            end

            if indicators.head_distracted
                obj.head_turn_timer = obj.head_turn_timer + obj.dt;
            else
                obj.head_turn_timer = 0;
            end

            % Priority-based state determination
            new_state = obj.current_state;

            % Priority 1: No face
            if obj.no_face_timer >= obj.NO_FACE_TIMEOUT
                new_state = obj.STATE_DISTRACTED;
            % Priority 2: Head turned too long
            elseif obj.head_turn_timer >= obj.HEAD_TURN_TIMEOUT
                new_state = obj.STATE_DISTRACTED;
            % Priority 3: PERCLOS fatigue
            elseif indicators.perclos >= obj.PERCLOS_FATIGUE
                new_state = obj.STATE_FATIGUE;
            % Priority 4: Warning indicators
            elseif indicators.ear < obj.EAR_WARNING || ...
                   indicators.perclos >= obj.PERCLOS_WARNING || ...
                   indicators.mar >= obj.MAR_YAWN
                new_state = obj.STATE_WARNING;
            else
                new_state = obj.STATE_ATTENTIVE;
            end

            % State entry/exit actions
            if ~strcmp(new_state, obj.current_state)
                obj.on_exit(obj.current_state);
                obj.current_state = new_state;
                obj.on_entry(new_state);
            end

            % Warning escalation
            obj.update_escalation();

            % Generate action commands
            state = obj.current_state;
            level = obj.warning_level;
            actions = obj.get_actions();
        end

        function on_entry(obj, state)
            %ON_ENTRY Actions when entering a state.
            switch state
                case obj.STATE_ATTENTIVE
                    obj.warning_level = 0;
                    obj.warning_timer = 0;
                case obj.STATE_WARNING
                    obj.warning_level = 1;
                    obj.warning_timer = 0;
                case obj.STATE_FATIGUE
                    obj.warning_level = 2;
                case obj.STATE_DISTRACTED
                    obj.warning_level = 2;
                case obj.STATE_EMERGENCY
                    obj.warning_level = 3;
            end
            fprintf('  [ECU] Enter state: %s (level=%d)\n', state, obj.warning_level);
        end

        function on_exit(obj, state)
            %ON_EXIT Actions when exiting a state.
            fprintf('  [ECU] Exit state: %s\n', state);
        end

        function update_escalation(obj)
            %UPDATE_ESCALATION Escalate warning level over time.
            if strcmp(obj.current_state, obj.STATE_WARNING) || ...
               strcmp(obj.current_state, obj.STATE_FATIGUE) || ...
               strcmp(obj.current_state, obj.STATE_DISTRACTED)
                obj.warning_timer = obj.warning_timer + obj.dt;
                if obj.warning_timer >= obj.ESCALATION_TIMEOUT
                    if obj.warning_level < 3
                        obj.warning_level = obj.warning_level + 1;
                        obj.warning_timer = 0;
                        if obj.warning_level >= 3
                            obj.current_state = obj.STATE_EMERGENCY;
                        end
                    end
                end
            end
        end

        function actions = get_actions(obj)
            %GET_ACTIONS Generate action commands based on warning level.
            actions = struct(...
                'audio_alert', false, ...
                'visual_alert', false, ...
                'seat_vibration', false, ...
                'steering_vibration', false, ...
                'emergency_braking', false, ...
                'pull_to_side', false, ...
                'hazard_lights', false ...
            );

            if obj.warning_level >= 1
                actions.audio_alert = true;
                actions.visual_alert = true;
            end
            if obj.warning_level >= 2
                actions.seat_vibration = true;
                actions.steering_vibration = true;
            end
            if obj.warning_level >= 3
                actions.emergency_braking = true;
                actions.pull_to_side = true;
                actions.hazard_lights = true;
            end
        end

        function simulate(obj, duration)
            %SIMULATE Run a complete simulation scenario.
            %   simulate(obj, duration) runs for 'duration' seconds.
            if nargin < 2
                duration = 60;
            end

            N = round(duration / obj.dt);
            states = cell(1, N);
            levels = zeros(1, N);
            t = (0:N-1) * obj.dt;

            for i = 1:N
                % Generate synthetic indicators
                ind = struct();
                ind.face_detected = true;
                ind.head_distracted = false;

                % Simulate drowsiness progression
                progress = i / N;
                if progress < 0.3
                    ind.perclos = 0.05;
                    ind.ear = 0.30;
                    ind.mar = 0.2;
                elseif progress < 0.6
                    ind.perclos = 0.05 + (progress - 0.3) / 0.3 * 0.35;
                    ind.ear = 0.30 - (progress - 0.3) / 0.3 * 0.15;
                    ind.mar = 0.2 + (progress - 0.3) / 0.3 * 0.4;
                else
                    ind.perclos = 0.40;
                    ind.ear = 0.15;
                    ind.mar = 0.6;
                    if progress > 0.8
                        ind.face_detected = false;
                    end
                end

                [state, level, ~] = obj.step(ind);
                states{i} = state;
                levels(i) = level;
            end

            % Plot results
            figure('Name', 'ECU Stateflow Simulation');

            % Convert states to numeric
            state_num = zeros(1, N);
            for i = 1:N
                switch states{i}
                    case obj.STATE_ATTENTIVE
                        state_num(i) = 1;
                    case obj.STATE_WARNING
                        state_num(i) = 2;
                    case obj.STATE_FATIGUE
                        state_num(i) = 3;
                    case obj.STATE_DISTRACTED
                        state_num(i) = 4;
                    case obj.STATE_EMERGENCY
                        state_num(i) = 5;
                end
            end

            subplot(2,1,1);
            stairs(t, state_num, 'b-', 'LineWidth', 2);
            yticks(1:5);
            yticklabels({'Attentive','Warning','Fatigue','Distracted','Emergency'});
            xlabel('Time (s)'); ylabel('State');
            title('ECU Stateflow: State Transitions');
            grid on;

            subplot(2,1,2);
            stairs(t, levels, 'r-', 'LineWidth', 2);
            yticks(0:3);
            yticklabels({'None','Level 1','Level 2','Level 3'});
            xlabel('Time (s)'); ylabel('Warning Level');
            title('Warning Escalation');
            grid on;

            sgtitle('ECU Stateflow Simulation - Daifi Meriem');
        end
    end
end
