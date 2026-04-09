%% DMS Simulation - PERCLOS and EAR Evolution
%% Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
%% This script simulates PERCLOS evolution, EAR signal with noise,
%% and state transitions over time.

clear; clc; close all;

%% Parameters
Fs = 30;                    % Sampling rate (frames per second)
duration = 120;             % Simulation duration (seconds)
t = 0:1/Fs:duration;
N = length(t);

%% Generate synthetic EAR signal
% Base EAR = 0.30 (open eyes), with gradual decrease for drowsiness
ear_base = 0.30 * ones(1, N);

% Add drowsiness phase (60-90s): EAR drops gradually
drowsy_start = 60 * Fs;
drowsy_end = 90 * Fs;
for i = drowsy_start:min(drowsy_end, N)
    progress = (i - drowsy_start) / (drowsy_end - drowsy_start);
    ear_base(i) = 0.30 - 0.15 * progress;
end

% Recovery phase (90-120s)
for i = min(drowsy_end, N):N
    progress = (i - drowsy_end) / (N - drowsy_end + 1);
    ear_base(i) = 0.15 + 0.15 * progress;
end

% Add Gaussian noise
ear_noise = 0.02 * randn(1, N);
ear_signal = max(ear_base + ear_noise, 0.05);

%% Compute PERCLOS (sliding window of 30s)
window_size = 30 * Fs;     % 30 seconds window
ear_threshold = 0.25;
perclos = zeros(1, N);

for i = 1:N
    win_start = max(1, i - window_size + 1);
    window = ear_signal(win_start:i);
    closed_count = sum(window < ear_threshold);
    perclos(i) = closed_count / length(window);
end

%% State determination
% States: 1=Attentive, 2=Warning, 3=Fatigue, 4=Emergency
perclos_warning = 0.20;
perclos_fatigue = 0.35;
perclos_emergency = 0.50;

state = ones(1, N);
for i = 1:N
    if perclos(i) >= perclos_emergency
        state(i) = 4;  % Emergency
    elseif perclos(i) >= perclos_fatigue
        state(i) = 3;  % Fatigue
    elseif perclos(i) >= perclos_warning
        state(i) = 2;  % Warning
    else
        state(i) = 1;  % Attentive
    end
end

%% Plotting
figure('Position', [100, 100, 1200, 800], 'Name', 'DMS Simulation');

% EAR vs Time
subplot(3, 1, 1);
plot(t, ear_signal, 'b-', 'LineWidth', 0.5);
hold on;
yline(ear_threshold, 'r--', 'LineWidth', 1.5, 'Label', 'EAR Threshold');
xlabel('Time (s)');
ylabel('EAR');
title('Eye Aspect Ratio (EAR) Over Time');
grid on;
legend('EAR Signal', 'Threshold');
xlim([0, duration]);

% PERCLOS vs Time
subplot(3, 1, 2);
plot(t, perclos * 100, 'g-', 'LineWidth', 1.5);
hold on;
yline(perclos_warning * 100, 'y--', 'LineWidth', 1.5, 'Label', 'Warning');
yline(perclos_fatigue * 100, 'r--', 'LineWidth', 1.5, 'Label', 'Fatigue');
yline(perclos_emergency * 100, 'm--', 'LineWidth', 1.5, 'Label', 'Emergency');
xlabel('Time (s)');
ylabel('PERCLOS (%)');
title('PERCLOS Evolution');
grid on;
legend('PERCLOS', 'Warning Threshold', 'Fatigue Threshold', 'Emergency Threshold');
xlim([0, duration]);

% State vs Time
subplot(3, 1, 3);
stairs(t, state, 'k-', 'LineWidth', 2);
yticks([1, 2, 3, 4]);
yticklabels({'Attentive', 'Warning', 'Fatigue', 'Emergency'});
xlabel('Time (s)');
ylabel('Driver State');
title('ECU State Transitions');
grid on;
xlim([0, duration]);
ylim([0.5, 4.5]);

sgtitle('DMS Simulation - Daifi Meriem, Expleo Group / Stellantis');

% Save figure
saveas(gcf, 'matlab/dms_simulation_results.png');
fprintf('Simulation complete. Results saved.\n');
