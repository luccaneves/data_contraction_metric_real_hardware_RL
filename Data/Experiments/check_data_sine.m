clc
clear all


start = ""



folders = {"FL/", "FL_RL/", "FL_RL_ONLINE/",
    "SMC/", "SMC_RL/", "SMC_RL_ONLINE/"};

files = {"20k_05hz","20k_1hz", "20k_15hz", "20k_2hz", "20k_25hz", "20k_3hz",...
    "10k_05hz","10k_1hz", "10k_15hz", "10k_2hz", "10k_25hz", "10k_3hz"}


size_folders = length(folders);
size_files = length(files);

%size_ = 1;

for k = 1:size_files
    for i = 1:size_folders
        folders(i)
        files(k)
        dataAll = mdfRead(start + folders(i) + files(k) + ".mf4");
        
        dataAll = dataAll{1};
        
        tempo = dataAll.HostService;
        tempo = seconds(tempo);
        F_filter = dataAll.("Model Root/Scope_Force_Load_Filter/In1");
        
        Ref = dataAll.("Model Root/ScopeRefSignal/In1");
        
        start_index = 5500;
        end_index = 19500;
        
        title_fontsize = 20;
        axis_fontsize = 40;
        legend_fontsize = 40;
        default_font_size = 40;
        
        rmse_value = rmse(Ref(start_index:end_index),F_filter(start_index:end_index))
    end
end


folders = {"FL/", "SMC/", "FL_RL_ONLINE/"};

files = {"5k_05hz","5k_1hz", "5k_15hz", "5k_2hz", "5k_25hz", "5k_3hz"}


size_folders = length(folders);
size_files = length(files);

%size_ = 1;

for k = 1:size_files
    for i = 1:size_folders
        folders(i)
        files(k)
        dataAll = mdfRead(start + folders(i) + files(k) + ".mf4");
        
        dataAll = dataAll{1};
        
        tempo = dataAll.HostService;
        tempo = seconds(tempo);
        F_filter = dataAll.("Model Root/Scope_Force_Load_Filter/In1");
        
        Ref = dataAll.("Model Root/ScopeRefSignal/In1");
        
        start_index = 5500;
        end_index = 19500;
        
        title_fontsize = 20;
        axis_fontsize = 40;
        legend_fontsize = 40;
        default_font_size = 40;
        
        rmse_value = rmse(Ref(start_index:end_index),F_filter(start_index:end_index))
    end
end
%%
% Figure 1
A = figure(1)
plot(tempo(start_index:end_index) - tempo(start_index), Ref(start_index:end_index), ...
    'DisplayName', '$f_r$', 'LineStyle', '--', 'LineWidth', 1.5)
hold on
plot(tempo(start_index:end_index) - tempo(start_index), F_filter(start_index:end_index), ...
    'DisplayName', '$f_l$', 'LineWidth', 3)
grid on

title('', 'FontSize', title_fontsize)
xlabel('Time (s)', 'FontSize', axis_fontsize)
ylabel('Force (N)', 'FontSize', axis_fontsize)

legend('show', 'FontSize', legend_fontsize, 'Interpreter', 'latex')
xlim([0 (end_index - start_index)/1000])
set(gca, 'FontSize', default_font_size)

saveas(A, folders + file + ".png")

%%

current = dataAll.("Model Root/ScopeControl/In1");
control = dataAll.("Model Root/ScopeOutput/In1");
Kp = dataAll.("Model Root/ScopeKpRl/In1");
Ki = dataAll.("Model Root/ScopeKiRl/In1");

B = figure(10)
hold on

% Solid line for Δλ₂
plot(tempo(start_index:end_index) - tempo(start_index), Kp(start_index:end_index), ...
     'LineWidth', 2, 'DisplayName', '$\Delta K_i$', ...
     'Color', [0.0078, 0.3961, 0.6549])  % blue

% Dotted line for Δλ₁
plot(tempo(start_index:end_index) - tempo(start_index), Ki(start_index:end_index), ...
     'LineWidth', 2, 'LineStyle', '--', ...
     'DisplayName', '$\Delta K_p$', ...
     'Color', [0.8500, 0.3250, 0.0980])  % orange

grid on
title('', 'Interpreter', 'latex', 'FontSize', title_fontsize)
xlabel('Time (s)', 'FontSize', axis_fontsize)
ylabel('', 'FontSize', axis_fontsize)  % customize if needed
xlim([0, (end_index - start_index)/1000])

legend('Interpreter', 'latex', 'FontSize', legend_fontsize)
set(gca, 'FontSize', default_font_size)


saveas(B, folders + file + "_gain_variation.png")

%%

% Figure 2
figure(2)
plot(tempo(start_index:end_index)- tempo(start_index), Kp(start_index:end_index), 'LineWidth', 1.5)
grid on
title('', ...
      'Interpreter', 'latex', ...
      'FontSize', title_fontsize)

ylabel('$\Delta \lambda_2$', ...
      'Interpreter', 'latex', ...
      'FontSize', axis_fontsize)
xlabel('Time (s)', 'FontSize', axis_fontsize)
xlim([0 (end_index - start_index)/1000])

set(gca, 'FontSize', default_font_size)

% Figure 3
figure(3)
plot(tempo(start_index:end_index)- tempo(start_index), Ki(start_index:end_index), 'LineWidth', 1.5)
grid on
title('', ...
      'Interpreter', 'latex', ...
      'FontSize', title_fontsize)
ylabel('$\Delta \lambda_1$', ...
      'Interpreter', 'latex', ...
      'FontSize', axis_fontsize)
xlabel('Time (s)', 'FontSize', axis_fontsize)
xlim([0 (end_index - start_index)/1000])

set(gca, 'FontSize', default_font_size)



%%
% If Step:
clc

figure (123)

plot(tempo(4980:end_index)- tempo(4980),F_filter(4980:end_index))
hold on
grid on
plot(tempo(4980:end_index)- tempo(4980),Ref(4980:end_index))

[rise_time, settling_time, overshoot] = analyze_step_response(tempo(4980:end_index)- tempo(4980), Ref(4980:end_index), F_filter(4980:end_index))







