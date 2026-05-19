clc 
clear all

start_index = 5000;
end_index = 25000;

title_fontsize = 20;
axis_fontsize = 40;
legend_fontsize = 40;
default_font_size = 40;


folders = {"FL/", "SMC/", "FL_RL_ONLINE/"};
size_folders = length(folders);
files = {"sweep_200_10hz_20k","sweep_150_10hz_10k","sweep_75_10hz_5k"};
size_files = length(files);
amps = [200, 150, 75];

plot_flag = 0;
 




function [wc] = bandSweep(force,Ref, maxf, amp)
    freq = linspace(0, maxf, length(force));
    meanF = mean(Ref);
    F = force - meanF;
    threshold_p = amp * (1/sqrt(2));
    threshold_n = -amp * (1/sqrt(2));
    F_p = F(F>0);
    F_n = F(F<0);
    freq_p = freq(F>0);
    freq_n = freq(F<0);
    [~,peaks_p] = findpeaks(F_p,'MinPeakDistance',100);
    [~,peaks_n] = findpeaks(abs(F_n),'MinPeakDistance',100);
    [wc_p,y_p]=intersections(freq_p(peaks_p),F_p(peaks_p),freq,threshold_p*ones(length(freq),1),1);
    [wc_n,y_n]=intersections(freq_n(peaks_n),F_n(peaks_n),freq,threshold_n*ones(length(freq),1),1);
    figure()
    plot(freq,F)
    hold on
    plot(freq_p(peaks_p),F_p(peaks_p),freq,threshold_p*ones(length(freq),1))
    plot(freq_n(peaks_n),F_n(peaks_n),freq,threshold_n*ones(length(freq),1))
    plot(wc_p,y_p,'*')
    plot(wc_n,y_n,'*')
    wc_p = max(wc_p);
    wc_n = max(wc_n);
    if 0 %(wc_p >= wc_n)
        wc = wc_n;
    else
        wc = wc_p;
    end
end

for i = 1:size_folders
    for j = 1:size_files
        dataAll = mdfRead(folders{i} + files{j} + ".mf4");
        %dataAll = mdfRead("dados/pid_rl_20k_3hz.mf4");
        
        dataAll = dataAll{1};
        
        tempo = dataAll.HostService;
        tempo = seconds(tempo);
        F_filter = dataAll.("Model Root/Scope_Force_Load_Filter/In1");
        
        Ref = dataAll.("Model Root/ScopeRefSignal/In1");
        
        
        wc_def = bandSweep(F_filter(start_index:end_index),Ref, 10, amps(j));
    end
end

folders = {"FL_RL/"};
size_folders = length(folders);
files = {"sweep_150_10hz_10k"};
size_files = length(files);
amps = [150];

for i = 1:size_folders
    for j = 1:size_files
        dataAll = mdfRead(folders{i} + files{j} + ".mf4");
        %dataAll = mdfRead("dados/pid_rl_20k_3hz.mf4");
        
        dataAll = dataAll{1};
        
        tempo = dataAll.HostService;
        tempo = seconds(tempo);
        F_filter = dataAll.("Model Root/Scope_Force_Load_Filter/In1");
        
        Ref = dataAll.("Model Root/ScopeRefSignal/In1");
        
        
        wc_def = bandSweep(F_filter(start_index:end_index),Ref, 10, amps(j));
    end
end