function [rise_time, settling_time, overshoot] = analyze_step_response(t, ref, y)
% ANALYZE_STEP_RESPONSE computes rise time, settling time, and overshoot
% from a reference and measured signal (y) over time (t).
%
% Inputs:
%   t   : time vector
%   ref : reference signal (e.g. step input)
%   y   : measured response signal
%
% Outputs:
%   rise_time      : time from 10% to 90% of final value
%   settling_time  : time for signal to stay within 2% of final value
%   overshoot      : (max(y) - final_value)/final_value * 100

    % Ensure column vectors
    t = t(:); ref = ref(:); y = y(:);

    % Identify final reference level (for step input)
    final_ref = ref(end);
    final_value = mean(y(end-10:end)); % steady-state average

    % Normalize for relative analysis
    y_norm = y / final_ref;
    final_value_norm = final_value / final_ref;

    % Rise time: 10% to 90% of final value
    y_10 = 0.1 * final_value_norm;
    y_90 = 0.9 * final_value_norm;

    idx_10 = find(y_norm >= y_10, 1, 'first');
    idx_90 = find(y_norm >= y_90, 1, 'first');

    if ~isempty(idx_10) && ~isempty(idx_90)
        rise_time = t(idx_90) - t(idx_10);
    else
        rise_time = NaN;
    end

    % Overshoot
    overshoot = (max(y) - final_value) / final_value * 100;

    % Settling time: within 2% of final value
    tol = 0.02 * abs(final_value);
    idx_settle = find(abs(y - final_value) > tol, 1, 'last');
    if isempty(idx_settle)
        settling_time = 0;
    else
        settling_time = t(idx_settle);
    end
end
