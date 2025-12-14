clearvars; close all; clc;

%% ---------------- Parameters ----------------
N = 100;                % number of students
simT = 10000;           % simulation horizon in seconds
dt = 1;                 % sampling interval for queue time-series (s)
arrival_jitter_split = 600;   % 10 minutes
arrival_jitter_all12 = 100;   % ~1 student / second for 100 students

trials = 100;           % Number of trials

% Class end times (seconds since 10:00AM)
t_10_20 = 20*60;        % 1200 s
t_12_00 = 120*60;       % 7200 s

% Check-in service distribution (uniform)
checkin_min = 2;
checkin_max = 5;

% Station capacities
capacity.P = 6;
capacity.E = 6;
capacity.G = 6;
capacity.C = 8;

% Station deterministic service times (s)
tP = 10; tE = 10; tG = 10; tC = 30;

% Food choice probabilities
P_pizza   = 0.3232;
P_entree  = 0.7448;
P_grill   = 0.5310;
P_corner  = 0.4115;

% Path probabilities (PEGC, PEG, PEC, PE)
P_PEGC = P_pizza * P_entree * P_grill * P_corner;
P_PEG = P_pizza * P_entree * P_grill;
P_PEC = P_pizza * P_entree * P_corner;
P_PE = P_pizza * P_entree;

total_prob = P_PEGC + P_PEG + P_PEC + P_PE;

P_PEGC_normalized = P_PEGC / total_prob;
P_PEG_normalized = P_PEG / total_prob;
P_PEC_normalized = P_PEC / total_prob;
P_PE_normalized = P_PE / total_prob;

path_probs = [P_PEGC_normalized, P_PEG_normalized, P_PEC_normalized, P_PE_normalized];

% Travel times between nodes
tt_checkin_to_station.P = 1; tt_checkin_to_station.E = 2;
tt_checkin_to_station.G = 3; tt_checkin_to_station.C = 4;

tt_station_to_station.P.E = 1; tt_station_to_station.P.G = 1; tt_station_to_station.P.C = 1;
tt_station_to_station.E.G = 1; tt_station_to_station.E.C = 1;
tt_station_to_station.G.C = 1;

% Output folder
logfolder = pwd;

%% Validate
if abs(sum(path_probs)-1) > 1e-12
    error('path_probs must sum to 1');
end

%% Scenarios
scenarios = {'split','all12'};
results = struct();

for sIdx = 1:numel(scenarios)
    scenario = scenarios{sIdx};
    fprintf('\n=== Scenario: %s ===\n', scenario);

    times = 0:dt:simT;
    nTimes = numel(times);

    % storage across trials
    queue_ts_trials = zeros(trials, nTimes);
    station_queue_ts_trials = zeros(trials, 4, nTimes); % P,E,G,C
    trials_summary = cell(trials,1);

    for tr = 1:trials
        fprintf(' Trial %d/%d ...\n', tr, trials);
        summary = struct();   % reset per trial


        % Generate arrivals and sample paths
        arrivals = generate_arrivals(N, scenario, ...
    t_10_20, t_12_00, arrival_jitter_split, arrival_jitter_all12);
        paths = sample_paths(N, path_probs);

        % Run check-in
        station_next_free.P = 0;
        station_next_free.E = 0;
        station_next_free.G = 0;
        station_next_free.C = 0;

        [ci_start, ci_end, ci_wait] = run_checkin(arrivals, checkin_min, checkin_max);

        % -------------------- Station simulation with food choice --------------------
        [student_timeline, station_records, station_queue_log] = ...
    simulate_stations( ...
        ci_end, paths, tP, tE, tG, tC, ...
        tt_checkin_to_station, tt_station_to_station, capacity);

        % Compute queue time-series
        q_checkin = compute_queue_ts(arrivals, ci_start, times); % waiting outside check-in
        qP = queue_ts_from_log(station_queue_log.P, times);
        qE = queue_ts_from_log(station_queue_log.E, times);
        qG = queue_ts_from_log(station_queue_log.G, times);
        qC = queue_ts_from_log(station_queue_log.C, times);

        % Save CSV logs
        % logfile = fullfile(logfolder, sprintf('checkin_log_%s_trial%02d.csv', scenario, tr));
        % T = table((1:N)', round(arrivals,3), round(ci_start,3), round(ci_end,3), round(ci_wait,3), paths(:), ...
        %     'VariableNames',{'student_id','arrival_s','checkin_start_s','checkin_end_s','checkin_wait_s','path'});
        % writetable(T, logfile);
        % fprintf('  Saved check-in CSV: %s\n', logfile);

        % stationfile = fullfile(logfolder, sprintf('station_log_%s_trial%02d.csv', scenario, tr));
        % Srows = cell(N,10);
        % for i = 1:N
        %     st = student_timeline{i};
        %     row = cell(1,10);
        %     row{1} = i;
        %     row{2} = paths{i};
        %     row{3} = NaN;
        %     row{4} = get_field_or_nan(st,'P',1);
        %     row{5} = get_field_or_nan(st,'P',2);
        %     row{6} = get_field_or_nan(st,'E',1);
        %     row{7} = get_field_or_nan(st,'E',2);
        %     row{8} = get_field_or_nan(st,'G',1);
        %     row{9} = get_field_or_nan(st,'G',2);
        %     row{10} = get_field_or_nan(st,'C',1);
        %     row{11} = get_field_or_nan(st,'C',2);
        %     Srows(i,1:numel(row)) = row;
        % end
        % S = cell2table(Srows, 'VariableNames', {'student_id','path','checkin_end','P_start','P_end','E_start','E_end','G_start','G_end','C_start','C_end'});
        % writetable(S, stationfile);
        % fprintf('  Saved station CSV: %s\n', stationfile);

        % Store timeseries
        queue_ts_trials(tr,:) = q_checkin;
        station_queue_ts_trials(tr,1,:) = qP;
        station_queue_ts_trials(tr,2,:) = qE;
        station_queue_ts_trials(tr,3,:) = qG;
        station_queue_ts_trials(tr,4,:) = qC;

        % Summaries
        summary.avg_checkin_queue = mean(q_checkin);
        [summary.peak_checkin, idx_peak] = max(q_checkin);
        summary.peak_checkin_time = times(idx_peak);
        summary.avg_checkin_wait = mean(ci_wait);
        summary.pct_wait_over30 = mean(ci_wait > 30) * 100;
        trials_summary{tr} = summary;
    end

    % Aggregate results
    results.(scenario).times = times;
    results.(scenario).queue_ts_mean = mean(queue_ts_trials, 1);
    results.(scenario).station_queue_ts_mean = squeeze(mean(station_queue_ts_trials, 1));
    results.(scenario).trials_summary = trials_summary;
end

%% ---------------- Plot results ----------------
figure('Name','Check-in & Station queues','NumberTitle','off','Units','normalized','Position',[0.05 0.05 0.9 0.85]);
t = results.split.times;
subplot(3,1,1);
plot(t, results.split.queue_ts_mean, 'LineWidth', 1.5); hold on;
plot(t, results.all12.queue_ts_mean, 'LineWidth', 1.5);
xlabel('Time since 10:00 AM (s)'); ylabel('Check-in queue');
legend('50/50 split','All@12:00'); grid on; title('Check-in queue');

subplot(3,1,2);
plot(t, results.split.station_queue_ts_mean(1,:), 'LineWidth', 1.2); hold on;
plot(t, results.split.station_queue_ts_mean(2,:), 'LineWidth', 1.2);
plot(t, results.split.station_queue_ts_mean(3,:), 'LineWidth', 1.2);
plot(t, results.split.station_queue_ts_mean(4,:), 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Queue length');
legend('Pizza','Entree','Grill','Corner'); grid on; title('Station queues (50/50 split)');

subplot(3,1,3);
plot(t, results.all12.station_queue_ts_mean(1,:), 'LineWidth', 1.2); hold on;
plot(t, results.all12.station_queue_ts_mean(2,:), 'LineWidth', 1.2);
plot(t, results.all12.station_queue_ts_mean(3,:), 'LineWidth', 1.2);
plot(t, results.all12.station_queue_ts_mean(4,:), 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Queue length');
legend('Pizza','Entree','Grill','Corner'); grid on; title('Station queues (All@12:00)');

%% ---------------- Helper functions ----------------

function x = get_field_or_nan(st,field,idx)
    if isfield(st,field)
        x = st.(field)(idx);
    else
        x = NaN;
    end
end

function arrivals = generate_arrivals(N, scenario, t1, t2, jitter_split, jitter_all12)

    switch scenario
        case 'split'
            n1 = floor(N/2);
            n2 = N - n1;
            a1 = t1 + rand(n1,1) * jitter_split;
            a2 = t2 + rand(n2,1) * jitter_split;
            arrivals = sort([a1; a2]);

        case 'all12'
            arrivals = sort(t2 + rand(N,1) * jitter_all12);

        otherwise
            error('Unknown scenario');
    end
end

function paths = sample_paths(N, probs)
    edges_cum = cumsum(probs(:)');
    r = rand(N,1);
    paths = cell(N,1);
    for i = 1:N
        if r(i) <= edges_cum(1)
            paths{i} = 'PEGC';
        elseif r(i) <= edges_cum(2)
            paths{i} = 'PEG';
        elseif r(i) <= edges_cum(3)
            paths{i} = 'PEC';
        else
            paths{i} = 'PE';
        end
    end
end

function [start_times, end_times, wait_times] = run_checkin(arrivals, smin, smax)

    N = numel(arrivals);
    start_times = zeros(N,1);
    end_times   = zeros(N,1);
    wait_times  = zeros(N,1);

    next_free = 0;

    for i = 1:N
        sstart = max(arrivals(i), next_free);
        service = smin + (smax - smin)*rand();
        send = sstart + service;

        start_times(i) = sstart;
        end_times(i)   = send;
        wait_times(i)  = sstart - arrivals(i);

        next_free = send;
    end
end

%% ---------------- Station simulation ----------------
function [student_timeline, station_records, station_queue_log] = ...
    simulate_stations(checkin_end, paths, tP, tE, tG, tC, ...
                      tt_ci, tt_ss, capacity)

    stations = {'P','E','G','C'};
    service = struct('P',tP,'E',tE,'G',tG,'C',tC);
    next_free = struct('P',0,'E',0,'G',0,'C',0);

    % queues store SERVICE START TIMES
    queue = struct('P',[],'E',[],'G',[],'C',[]);

    % logs
    for s = stations
        station_records.(s{1}).arrivals = [];
        station_records.(s{1}).starts   = [];
        station_queue_log.(s{1}).time   = [];
        station_queue_log.(s{1}).qlen   = [];
    end

    N = numel(checkin_end);
    student_timeline = cell(N,1);
    [~, order] = sort(checkin_end);

    for k = 1:N
        i = order(k);
        cur_t = checkin_end(i);
        last = '';
        st = struct();

        for j = 1:numel(paths{i})
            s = paths{i}(j);

            % travel
            if isempty(last)
                t_arr = cur_t + tt_ci.(s);
            else
                t_arr = cur_t + tt_ss.(last).(s);
            end

            % clean queue (remove started)
            queue.(s)(queue.(s) <= t_arr) = [];

            % capacity block
            if numel(queue.(s)) >= capacity.(s)
                t_arr = min(queue.(s));   % wait for first to start
                queue.(s)(queue.(s) <= t_arr) = [];
            end

            s_start = max(t_arr, next_free.(s));
            s_end   = s_start + service.(s);

            % update
            queue.(s) = [queue.(s), s_start];
            next_free.(s) = s_end;

            station_records.(s).arrivals(end+1) = t_arr;
            station_records.(s).starts(end+1)   = s_start;

            station_queue_log.(s).time(end+1) = t_arr;
            station_queue_log.(s).qlen(end+1) = numel(queue.(s));

            st.(s) = [s_start s_end];
            cur_t = s_end;
            last = s;
        end

        student_timeline{i} = st;
    end
end

%% ---------------- Queue timeseries ----------------
function q = compute_queue_ts(arrivals, service_starts, times)
    arrivals_sorted = sort(arrivals(:));
    starts_sorted = sort(service_starts(:));
    nTimes = numel(times);
    q = zeros(1, nTimes);
    ia = 1; isv = 1;
    nA = numel(arrivals_sorted); nS = numel(starts_sorted);
    for k = 1:nTimes
        t = times(k);
        while ia <= nA && arrivals_sorted(ia) <= t
            ia = ia + 1;
        end
        a_count = ia - 1;
        while isv <= nS && starts_sorted(isv) <= t
            isv = isv + 1;
        end
        s_count = isv - 1;
        q(k) = max(0, a_count - s_count);
    end
end

function q = queue_ts_from_log(log, times)

    q = zeros(size(times));
    idx = 1;

    for k = 1:numel(times)
        while idx <= numel(log.time) && log.time(idx) <= times(k)
            q(k) = log.qlen(idx);
            idx = idx + 1;
        end
    end
end