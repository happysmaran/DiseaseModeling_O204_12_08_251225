clearvars; close all; clc;

%% ---------------- Parameters ----------------
N = 100;                % number of students
simT = 10000;           % simulation horizon in seconds
dt = 1;                 % sampling interval for queue time-series (s)
trials = 1;             % Monte Carlo trials
arrival_jitter = 600;   % jitter after class end (s)

% Class end times (seconds since 10:00AM)
t_10_20 = 20*60;        % 1200 s
t_12_00 = 120*60;       % 7200 s

% Check-in service distribution (uniform)
checkin_min = 2;
checkin_max = 5;

% Station deterministic service times (s)
tP = 10; tE = 10; tG = 10; tC = 30;

% Path probabilities (PEGC, PEG, PEC, PE)
path_probs = [0.101, 0.246, 0.190, 0.463];

% Food choice probabilities
P_pizza   = 0.3232;
P_entree  = 0.7448;
P_grill   = 0.5310;
P_corner  = 0.4115;

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

        % Generate arrivals and sample paths
        arrivals = generate_arrivals(N, scenario, t_10_20, t_12_00, arrival_jitter);
        paths = sample_paths(N, path_probs);

        % Run check-in
        station_next_free.P = 0;
        station_next_free.E = 0;
        station_next_free.G = 0;
        station_next_free.C = 0;

        [ci_start, ci_end, ci_wait] = run_checkin(arrivals, checkin_min, checkin_max, paths, ...
                                         tP, tE, tG, tC, station_next_free);

        % -------------------- Station simulation with food choice --------------------
        [student_timeline, station_records] = simulate_stations_with_paths_and_food( ...
            ci_end, paths, P_pizza, P_entree, P_grill, P_corner, ...
            tP, tE, tG, tC, tt_checkin_to_station, tt_station_to_station);

        % Compute queue time-series
        q_checkin = compute_queue_ts(arrivals, ci_start, times); % waiting outside check-in
        qP = compute_queue_ts(station_records.P.arrivals, station_records.P.start_times, times);
        qE = compute_queue_ts(station_records.E.arrivals, station_records.E.start_times, times);
        qG = compute_queue_ts(station_records.G.arrivals, station_records.G.start_times, times);
        qC = compute_queue_ts(station_records.C.arrivals, station_records.C.start_times, times);

        % Save CSV logs
        logfile = fullfile(logfolder, sprintf('checkin_log_%s_trial%02d.csv', scenario, tr));
        T = table((1:N)', round(arrivals,3), round(ci_start,3), round(ci_end,3), round(ci_wait,3), paths(:), ...
            'VariableNames',{'student_id','arrival_s','checkin_start_s','checkin_end_s','checkin_wait_s','path'});
        writetable(T, logfile);
        fprintf('  Saved check-in CSV: %s\n', logfile);

        stationfile = fullfile(logfolder, sprintf('station_log_%s_trial%02d.csv', scenario, tr));
        Srows = cell(N,10);
        for i = 1:N
            st = student_timeline{i};
            row = cell(1,10);
            row{1} = i;
            row{2} = paths{i};
            row{3} = NaN;
            row{4} = get_field_or_nan(st,'P',1);
            row{5} = get_field_or_nan(st,'P',2);
            row{6} = get_field_or_nan(st,'E',1);
            row{7} = get_field_or_nan(st,'E',2);
            row{8} = get_field_or_nan(st,'G',1);
            row{9} = get_field_or_nan(st,'G',2);
            row{10} = get_field_or_nan(st,'C',1);
            row{11} = get_field_or_nan(st,'C',2);
            Srows(i,1:numel(row)) = row;
        end
        S = cell2table(Srows, 'VariableNames', {'student_id','path','checkin_end','P_start','P_end','E_start','E_end','G_start','G_end','C_start','C_end'});
        writetable(S, stationfile);
        fprintf('  Saved station CSV: %s\n', stationfile);

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
    results.(scenario).station_queue_ts_mean = squeeze(mean(station_queue_ts_trials,1));
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

function arrivals = generate_arrivals(N, scenario, t1, t2, jitter)
    switch scenario
        case 'split'
            n1 = floor(N/2);
            n2 = N - n1;
            a1 = t1 + rand(n1,1) * jitter;
            a2 = t2 + rand(n2,1) * jitter;
            arrivals = sort([a1; a2]);
        case 'all12'
            arrivals = sort(t2 + rand(N,1) * jitter);
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

function [start_times, end_times, wait_times] = run_checkin(arrivals, smin, smax, paths, tP, tE, tG, tC, station_next_free)
    N = numel(arrivals);
    start_times = zeros(N,1);
    end_times   = zeros(N,1);
    wait_times  = zeros(N,1);
    next_free_checkin = 0;
    
    for i = 1:N
        arr = arrivals(i);
        sstart = max(arr, next_free_checkin);
        service_time = smin + (smax - smin) * rand();
        send = sstart + service_time;
        
        path = paths{i};
        if contains(path,'P')
            next_station = 'P';
        elseif contains(path,'E')
            next_station = 'E';
        elseif contains(path,'G')
            next_station = 'G';
        elseif contains(path,'C')
            next_station = 'C';
        else
            next_station = ''; % safety
        end
        
        if ~isempty(next_station)
            send = max(send, station_next_free.(next_station));
            station_next_free.(next_station) = send;
        end
        
        start_times(i) = sstart;
        end_times(i)   = send;
        wait_times(i)  = sstart - arr;
        next_free_checkin = send;
    end
end

%% ---------------- Station simulation with paths + food choice ----------------
function [student_timeline, station_records] = simulate_stations_with_paths_and_food(checkin_end_times, paths, ...
        P_pizza, P_entree, P_grill, P_corner, tP, tE, tG, tC, tt_checkin_to_station, tt_station_to_station)

    N = numel(checkin_end_times);
    student_timeline = cell(N,1);

    stations = {'P','E','G','C'};
    service_time = struct('P',tP,'E',tE,'G',tG,'C',tC);
    queue_capacity = struct('P',6,'E',6,'G',6,'C',8); % <-- new capacity limits
    next_free = struct('P',0,'E',0,'G',0,'C',0);

    % Initialize station records
    for s = stations
        station_records.(s{1}).arrivals = [];
        station_records.(s{1}).start_times = [];
    end

    % Track current queue at each station
    station_queue = struct('P',[],'E',[],'G',[],'C',[]);

    % Generate food choices only for stations in path
    food_choice = zeros(N,4);
    for i = 1:N
        p = paths{i};
        if contains(p,'P'); food_choice(i,1) = rand() < P_pizza; end
        if contains(p,'E'); food_choice(i,2) = rand() < P_entree; end
        if contains(p,'G'); food_choice(i,3) = rand() < P_grill; end
        if contains(p,'C'); food_choice(i,4) = rand() < P_corner; end
    end

    [~, order] = sort(checkin_end_times);

    for k = 1:N
        i = order(k);
        path = paths{i};
        st = struct();
        cur_time = checkin_end_times(i);
        last_station = '';

        for j = 1:numel(path)
            station = path(j);
            idx = find(strcmp(stations, station));
            wants_food = food_choice(i,idx);

            % Travel time
            if isempty(last_station)
                travel = tt_checkin_to_station.(station);
            else
                travel = tt_station_to_station.(last_station).(station);
            end
            t_arr = cur_time + travel;

            % ---------------- Queue capacity handling ----------------
            if wants_food
                % Clean up the queue by removing people who already started service
                station_queue.(station)(station_queue.(station) <= t_arr) = [];

                % If queue is full, wait until a spot is free
                if numel(station_queue.(station)) >= queue_capacity.(station)
                    t_arr = max(t_arr, station_queue.(station)(1)); % wait for first person to leave
                end
            end

            % Queue handling
            s_start = max(t_arr, next_free.(station));

            % Service duration
            duration = service_time.(station) * wants_food;
            s_end = s_start + duration;

            % Update station records
            if wants_food
                station_queue.(station) = [station_queue.(station), s_end]; % student occupies queue until start
            end
            station_records.(station).arrivals(end+1,1) = t_arr;
            station_records.(station).start_times(end+1,1) = s_start;

            % Timeline entry
            if wants_food
                st.(station) = [s_start, s_end];
            else
                st.(station) = [NaN, NaN];
            end

            next_free.(station) = max(next_free.(station), s_end);
            cur_time = s_end;
            last_station = station;
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