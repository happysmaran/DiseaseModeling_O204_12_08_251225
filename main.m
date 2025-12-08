% CONFIGURATION
NUM_STUDENTS = 100;       % Total population size for the simulation
RANDOM_SEED = 42;
SIMULATION_TIME = 10000;  % Seconds (roughly 3 hours: 10am - 1pm)
MONITOR_INTERVAL = 10;    % Check queue every 10 seconds

% Service Times (seconds)
TIME_ENTRY_MIN = 2;
TIME_ENTRY_MAX = 5;
TIME_ENTREE = 10;
TIME_GRILL = 10;
TIME_CORNER = 30;
TIME_PIZZA = 15;

% Global Station Data
STATION_KEYS = {'Entree', 'Pizza', 'Grill', 'Corner'};
STATION_TIMES = containers.Map(...
    {'Entree', 'Grill', 'Corner', 'Pizza'}, ...
    {TIME_ENTREE, TIME_GRILL, TIME_CORNER, TIME_PIZZA} ...
);

% Student Profile Class Definition (MATLAB Struct/Function)
% In MATLAB, a class is usually defined in a separate file, but I will use 
% a struct or something for simplicity here. Get wrecked Java users.

function name = clean_station_name(raw_name)
    % A simple case-insensitive cleaning function. Need to confirm if this
    % is actually correct.
    raw_name = lower(raw_name);
    if contains(raw_name, 'entree')
        name = 'Entree';
    elseif contains(raw_name, 'pizza') || contains(raw_name, 'pasta')
        name = 'Pizza';
    elseif contains(raw_name, 'grill')
        name = 'Grill';
    elseif contains(raw_name, 'corner')
        name = 'Corner';
    else
        name = 'Entree'; % Default
    end
end

function profile = create_student_profile(row)
    profile = struct();
    profile.first_station = clean_station_name(row{3});
    
    % Num Stations (Col 4)
    try
        profile.num_stations = max(1, min(4, str2double(row{4})));
    catch
        profile.num_stations = 1;
    end
    
    % Weights (Col 5-8, Pizza, Entree, Grill, Corner)
    weights = [str2double(row{5}), str2double(row{6}), str2double(row{7}), str2double(row{8})];
    if any(isnan(weights))
        weights = [1, 1, 1, 1];
    end
    
    STATION_KEYS = {'Entree', 'Pizza', 'Grill', 'Corner'};
    profile.weights = containers.Map(STATION_KEYS, num2cell(weights));
end

function profiles = load_profiles_matlab()
    % Row format: {Col1, Col2, FirstStation, NumStations, W_Pizza, W_Entree, W_Grill, W_Corner}
    dummy_data = { % Dummy data for now, the CSV reader was garbage.
        {'', '', 'Entree bar', '2', '1', '5', '1', '1'};
        {'', '', 'Pizza', '3', '3', '1', '1', '1'};
        {'', '', 'Grill', '1', '1', '1', '5', '1'};
        {'', '', 'Corner', '4', '1', '1', '1', '3'};
        {'', '', 'Entree bar', '2', '1', '5', '1', '1'};
        {'', '', 'Pizza', '3', '3', '1', '1', '1'};
        {'', '', 'Grill', '1', '1', '5', '1', '1'};
        {'', '', 'Corner', '4', '1', '1', '1', '3'};
        {'', '', 'Entree bar', '2', '1', '5', '1', '1'};
        {'', '', 'Pizza', '3', '3', '1', '1', '1'};
    };

    profiles = {};
    for i = 1:length(dummy_data)
        profiles{end+1} = create_student_profile(dummy_data{i});
    end
end

% SIMULATION CORE

function stats = run_simulation_matlab(scenario_type, profiles)
    
    % Randomness setup
    RANDOM_SEED = 42;
    rng(RANDOM_SEED);
    
    % Initialize Stats
    stats = struct('wait_times', [], 'queue_over_time', []);

    % Resource/Queue Management
    entry_queue = []; % List of student IDs waiting for Entry
    entry_busy_until = 0; % Time when the Entry Resource becomes free
    student_arrival_events = {}; % Initial list of all student arrival times
    
    % Time tracking for queue monitoring
    next_monitor_time = 0;

    % 1. Generate all Student Arrival Events
    NUM_STUDENTS = 100;
    for i = 1:NUM_STUDENTS
        profile = profiles{randi(length(profiles))}; % Select random profile
        
        % Determine Arrival Time based on Scenario
        if strcmp(scenario_type, 'Two Sessions')
            if rand() < 0.5
                arrival_time = normrnd(900, 300); % 10:15 (900s) +/- 5m
            else
                arrival_time = normrnd(7500, 300); % 12:05 (7500s) +/- 5m
            end
        else % 'One Session'
            arrival_time = normrnd(7500, 300); % 12:05 only
        end
        
        % Store initial event: {Time, Type='Arrival', StudentID, Profile}
        student_arrival_events(end+1,:) = {max(0, arrival_time), 'Arrival', i, profile};
    end
    
    % Sort events by time (essential for DES)
    [~, sorted_idx] = sort([student_arrival_events{:,1}]);
    event_list = student_arrival_events(sorted_idx, :);

    current_time = 0;
    event_idx = 1;
    
    SIMULATION_TIME = 10000;
    % 2. Main Simulation Loop (Discrete-Event Scheduler)
    while current_time < SIMULATION_TIME || ~isempty(entry_queue) || event_idx <= size(event_list, 1)
        
        % A. MONITOR QUEUE
        if current_time >= next_monitor_time
            stats.queue_over_time(end+1) = length(entry_queue);
            MONITOR_INTERVAL = 10;
            next_monitor_time = current_time + MONITOR_INTERVAL;
        end

        % B. HANDLE ARRIVALS and SERVICE COMPLETIONS
        
        next_event_time = SIMULATION_TIME + 1; % Initialize to far future
        
        % Find next Arrival time
        if event_idx <= size(event_list, 1)
            next_event_time = min(next_event_time, event_list{event_idx, 1});
        end
        
        % Find next Service Completion time (Entry Resource)
        if entry_busy_until < next_event_time
            next_event_time = entry_busy_until;
        end
        
        % If next event is beyond SIMULATION_TIME, break
        if next_event_time > SIMULATION_TIME && isempty(entry_queue) && entry_busy_until >= SIMULATION_TIME
            break;
        end
        
        % Advance time to the next event
        current_time = next_event_time;
        
        % --- C. PROCESS EVENTS AT current_time ---
        
        % C1. Process Arrivals
        while event_idx <= size(event_list, 1) && event_list{event_idx, 1} <= current_time
            arrival_time = event_list{event_idx, 1};
            student_id = event_list{event_idx, 3};
            profile = event_list{event_idx, 4};
            
            % Add to Entry Queue
            entry_queue(end+1) = student_id;
            
            % If entry resource is free, start service now
            if entry_busy_until <= current_time
                entry_busy_until = current_time;
            end
            
            event_idx = event_idx + 1;
        end
        
        % C2. Process Service Completion at Entry
        while entry_busy_until <= current_time && ~isempty(entry_queue)
            
            % Service Completion of the previous student
            if entry_busy_until < current_time && entry_busy_until ~= 0
                 % Move to food stations - not tracked in detail for queue stats
            end
            
            % Start service for the next student in queue
            served_student_id = entry_queue(1);
            entry_queue(1) = []; % Dequeue
            
            % Calculate Wait Time
            % Hmmm this is quite tricky without tracking individual arrival times in the queue.
            % APPROXIMATION: Assume the wait time is the time spent waiting for the Entry to be free.
            % A Python version would track individual wait times, but here we just track 
            % the queue size change. To accurately track wait time,
            % we would need to store {ID, ArrivalTime} in the queue. 
            
            % Service time at Entry
            service_duration = TIME_ENTRY_MIN + (TIME_ENTRY_MAX - TIME_ENTRY_MIN) * rand();

            % Update resource busy time
            entry_busy_until = current_time + service_duration;
        end
        
    end % End While Loop
end

% EXECUTION

% Load dummy profiles for now
profiles = load_profiles_matlab();

disp('RUNNING SIMULATION (MATLAB DES)');

% Scenario 1: Split
stats_split = run_simulation_matlab('Two Sessions', profiles);

% Scenario 2: Merged
stats_merged = run_simulation_matlab('One Session', profiles);

% RESULTS
% Calculate Metrics
max_q_split = max(stats_split.queue_over_time);
avg_q_split = mean(stats_split.queue_over_time);

max_q_merged = max(stats_merged.queue_over_time);
avg_q_merged = mean(stats_merged.queue_over_time);

fprintf('\nSCENARIO 1: Two Classes (10:10 & 12:00)');
fprintf('Max Queue Length: %.0f', max_q_split);
fprintf('Avg Queue Length (over time): %.2f', avg_q_split);

fprintf('\nSCENARIO 2: One Class (12:00 only)');
fprintf('Max Queue Length: %.0f', max_q_merged);
fprintf('Avg Queue Length (over time): %.2f', avg_q_merged);

fprintf('\nRESULT');
if max_q_merged > max_q_split
    increase = ((max_q_merged/max_q_split) - 1) * 100;
    fprintf('Merging the classes causes the peak queue to grow by %.1f%%.', increase);
else
    disp('The split class scenario resulted in a higher or equal peak queue length.');
end