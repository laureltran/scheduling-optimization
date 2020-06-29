from ortools.sat.python import cp_model
import collections
import datetime
import json
import math

with open('data.json') as f:
    data = json.load(f)


assignments = data['assignments']
blocked_times = data['blocked_times']
inspectors = data['inspectors']
factories = data['factories']
distances = data['distances']

def integer_to_day_hour(num_integer, within_day=True):
    num_hours_per_day = 24
    start_hour_of_day = 6
    plus_number = num_integer % num_hours_per_day
    hours = start_hour_of_day + plus_number
    day = math.floor(num_integer / num_hours_per_day)
    if plus_number == 0 and day != 0 and not within_day:
        day = day - 1
        hours = start_hour_of_day + num_hours_per_day

    date = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][day]
    string_return = date + "-" + str(hours)
    return string_return

# Ortools helper
def helper_get_bound_of_weekday(hours_per_day_model, weekday, hour_from, hour_to):
    start_bound = weekday * hours_per_day_model + hour_from
    end_bound = start_bound + hour_to - hour_from
    return start_bound, end_bound


def helper_get_weekday_from_datetime(dt):
    year, month, day = (int(x) for x in dt.split('-'))
    weekday = datetime.date(year, month, day).weekday()
    return weekday


def helper_get_list_date_of_week(date_from, date_to):
    date_range = range((date_to - date_from).days + 1)
    list_date = [(date_from + datetime.timedelta(i)).strftime('%Y-%m-%d') for i in date_range]
    return list_date


def helper_get_date_from_integer_time(weekdays, integer_time, hours_per_day, start_hour_of_day, start_integer=True):
    plus = integer_time % hours_per_day
    hours = start_hour_of_day + plus
    day_index = math.floor(integer_time / hours_per_day)

    if plus == 0 and day_index != 0 and not start_integer:
        day_index = day_index - 1
        hours = start_hour_of_day + hours_per_day

    date = weekdays[day_index]
    return date, hours


def helper_get_distance_between_point(distances_dict, measure_point, reference_point):
    """Returns the distance between tasks of job measure_point and tasks of job reference_point."""
    key_tuple = (measure_point, reference_point)
    if key_tuple not in distances_dict.keys():
        key_tuple = (reference_point, measure_point)
    hours = distances_dict[key_tuple]
    return hours

configs = {
        "factory_change": 1,
        "travel_time": 2,
        "date_change": 3,
        "load_balancing": 4
    }
num_search_workers = 12
weekdays_int = list(range(0, 5))
HOURS_PER_DAY_MODEL = 24
horizon = 5 * HOURS_PER_DAY_MODEL

print("ASSIGNMENTS ", len(assignments))
print("BLOCKED_TIMES ", len(blocked_times))
print("INSPECTORS ", len(inspectors))
print("HORIZON ", horizon)

# Tranform date to integer
for a in assignments:
    a['origin_expected_date'] = a['expected_date']
    a['expected_date'] = helper_get_weekday_from_datetime(a['expected_date'])
    if not a['shipment_date']:
        a['shipment_date'] = -1
    else:
        a['shipment_date'] = helper_get_weekday_from_datetime(a['shipment_date'])

for b in blocked_times:
    b['origin_requested_date'] = b['requested_date']
    b['requested_date'] = helper_get_weekday_from_datetime(b['requested_date'])

factories_dict = {}
for f in factories:
    factories_dict[f['factory_customer_id']] = f['manager_id']

total_distance_between_matrix = 0
distances_dict = {}
for distance_obj in distances:
    tuple_distance = (distance_obj["measure_point"], distance_obj["reference_point"])
    distances_dict[tuple_distance] = distance_obj["hours"]
    total_distance_between_matrix += distance_obj["hours"]

model = cp_model.CpModel()

assignment_type = collections.namedtuple('booking_type', 'start end interval duration bool_var')
block_type = collections.namedtuple('booking_type', 'start end interval duration')
dummy_type = collections.namedtuple('dummy_type', 'start end interval duration')
all_bookings = {}

# Model assignments
for a in assignments:
    for i in inspectors:
        if a['activity_type'] in i['preferred'] or 'General' in i['preferred']:
            label_tuple = (a['assignment_id'], i['inspector_customer_id'])
            start_var = model.NewIntVar(0, horizon, 'start_assign_%s_%s' % label_tuple)
            duration = a['activity_duration']
            end_var = model.NewIntVar(0, horizon, 'end_assign_%s_%s' % label_tuple)
            bool_var = model.NewBoolVar('bool_assign_%s_%s' % label_tuple)
            optional_interval_var = model.NewOptionalIntervalVar(
                start_var, duration, end_var, bool_var,
                'optional_interval_assign_%s_%s' % label_tuple
            )

            all_bookings[label_tuple] = assignment_type(
                start=start_var, end=end_var, interval=optional_interval_var,
                duration=duration, bool_var=bool_var
            )

# Model blocks
for b in blocked_times:
    for i in inspectors:
        if b['inspector_customer_id'] == i['inspector_customer_id']:
            label_blocked = b['blocked_id']
            start_var = model.NewIntVar(0, horizon, 'start_block_%s' % label_blocked)
            duration = b['activity_duration']
            end_var = model.NewIntVar(0, horizon, 'end_block_%s' % label_blocked)
            bool_var = model.NewBoolVar('bool_block_%s' % label_blocked)
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, 'interval_block_%s' % label_blocked
            )

            all_bookings[label_blocked] = block_type(
                start=start_var, end=end_var, interval=interval_var, duration=duration
            )

# Model dummy blocks
for w in weekdays_int:
    for i in inspectors:
        start_bound = w * HOURS_PER_DAY_MODEL

        # Dummy day
        label_dummy_day = (w, i['inspector_customer_id'], 'day')
        end_day_from = start_bound + 6
        start_day_var = model.NewIntVar(
            start_bound, end_day_from, 'start_day_dummy_%i_%s_%s' % label_dummy_day
        )
        end_day_var = model.NewIntVar(
            start_bound, end_day_from, 'end_day_dummy_%i_%s_%s' % label_dummy_day
        )
        duration_day = 6
        day_interval_var = model.NewIntervalVar(
            start_day_var, duration_day, end_day_var, 'interval_day_dummy_%i_%s_%s' % label_dummy_day
        )
        all_bookings[label_dummy_day] = dummy_type(
            start=start_day_var, end=end_day_var, interval=day_interval_var, duration=duration_day
        )

        # Dummy night
        label_dummy_night = (w, i['inspector_customer_id'], 'night')
        start_night_from = start_bound + 18
        end_night_from = (w + 1) * HOURS_PER_DAY_MODEL
        start_night_var = model.NewIntVar(
            start_night_from, end_night_from, 'start_night_dummy_%i_%s_%s' % label_dummy_night
        )
        end_night_var = model.NewIntVar(
            start_night_from, end_night_from, 'end_night_dummy_%i_%s_%s' % label_dummy_night
        )
        duration_night = HOURS_PER_DAY_MODEL - 18
        night_interval_var = model.NewIntervalVar(
            start_night_var, duration_night, end_night_var, 'interval_night_dummy_%i_%s_%s' % label_dummy_night
        )
        all_bookings[label_dummy_night] = dummy_type(
            start=start_night_var, end=end_night_var, interval=night_interval_var, duration=duration_night
        )

# CONSTRAINT AND OBJECTIVE FORMULATION
# Each assignment execute by only 1 inspectors & load balancing | factory mapping objective formulation
diff_of_vector_balancing = []
avg_inspection_of_inspectors = int(len(assignments) / len(inspectors))
max_diff_balancing_integer = len(assignments) - avg_inspection_of_inspectors
max_diff_balancing_var = model.NewIntVar(0, max_diff_balancing_integer, 'max_diff_balancing')
bools_factory_mapping = []
for a in assignments:
    bool_assignments = []
    for i in inspectors:
        if a['activity_type'] in i['preferred'] or 'General' in i['preferred']:
            label_tuple = (a['assignment_id'], i['inspector_customer_id'])
            bool_assignments.append(all_bookings[label_tuple].bool_var)
            if factories_dict[a['factory_customer_id']] == i['inspector_customer_id']:
                bools_factory_mapping.append(all_bookings[label_tuple].bool_var.Not())
    model.Add(sum(bool_assignments) == 1)

    # Model load balancing objective
    diff_var = model.NewIntVar(-avg_inspection_of_inspectors, max_diff_balancing_integer, 'diff_with_avg_%s' % a['assignment_id'])
    model.Add(diff_var == sum(bool_assignments) - avg_inspection_of_inspectors)
    abs_diff_of_balancing_var = model.NewIntVar(0, max_diff_balancing_integer, 'abs_diff_with_avg_%s' % a['assignment_id'])
    model.AddAbsEquality(abs_diff_of_balancing_var, diff_var)
    diff_of_vector_balancing.append(abs_diff_of_balancing_var)

model.AddMaxEquality(max_diff_balancing_var, diff_of_vector_balancing)

# Data change objective
abs_integer_dates_distance = []
# Travel time objective
switch_transit_literals = []
switch_transition_times = []
# Distance objective
total_avg_distances = 0
for i in inspectors:
    intervals = []
    executor_starts = []
    executor_ends = []
    executor_bools = []
    executor_intervals = []
    location_ids_mapping = []

    distance_i_to_fs = 0
    # NoOverLap contraint for assignments
    for a in assignments:
        if a['activity_type'] in i['preferred'] or 'General' in i['preferred']:
            label_tuple = (a['assignment_id'], i['inspector_customer_id'])
            booking = all_bookings[label_tuple]

            # Add to list for NoOverlap constraint
            intervals.append(booking.interval)
            # Add to executor node for dense graph
            executor_intervals.append(booking.interval)

            # Add to executor node for dense graph distance reference
            executor_starts.append(booking.start)
            executor_ends.append(booking.end)

            # Add to executor bool for dense graph node reference
            executor_bools.append(booking.bool_var)

            # Add to executor bool for dense graph location mapping
            location_ids_mapping.append(a['factory_customer_id'])

            # Booking must happens before shipment date (deadline)
            if a['shipment_date'] >= 0:
                deadline_start_bound, deadline_end_bound = helper_get_bound_of_weekday(
                    HOURS_PER_DAY_MODEL, a['shipment_date'], 6, 18
                )
                model.Add(booking.end <= deadline_end_bound)

            # Model variable for date change objective
            integer_date_of_assignment_var = model.NewIntVar(
                0, weekdays_int[-1], 'integer_date_assignment_%s_%s' % label_tuple
            )
            integer_dates_distance_var = model.NewIntVar(
                -weekdays_int[-1], weekdays_int[-1], 'integer_date_distance_%s_%s' % label_tuple
            )
            abs_distance_var = model.NewIntVar(
                0, weekdays_int[-1], 'integer_date_distance_abs_%s_%s' % label_tuple
            )
            model.AddDivisionEquality(integer_date_of_assignment_var, booking.start, HOURS_PER_DAY_MODEL)
            model.Add(integer_date_of_assignment_var - a['expected_date'] == integer_dates_distance_var)
            model.AddAbsEquality(abs_distance_var, integer_dates_distance_var)
            abs_integer_dates_distance.append(abs_distance_var)

            # Traveling time object avg
            distance = helper_get_distance_between_point(
                distances_dict, i['inspector_customer_id'], a['factory_customer_id']
            )
            distance_i_to_fs += distance

    avg_distance_from_i_to_fs = int(distance_i_to_fs / len(inspectors))
    total_avg_distances += avg_distance_from_i_to_fs

    for b in blocked_times:
        if b['inspector_customer_id'] == i['inspector_customer_id']:
            label_blocked = b['blocked_id']
            booking = all_bookings[label_blocked]

            # Add to list for NoOverlap constraint
            intervals.append(booking.interval)

            # Booking must happens in requested_date date
            block_start_bound, block_end_bound = helper_get_bound_of_weekday(
                HOURS_PER_DAY_MODEL, b['requested_date'], 6, 18
            )
            model.Add(all_bookings[label_blocked].start >= block_start_bound)
            model.Add(all_bookings[label_blocked].end <= block_end_bound)

    dummy_bools = []
    for w in weekdays_int:
        label_dummy_day = (w, i['inspector_customer_id'], 'day')
        booking_day = all_bookings[label_dummy_day]

        label_dummy_night = (w, i['inspector_customer_id'], 'night')
        booking_night = all_bookings[label_dummy_night]

        # Add to list for NoOverlap constraint
        intervals.append(booking_day.interval)
        intervals.append(booking_night.interval)
        # Add to executor node for dense graph
        executor_intervals.append(booking_day.interval)
        executor_intervals.append(booking_night.interval)

        bool_day = model.NewBoolVar('day_dummy_%i_%s_%s' % label_dummy_day)
        bool_night = model.NewBoolVar('night_dummy_%i_%s_%s' % label_dummy_night)
        # Add to bools list to indicate successor of dense graph
        dummy_bools.append(bool_day)
        dummy_bools.append(bool_night)
        # Add to executor bool for dense graph node reference
        executor_bools.append(bool_day)
        executor_bools.append(bool_night)
        # Add to executor bool for dense graph comparing distance
        executor_starts.append(booking_day.start)
        executor_starts.append(booking_night.start)
        executor_ends.append(booking_day.end)
        executor_ends.append(booking_night.end)

        location_ids_mapping.append(i['inspector_customer_id'])
        location_ids_mapping.append(i['inspector_customer_id'])

    # Enable to True all of dummy block
    model.Add(sum(dummy_bools) == len(weekdays_int * 2))

    # Non overlap all tasks
    model.AddNoOverlap(intervals)

    # Model Distance and Objectives: travel time - factory change
    arcs = []
    for idx_i, a_i in enumerate(executor_intervals):
        # dummy node of CIRCUIT
        start_literal = model.NewBoolVar('%i_first_job' % idx_i)
        end_literal = model.NewBoolVar('%i_last_job' % idx_i)
        arcs.append([0, idx_i + 1, start_literal])
        arcs.append([idx_i + 1, 0, end_literal])
        # Self arc if the assignment is not performed.
        arcs.append([idx_i + 1, idx_i + 1, executor_bools[idx_i].Not()])
        i_point = location_ids_mapping[idx_i]

        for idx_j, a_j in enumerate(executor_intervals):
            if idx_i == idx_j:
                continue

            literal = model.NewBoolVar('%i_follows_%i' % (idx_j, idx_i))
            arcs.append([idx_i + 1, idx_j + 1, literal])
            model.AddImplication(literal, executor_bools[idx_i])
            model.AddImplication(literal, executor_bools[idx_j])

            j_point = location_ids_mapping[idx_j]
            # Constraint distance if j is successor of i
            if i_point != j_point:
                # Constraint distance factory <-> factory
                i_to_j_distance = helper_get_distance_between_point(
                    distances_dict, i_point, j_point
                )

                # Add to objective for factory change and transition times
                switch_transit_literals.append(literal)
                switch_transition_times.append(i_to_j_distance)
            else:
                i_to_j_distance = 0

            # Reified transition to link the literals with the times
            model.Add(
                executor_starts[idx_j] >= executor_ends[idx_i] + i_to_j_distance
            ).OnlyEnforceIf(literal)

    model.AddCircuit(arcs)

# Objectives
# Modeling normalize traveling time
result_traveling_time_var = model.NewIntVar(0, total_distance_between_matrix, 'result_traveling_time')
traveling_time_objective_var = model.NewIntVar(0, total_distance_between_matrix, 'traveling_time_objective')
model.Add(sum([
    s * switch_transition_times[idx] for idx, s in enumerate(switch_transit_literals)
]) == result_traveling_time_var)
model.AddDivisionEquality(
    traveling_time_objective_var, result_traveling_time_var, total_avg_distances
)

# Model multi-objectives
weights = [5 - val for key, val in configs.items()]
objectives = [
    sum(bools_factory_mapping), 
    traveling_time_objective_var,
    sum(abs_integer_dates_distance),
    max_diff_balancing_var
]

model.Minimize(sum([w * objectives[idx] for idx, w in enumerate(weights)]))

# Solve problem with model
solver = cp_model.CpSolver()
solver.parameters.num_search_workers = num_search_workers
solver.parameters.log_search_progress = True
solver.parameters.max_time_in_seconds = 24 * 60 * 60
status = solver.Solve(model)

print('  - status          : %s' % solver.StatusName(status))
print('  - conflicts       : %i' % solver.NumConflicts())
print('  - branches        : %i' % solver.NumBranches())
print('  - wall time       : %f s' % solver.WallTime())
print('  - Objective       : %f s' % solver.ObjectiveValue())

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for i in inspectors:
        print("Inspector %s" % i['inspector_customer_id'])
        for a in assignments:
            if a['activity_type'] in i['preferred'] or 'General' in i['preferred']:
                label_tuple = (a['assignment_id'], i['inspector_customer_id'])
                if solver.BooleanValue(all_bookings[label_tuple].bool_var):
                    
                    name_start = 'start_%s_%s' % (a['factory_customer_id'], a['assignment_id'])
                    name_end = 'end_%s' %(a['factory_customer_id'])


                    value_start = integer_to_day_hour(solver.Value(all_bookings[label_tuple].start), True)  + "-" + str(solver.Value(all_bookings[label_tuple].start))
                    value_end = integer_to_day_hour(solver.Value(all_bookings[label_tuple].end), False)  + "-" + str(solver.Value(all_bookings[label_tuple].end))
                    print(name_start + ": " + value_start)
                    print(name_end + ": " + value_end)

        for b in blocked_times:
            if b['inspector_customer_id'] == i['inspector_customer_id']:
                label_blocked = b['blocked_id']
                name_start = 'start_block_%s' % label_blocked
                name_end = 'end_block_%s' % label_blocked
                value_start = integer_to_day_hour(solver.Value(all_bookings[label_blocked].start), True) + "-" + str(solver.Value(all_bookings[label_blocked].start))
                value_end = integer_to_day_hour(solver.Value(all_bookings[label_blocked].end), False) + "-" + str(solver.Value(all_bookings[label_blocked].end))
                print(name_start + ": " + value_start)
                print(name_end + ": " + value_end)

        print("")
        print("")
