# Scheduling optimization with Google Ortools
Given list of inspectors need to inspect with many factories in diffrent location.
Each person has diffent shift hours, leaving day, living in different place.
Scheduling a timetable for each person meet the constraint.
More over, optimize as much as possible for them like: reduce travel time, spread tasks to make balance for people ...

## Constraint
- Allow person on leave on pre-defined day, other person will replace this shift
- Each tasks required specific skills, only few person can take up this task.
- Some person live near factories, some far. Travel time must included in working hour.

## Objective
- Minimize travel time.
- Balance work between person.
