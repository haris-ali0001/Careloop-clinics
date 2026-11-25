-- Part 2 Queries - CareLoop Clinics

-- Q1: Total appointments and no-show rate
SELECT COUNT(*) AS total,
SUM(CASE WHEN status = 'no-show' THEN 1 ELSE 0 END) AS no_shows,
ROUND(100.0 * SUM(CASE WHEN status = 'no-show' THEN 1 ELSE 0 END) / COUNT(*), 2) AS no_show_pct
FROM appointments;

-- Q2: No-show rate by clinic
SELECT clinic_id, COUNT(*) AS total,
ROUND(100.0 * SUM(CASE WHEN status = 'no-show' THEN 1 ELSE 0 END) / COUNT(*), 2) AS no_show_pct
FROM appointments GROUP BY clinic_id ORDER BY no_show_pct DESC;

-- Q3: No-show rate by reminder_sent
SELECT reminder_sent, COUNT(*) AS total,
ROUND(100.0 * SUM(CASE WHEN status = 'no-show' THEN 1 ELSE 0 END) / COUNT(*), 2) AS no_show_pct
FROM appointments GROUP BY reminder_sent;

-- Q4: No-show rate by channel
SELECT channel, COUNT(*) AS total,
ROUND(100.0 * SUM(CASE WHEN status = 'no-show' THEN 1 ELSE 0 END) / COUNT(*), 2) AS no_show_pct
FROM appointments GROUP BY channel;

-- Q5: Average distance for show vs no-show
SELECT status, ROUND(AVG(distance_km), 2) AS avg_distance_km, COUNT(*) AS total
FROM appointments GROUP BY status;

-- Q6: Duplicate appointment_id check
SELECT appointment_id, COUNT(*) AS cnt
FROM appointments GROUP BY appointment_id HAVING COUNT(*) > 1;

-- Duplicate-key error test (as comment):
-- Tried to INSERT same appointment_id twice.
-- Error: duplicate key value violates unique constraint "appointments_pkey"
-- DETAIL: Key (appointment_id)=(A0001) already exists.