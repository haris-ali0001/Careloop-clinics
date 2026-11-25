CREATE TABLE IF NOT EXISTS clinics (
  clinic_id VARCHAR(20) PRIMARY KEY,
  clinic_name VARCHAR(100) NOT NULL,
  city VARCHAR(50),
  latitude DECIMAL(9,6),
  longitude DECIMAL(9,6),
  temp_c DECIMAL(5,2)
);

CREATE TABLE IF NOT EXISTS appointments (
  appointment_id VARCHAR(20) PRIMARY KEY,
  patient_id VARCHAR(20) NOT NULL,
  clinic_id VARCHAR(20),
  booked_on DATE,
  appt_date DATE,
  age INT,
  gender VARCHAR(20),
  distance_km DECIMAL(6,2),
  channel VARCHAR(20),
  fee DECIMAL(8,2),
  reminder_sent VARCHAR(10),
  prior_no_shows INT,
  status VARCHAR(20),
  lead_days INT,
  no_show INT,
  temperature_c DECIMAL(5,2),
  FOREIGN KEY (clinic_id) REFERENCES clinics(clinic_id)
);
