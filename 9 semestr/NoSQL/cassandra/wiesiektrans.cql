CREATE KEYSPACE IF NOT EXISTS wiesiektrans
WITH replication = {
  'class': 'SimpleStrategy',
  'replication_factor': 1
};

USE wiesiektrans;

CREATE TABLE IF NOT EXISTS drivers (
    driver_name text PRIMARY KEY,
    password text,
    phone_number text,
    skills set<text>
);

CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id uuid PRIMARY KEY,
    vehicle_type text,
    model text,
    max_speed int
);

CREATE TABLE IF NOT EXISTS vehicle_current_position (
    vehicle_id uuid PRIMARY KEY,
    latitude double,
    longitude double,
    speed double,
    last_update timestamp
);

CREATE TABLE IF NOT EXISTS vehicle_position_history (
    vehicle_id uuid,
    date text,
    event_time timestamp,
    latitude double,
    longitude double,
    speed double,
    PRIMARY KEY ((vehicle_id, date), event_time)
) WITH CLUSTERING ORDER BY (event_time DESC);

CREATE TABLE IF NOT EXISTS driver_availability (
    driver_name text,
    date text,
    is_available boolean,
    PRIMARY KEY ((driver_name), date)
);

CREATE TABLE IF NOT EXISTS vehicle_availability (
    vehicle_id uuid,
    date text,
    is_available boolean,
    PRIMARY KEY ((vehicle_id), date)
);

CREATE TABLE IF NOT EXISTS schedules (
    vehicle_id uuid,
    departure_time timestamp,
    arrival_time timestamp,
    from_station text,
    to_station text,
    driver_name text,
    PRIMARY KEY ((vehicle_id), departure_time)
);

INSERT INTO drivers (driver_name, password, phone_number, skills)
VALUES (
  'jan_kowalski',
  'hashed_pass',
  '600700800',
  {'lokomotywa_elektryczna', 'lokomotywa_spalinowa'}
);

INSERT INTO vehicles (vehicle_id, vehicle_type, model, max_speed)
VALUES (
  1,
  'lokomotywa_elektryczna',
  'EU07',
  125
);

INSERT INTO vehicle_current_position (
  vehicle_id, latitude, longitude, speed, last_update
)
VALUES (
  1,
  52.2297,
  21.0122,
  80.5,
  toTimestamp(now())
);
