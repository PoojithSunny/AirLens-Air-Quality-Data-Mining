-- Run these in DB Browser for SQLite or a SQLite command-line client after pipeline completion.

-- 1. Check warehouse row counts.
SELECT 'dim_station' AS table_name, COUNT(*) AS row_count FROM dim_station
UNION ALL
SELECT 'dim_time', COUNT(*) FROM dim_time
UNION ALL
SELECT 'fact_air_quality', COUNT(*) FROM fact_air_quality
UNION ALL
SELECT 'station_month_summary', COUNT(*) FROM station_month_summary;

-- 2. Monthly mean PM2.5 by station (time drill-down / roll-up example).
SELECT s.station, t.year, t.month,
       ROUND(AVG(f."PM2.5"), 2) AS mean_pm25,
       COUNT(*) AS observations
FROM fact_air_quality AS f
JOIN dim_station AS s ON f.station_key = s.station_key
JOIN dim_time AS t ON f.time_key = t.time_key
GROUP BY s.station, t.year, t.month
ORDER BY t.year, t.month, mean_pm25 DESC;

-- 3. Overall monitoring-station summary.
SELECT s.station,
       ROUND(AVG(f."PM2.5"), 2) AS mean_pm25,
       ROUND(AVG(f."PM10"), 2) AS mean_pm10,
       ROUND(AVG(f.TEMP), 2) AS mean_temperature,
       ROUND(AVG(f.WSPM), 2) AS mean_wind_speed,
       COUNT(*) AS observations
FROM fact_air_quality AS f
JOIN dim_station AS s ON f.station_key = s.station_key
GROUP BY s.station
ORDER BY mean_pm25 DESC;

-- 4. Example seasonal roll-up from the time dimension.
SELECT t.season,
       ROUND(AVG(f."PM2.5"), 2) AS mean_pm25,
       ROUND(AVG(f."PM10"), 2) AS mean_pm10,
       COUNT(*) AS observations
FROM fact_air_quality AS f
JOIN dim_time AS t ON f.time_key = t.time_key
GROUP BY t.season
ORDER BY mean_pm25 DESC;
