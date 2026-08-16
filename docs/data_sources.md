# Phase 1 Data Sources

## 1. Bureau of Transportation Statistics (BTS)

**Dataset:** Reporting Carrier On-Time Performance

Use individual domestic flight records. Relevant fields include scheduled and actual departure/arrival times, origin/destination, carrier, cancellation/diversion flags, and delay causes.

Initial scope: flights where `Origin == OGG` or `Dest == OGG`.

Historical availability: reporting-carrier on-time data extend back to 1987; the current BTS table includes data through 2026.

## 2. NOAA/NCEI Local Climatological Data (LCD)

Use airport-station hourly observations for weather conditions near OGG. Candidate variables include temperature, dew point, relative humidity, station pressure, visibility, precipitation, wind speed, wind direction, gusts, sky condition, and weather type.

LCDv2 covers airport and other prominent weather stations and provides hourly observations. The current version covers 2005 onward.

## 3. NOAA Storm Events Database

Use event records as incident-level context for major weather episodes affecting Hawaii/Maui. Relevant fields include event type, start/end times, location, magnitude, damage, injuries/fatalities, and narrative text.

This source is not a replacement for hourly weather observations; it is an additional event-context layer.

## 4. AviationWeather.gov API

Use for live/recent METAR and aviation-weather features in the future production app. The public weather database currently provides roughly the previous 15 days, so it is not the main historical training source.

## Initial modeling unit

One row = one scheduled flight departing from or arriving at OGG.

## Initial target

- `normal`: completed, arrival delay < 15 minutes
- `delay`: completed, arrival delay 15–179 minutes
- `severe_delay`: completed, arrival delay >= 180 minutes
- `cancelled`: cancelled

Diversions are preserved separately for analysis and possible later inclusion as a fifth class.

## Join strategy

1. Normalize flight schedule timestamps to Hawaii local time.
2. Normalize NOAA station timestamps.
3. Match each flight to the nearest prior observation and generate rolling weather summaries for the preceding 1h, 3h, and 6h.
4. Add severe-weather incident flags from NOAA Storm Events.
5. Add airport congestion, airline behavior, and inbound-aircraft features in later phases.
