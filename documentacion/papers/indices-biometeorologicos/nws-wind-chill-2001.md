# Wind Chill — ecuación NWS (2001)

> **Dominio público** (obra del gobierno de EE. UU., NOAA/National Weather
> Service — no sujeta a copyright). Fuente:
> National Weather Service, *Wind Chill Temperature Index* (vigente desde
> 01/11/2001). Reproducido de:
> https://www.weather.gov/safety/cold-wind-chill-chart

## Fórmula oficial (unidades imperiales, °F y mph)

```
WC(°F) = 35.74 + 0.6215·T - 35.75·V^0.16 + 0.4275·T·V^0.16
```

- `T` = temperatura del aire en °F
- `V` = velocidad del viento en mph
- Definida solo para **T ≤ 50 °F** y **V > 3 mph**.

## Versión métrica (°C y km/h) — la que usa el proyecto

```
WC(°C) = 13.12 + 0.6215·T - 11.37·V^0.16 + 0.3965·T·V^0.16
```

- Válida para **T ≤ 10 °C** y **V > 4.8 km/h**. Fuera de rango el efecto del
  viento no es significativo → se usa la temperatura real.

## Tiempo hasta congelación (frostbite)

**Importante:** el tiempo de congelación NO es función limpia del Wind Chill ya
calculado, sino de la combinación original T×V — dos pares (T, V) distintos
pueden dar el mismo WC pero tiempos distintos. Ejemplo NWS: a WC −19 °F (0 °F +
15 mph) la piel expuesta se congela en ~30 min. La rejilla oficial T×V está en
https://www.weather.gov/media/safety/windchillchart3.pdf

## Uso en el proyecto

Es la fórmula base de `wind_chill_c` (ver `formulas_deterministas.md`
§2.1 y `climasafeai/features/weather_indices.py`).
