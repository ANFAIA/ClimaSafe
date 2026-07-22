# Heat Index — ecuación de Rothfusz (1990)

> **Dominio público** (obra del gobierno de EE. UU., NOAA/National Weather
> Service — no sujeta a copyright). Fuente:
> Rothfusz LP. *The Heat Index Equation.* NWS Technical Attachment SR 90-23,
> Scientific Services Division, NWS Southern Region, 1990.
> Reproducido de: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

## Ecuación de regresión principal (input en °F, RH en %)

```
HI = -42.379
     + 2.04901523·T
     + 10.14333127·RH
     - 0.22475541·T·RH
     - 0.00683783·T·T
     - 0.05481717·RH·RH
     + 0.00122874·T·T·RH
     + 0.00085282·T·RH·RH
     - 0.00000199·T·T·RH·RH
```

- `T` = temperatura del aire en °F
- `RH` = humedad relativa en % (0–100)
- `HI` = heat index (temperatura aparente) en °F

## Ajustes y casos especiales

**Humedad baja** — si RH < 13 % y 80 °F ≤ T ≤ 112 °F, restar:
```
AJUSTE = [(13-RH)/4] · SQRT{[17-ABS(T-95)]/17}
```

**Humedad alta** — si RH > 85 % y 80 °F ≤ T ≤ 87 °F, sumar:
```
AJUSTE = [(RH-85)/10] · [(87-T)/5]
```

**Por debajo de ~80 °F** (la ecuación completa no es fiable) — fórmula simple:
```
HI = 0.5 · {T + 61.0 + [(T-68.0)·1.2] + (RH·0.094)}
```

## Uso en el proyecto

Es la fórmula base de `heat_index_c` (ver `formulas_deterministas.md`
§1.1 y `climasafeai/features/weather_indices.py`). El proyecto trabaja en °C:
convierte `T(°F)=T(°C)·9/5+32` antes y `T(°C)=(HI(°F)-32)·5/9` después.
