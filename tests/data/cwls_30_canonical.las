~Version Information
VERS.  3.0 : CWLS LOG ASCII STANDARD - VERSION 3.0
WRAP.  NO  : ONE LINE PER RECORD FLAG
DLM .  COMMA : DELIMITING CHARACTER
~Well Information
STRT.M        1670.0000 : START DEPTH
STOP.M        1672.0000 : STOP DEPTH
STEP.M           0.5000 : STEP
NULL.         -999.2500 : NULL VALUE
WELL.       CANONICAL WELL #3 : WELL NAME
COMP.       TURBOLAS ENERGY CORP : COMPANY
FLD .       NORTH SEA : FIELD
LOC .       BLOCK 34/10-A : LOCATION
SRVC.       SCHLUMBERGER : SERVICE COMPANY
DATE.       2026-11-08 : LOG DATE
UWI .       100/01-01-001-01W4/0 : UNIQUE WELL ID
~Log_Definition | Primary Wireline Log Dataset
DEPT.M               : 1 DEPTH
GR  .GAPI            : 2 GAMMA RAY
NPHI.V/V             : 3 NEUTRON POROSITY
RHOB.K/M3            : 4 BULK DENSITY
~Log_Data | Primary Wireline Log Data
1670.0000,   45.2000,   0.2150,   2.4500
1670.5000,   48.7000,   0.2200,   2.4600
1671.0000, -999.2500,   0.2280,   2.4550
1671.5000,   52.1000,   0.2350, -999.2500
1672.0000,   55.4000, -999.2500,   2.4800
~Core_Definition | Routine Core Analysis Dataset
CDEP.M               : 1 CORE DEPTH
CPOR.%               : 2 CORE POROSITY
CPERM.MD             : 3 CORE AIR PERMEABILITY
CGD .G/C3            : 4 CORE GRAIN DENSITY
~Core_Data | Routine Core Analysis Data
1670.2000,   18.5000,  125.4000,   2.6500
1670.8000,   17.8000,   98.2000,   2.6600
1671.4000,   19.1000,  152.0000,   2.6400
~Inclinometry_Definition | Wellbore Trajectory
DEPT.M               : 1 MEASURED DEPTH
HAZI.DEG             : 2 HOLE AZIMUTH
DEVI.DEG             : 3 HOLE DEVIATION
~Inclinometry_Data | Wellbore Trajectory Data
1670.0000,  125.4000,    2.1000
1671.0000,  126.1000,    2.3000
1672.0000,  126.8000,    2.4500
~Other Information
Comments: LAS 3.0 canonical multi-section test file for TurboLAS testing.
Contains Log, Core, and Inclinometry data groups with DLM. COMMA.
