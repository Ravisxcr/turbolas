~VERSION INFORMATION
 VERS.   2.0 :   CWLS LOG ASCII STANDARD -VERSION 2.0
 WRAP.    NO :   ONE LINE PER RECORD FLAG
~WELL INFORMATION
#MNEM.UNIT       DATA                       DESCRIPTION
#---------       ----                       -----------
 STRT.FT         8500.0000                  : START DEPTH
 STOP.FT         8504.0000                  : STOP DEPTH
 STEP.FT            1.0000                  : STEP
 NULL.           -9999.00                   : CUSTOM NULL SENTINEL
 COMP.           WILDCAT EXPLORATION CORP   : COMPANY
 WELL.           WILDCAT #9                 : WELL NAME
 FLD .           PERMIAN BASIN              : FIELD
 LOC .           SEC 14 T20S R32E           : LOCATION
 SRVC.           PIONEER WIRELINE SERVICES  : SERVICE COMPANY
 DATE.           15-AUG-2025                : DATE LOGGED
 API .           42-123-45678-0000          : API NUMBER
~CURVE INFORMATION
#MNEM.UNIT       API CODE                   CURVE DESCRIPTION
#---------       --------                   -----------------
 DEPTH.FT                                   : 1 DEPTH
 GR   .GAPI                                 : 2 GAMMA RAY
 NPHI .V/V                                  : 3 NEUTRON POROSITY
 CALI .IN                                   : 4 CALIPER
 DTC  .US/F                                 : 5 COMPRESSIONAL SLOWNESS
~PARAMETER INFORMATION
 EKB .FT          3450.0000                 : ELEVATION KELLY BUSHING
 EGL .FT          3425.0000                 : ELEVATION GROUND LEVEL
 BHT .DEGF         185.0000                 : BOTTOM HOLE TEMPERATURE
~OTHER
 Custom null sentinel test dataset with non-standard -9999.00 null value.
~A
 8500.0   55.40   0.220   8.50   72.40
 8501.0 -9999.00  0.240   8.50   73.10
 8502.0   58.20 -9999.00  8.60   74.00
 8503.0 -999.25   0.260 -9999.00   NaN
 8504.0   62.00   0.280   8.70   75.20
