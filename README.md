# Projectv2 Build and Run Commands

This folder contains three versions of the N-body simulation:
- `main_serial.cpp` (serial)
- `main.cpp` (OpenMP)
- `main_mpi.cpp` (MPI)

All executables now support runtime arguments so you can benchmark with different parameters.

## Prerequisites (Windows)

- MinGW g++ available in PATH
- OpenMP support in g++
- Microsoft MPI runtime and SDK installed
- PowerShell terminal

## Go to project folder

```powershell
Set-Location "C:/Users/KURUV PATEL/OneDrive/Documents/LAB/HPC Lab/Projectv2"
```

## 1) Serial build and run

Build:

```powershell
g++ -O2 main_serial.cpp -o nbody_serial.exe
```

Run:

```powershell
./nbody_serial.exe 1000 1000 0
```

Output file:
- Optional. Third arg controls CSV write: `0` (off), `1` (on)

## 2) OpenMP build and run

Build:

```powershell
g++ -O2 -fopenmp main.cpp -o nbody_openmp.exe
```

Run:

```powershell
./nbody_openmp.exe 1000 1000 8 0
```

Output file:
- Optional. Fourth arg controls CSV write: `0` (off), `1` (on)

## 3) MPI build and run (MS-MPI on Windows)

`mpic++` is not available on this machine, so compile with g++ plus MS-MPI include/lib paths.

Build:

```powershell
g++ -O2 main_mpi.cpp -o nbody_mpi.exe -I"$env:MSMPI_INC" -L"$env:MSMPI_LIB64" -lmsmpi
```

Run with 4 processes:

```powershell
mpiexec -n 4 ./nbody_mpi.exe 1000 1000 0
```

Output file:
- Optional. Third arg controls CSV write: `0` (off), `1` (on)

## Quick performance test

Run each executable and compare printed times:

```powershell
./nbody_serial.exe 1000 1000 0
./nbody_openmp.exe 1000 1000 8 0
mpiexec -n 4 ./nbody_mpi.exe 1000 1000 0
```

## 4) Streamlit benchmark app (Enhanced UI)

Install dependencies:

```powershell
pip install streamlit pandas matplotlib
```

Run app:

```powershell
streamlit run streamlit_app.py
```

### Features:
- **📊 Results Tab**: Run benchmarks, view aggregate statistics (mean, std dev, min/max), compare execution times
- **📈 History Tab**: Track previous benchmark runs with full details and summaries
- **📋 Logs Tab**: View build and execution logs for debugging
- **Multiple Runs**: Average results across 1-10 runs for more stable metrics
- **Speedup Metrics**: Calculate actual speedup vs serial baseline
- **Data Export**: Download results as CSV or JSON
- **Charts**: Time comparison and speedup visualization
- **Session State**: Persistent run history during app session

### Configuration:
- Adjust N (bodies), simulation steps, OpenMP threads, and MPI processes from sidebar
- Option to write CSV outputs (disabled by default for fair timing)
- Control number of averaging runs for measurement stability
