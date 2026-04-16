import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import json

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="N-Body Timing Benchmark", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent


# Initialize session state
if "run_history" not in st.session_state:
    st.session_state.run_history = []
if "build_logs" not in st.session_state:
    st.session_state.build_logs = {}
if "last_run_logs" not in st.session_state:
    st.session_state.last_run_logs = {}


def parse_time_seconds(output: str) -> float:
    match = re.search(r"TIME_SECONDS:\s*([0-9]+(?:\.[0-9]+)?)", output)
    if match:
        return float(match.group(1))
    raise ValueError("TIME_SECONDS line not found in output")


def run_command(command: List[str], cwd: Path) -> Tuple[int, str, str]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def build_binaries() -> Dict[str, str]:
    logs: Dict[str, str] = {}

    serial_cmd = ["g++", "-O2", "main_serial.cpp", "-o", "nbody_serial.exe"]
    openmp_cmd = ["g++", "-O2", "-fopenmp", "main.cpp", "-o", "nbody_openmp.exe"]

    msmpi_inc = os.environ.get("MSMPI_INC", "")
    msmpi_lib64 = os.environ.get("MSMPI_LIB64", "")
    mpi_cmd = ["g++", "-O2", "main_mpi.cpp", "-o", "nbody_mpi.exe"]
    if msmpi_inc and msmpi_lib64:
        mpi_cmd.extend([f"-I{msmpi_inc}", f"-L{msmpi_lib64}", "-lmsmpi"])

    commands = {
        "Serial": serial_cmd,
        "OpenMP": openmp_cmd,
        "MPI": mpi_cmd,
    }

    for name, cmd in commands.items():
        code, out, err = run_command(cmd, ROOT)
        logs[name] = f"Command: {' '.join(cmd)}\nExit: {code}\nSTDOUT:\n{out}\nSTDERR:\n{err}"

    return logs


def benchmark(N: int, steps: int, omp_threads: int, mpi_processes: int, write_output: bool) -> Tuple[pd.DataFrame, Dict[str, str]]:
    logs: Dict[str, str] = {}
    rows: List[Dict[str, object]] = []
    write_flag = "1" if write_output else "0"

    runs = {
        "Serial": [str(ROOT / "nbody_serial.exe"), str(N), str(steps), write_flag],
        "OpenMP": [str(ROOT / "nbody_openmp.exe"), str(N), str(steps), str(omp_threads), write_flag],
        "MPI": ["mpiexec", "-n", str(mpi_processes), str(ROOT / "nbody_mpi.exe"), str(N), str(steps), write_flag],
    }

    for strategy, cmd in runs.items():
        code, out, err = run_command(cmd, ROOT)
        logs[strategy] = f"Command: {' '.join(cmd)}\nExit: {code}\nSTDOUT:\n{out}\nSTDERR:\n{err}"

        if code != 0:
            rows.append({"Strategy": strategy, "Time (s)": None, "Status": "Failed"})
            continue

        try:
            t = parse_time_seconds(out)
            rows.append({"Strategy": strategy, "Time (s)": t, "Status": "OK"})
        except ValueError:
            rows.append({"Strategy": strategy, "Time (s)": None, "Status": "No TIME_SECONDS output"})

    df = pd.DataFrame(rows)

    serial_time = df.loc[df["Strategy"] == "Serial", "Time (s)"].dropna()
    if not serial_time.empty:
        serial_value = float(serial_time.iloc[0])
        speedups = []
        for _, row in df.iterrows():
            t = row["Time (s)"]
            if pd.notna(t) and float(t) > 0:
                speedups.append(serial_value / float(t))
            else:
                speedups.append(None)
        df["Speedup"] = speedups
    else:
        df["Speedup"] = None

    return df, logs


st.title("🚀 N-Body Performance Benchmark")
st.markdown(
    "Measure and compare Serial, OpenMP, and MPI execution times with customizable parameters."
)

with st.sidebar:
    st.header("⚙️ Benchmark Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Bodies (N)", min_value=10, max_value=20000, value=1000, step=50)
        omp_threads = st.number_input("OpenMP threads", min_value=1, max_value=128, value=8, step=1)
    with col2:
        steps = st.number_input("Simulation steps", min_value=1, max_value=200000, value=1000, step=100)
        mpi_processes = st.number_input("MPI processes", min_value=1, max_value=64, value=4, step=1)
    
    st.divider()
    
    num_runs = st.slider("Number of runs (for averaging)", min_value=1, max_value=10, value=1)
    write_output = st.checkbox("Write CSV outputs", value=False,
                              help="Disable for fair timing (file I/O overhead can dominate)")
    
    st.divider()
    st.subheader("Build & Test")
    
    build_col, run_col = st.columns(2)
    with build_col:
        build_clicked = st.button("🔨 Build", use_container_width=True)
    with run_col:
        run_clicked = st.button("⚡ Run", type="primary", use_container_width=True)

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 History", "📋 Logs"])

with tab1:
    if run_clicked:
        missing = [
            name
            for name, exe in {
                "Serial": ROOT / "nbody_serial.exe",
                "OpenMP": ROOT / "nbody_openmp.exe",
                "MPI": ROOT / "nbody_mpi.exe",
            }.items()
            if not exe.exists()
        ]

        if missing:
            st.error("Missing executable(s): " + ", ".join(missing) + ". Build first.")
        else:
            with st.spinner(f"Running {num_runs} benchmark round(s)..."):
                all_results = []
                run_details = []
                
                for run_idx in range(num_runs):
                    progress_text = f"Round {run_idx + 1}/{num_runs}"
                    st.progress((run_idx + 1) / num_runs, text=progress_text)
                    
                    results_df, run_logs = benchmark(int(N), int(steps), int(omp_threads), int(mpi_processes), write_output)
                    all_results.append(results_df)
                    
                    run_details.append({
                        "run_num": run_idx + 1,
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "N": int(N),
                        "steps": int(steps),
                        "omp_threads": int(omp_threads),
                        "mpi_processes": int(mpi_processes),
                        "results": results_df.to_dict(),
                        "logs": run_logs,
                    })
                    
                    if run_idx == num_runs - 1:
                        st.session_state.last_run_logs = run_logs
                
                st.session_state.run_history.append({
                    "config": {
                        "N": int(N),
                        "steps": int(steps),
                        "omp_threads": int(omp_threads),
                        "mpi_processes": int(mpi_processes),
                        "num_runs": num_runs,
                    },
                    "run_details": run_details,
                })
                
                # Aggregate results across runs
                combined_df = pd.concat(all_results, ignore_index=True)
                summary_df = combined_df.groupby("Strategy")[["Time (s)"]].agg(["mean", "std", "min", "max",  "count"]).round(6)
                summary_df.columns = ["Mean (s)", "Std Dev (s)", "Min (s)", "Max (s)", "Count"]
                summary_df = summary_df.reset_index()
                
                # Calculate speedups
                serial_mean = summary_df[summary_df["Strategy"] == "Serial"]["Mean (s)"].values
                if len(serial_mean) > 0 and serial_mean[0] > 0:
                    speedups = []
                    for _, row in summary_df.iterrows():
                        speedups.append(serial_mean[0] / row["Mean (s)"])
                    summary_df["Speedup"] = speedups
                
                st.success(f"✅ Completed {num_runs} run(s)!")
                
                st.subheader("📊 Performance Summary")
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                serial_row = summary_df[summary_df["Strategy"] == "Serial"]
                if not serial_row.empty:
                    col1.metric("Serial Time", f"{serial_row['Mean (s)'].values[0]:.6f}s")
                
                openmp_row = summary_df[summary_df["Strategy"] == "OpenMP"]
                if not openmp_row.empty:
                    omp_time = openmp_row['Mean (s)'].values[0]
                    omp_speedup = openmp_row['Speedup'].values[0] if "Speedup" in openmp_row else 1.0
                    col2.metric("OpenMP Time", f"{omp_time:.6f}s", delta=f"{omp_speedup:.2f}x faster")
                
                mpi_row = summary_df[summary_df["Strategy"] == "MPI"]
                if not mpi_row.empty:
                    mpi_time = mpi_row['Mean (s)'].values[0]
                    mpi_speedup = mpi_row['Speedup'].values[0] if "Speedup" in mpi_row else 1.0
                    col3.metric("MPI Time", f"{mpi_time:.6f}s", delta=f"{mpi_speedup:.2f}x faster")
                
                col4.metric("Runs Averaged", num_runs)
                
                st.divider()
                st.dataframe(summary_df.set_index("Strategy"), use_container_width=True)
                
                # Visualizations
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("Execution Time")
                    time_chart = summary_df.set_index("Strategy")[["Mean (s)"]]
                    st.bar_chart(time_chart)
                
                with col_chart2:
                    st.subheader("Speedup vs Serial")
                    if "Speedup" in summary_df.columns:
                        speedup_chart = summary_df.set_index("Strategy")[["Speedup"]]
                        st.bar_chart(speedup_chart)
                
                # Export results
                st.divider()
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    csv_buffer = summary_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv_buffer,
                        file_name=f"nbody_benchmark_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
                
                with col_export2:
                    json_data = summary_df.to_json(orient="records", indent=2)
                    st.download_button(
                        "📥 Download Results (JSON)",
                        json_data,
                        file_name=f"nbody_benchmark_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )
    else:
        st.info("👈 Configure parameters and click **Run** to start benchmarking.")

with tab2:
    if st.session_state.run_history:
        st.subheader(f"📜 Run History ({len(st.session_state.run_history)} run(s))")
        
        for i, run_record in enumerate(reversed(st.session_state.run_history)):
            with st.expander(f"Run {len(st.session_state.run_history) - i}: N={run_record['config']['N']}, "
                           f"Steps={run_record['config']['steps']}, "
                           f"Threads={run_record['config']['omp_threads']}, "
                           f"Procs={run_record['config']['mpi_processes']}"):
                
                # Reconstruct history dataframe
                history_dfs = []
                for detail in run_record["run_details"]:
                    detail_df = pd.DataFrame(detail["results"])
                    detail_df["Run"] = detail["run_num"]
                    history_dfs.append(detail_df)
                
                history_combined = pd.concat(history_dfs, ignore_index=True)
                
                col_left, col_right = st.columns([2, 1])
                with col_left:
                    st.write("**All Runs:**")
                    st.dataframe(history_combined, use_container_width=True)
                
                with col_right:
                    summary = history_combined.groupby("Strategy")[["Time (s)"]].agg(["mean", "std"]).round(6)
                    st.write("**Summary:**")
                    st.dataframe(summary)
        
        if st.button("🗑️ Clear History"):
            st.session_state.run_history = []
            st.rerun()
    else:
        st.info("No run history yet. Run benchmarks to populate history.")

with tab3:
    st.subheader("Build Logs")
    if build_clicked:
        with st.spinner("Building binaries..."):
            build_logs = build_binaries()
        st.session_state.build_logs = build_logs
        st.success("Build completed!")
    
    if st.session_state.build_logs:
        for strategy, log_text in st.session_state.build_logs.items():
            with st.expander(f"Build log: {strategy}"):
                st.code(log_text)
    else:
        st.info("Click **Build** to see compilation logs.")
    
    st.divider()
    st.subheader("Run Logs")
    if st.session_state.last_run_logs:
        for strategy, log_text in st.session_state.last_run_logs.items():
            with st.expander(f"Run log: {strategy}"):
                st.code(log_text)
    else:
        st.info("Click **Run** to see execution logs.")

st.divider()
st.info(
    "💡 **Tips:** Disable CSV output for fair timing comparisons. "
    "Multiple runs show stability. OpenMP overhead is visible at small N."
)
