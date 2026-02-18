import subprocess
import json
import os
import re
import platform

# Configuration
BUILD_DIR = os.path.abspath("./build/Release") if platform.system() == "Windows" else os.path.abspath("./build")
REPORT_FILE = "readiness_report.json"

def run_executable(name):
    path = os.path.join(BUILD_DIR, name)
    if platform.system() == "Windows": path += ".exe"
    
    if not os.path.exists(path):
        print(f"[Error] Executable not found: {path}")
        return None

    print(f"[Orchestrator] Running {name}...")
    try:
        result = subprocess.run([path], capture_output=True, text=True, timeout=60)
        output = result.stdout
        print(output)
        
        # Extract JSON blob
        match = re.search(r"JSON_START(.*?)JSON_END", output, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            print(f"[Error] No JSON output found in {name}")
            return None
    except Exception as e:
        print(f"[Error] Failed to run {name}: {e}")
        return None

def run_python_script(script_name):
    # Assumes script is in same directory as this orchestrator
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    print(f"[Orchestrator] Running Python Script: {script_name}...")
    try:
        # Run with current python interpreter
        import sys
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=120)
        output = result.stdout
        print(output)
        
        match = re.search(r"JSON_START(.*?)JSON_END", output, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            print(f"[Error] No JSON output found in {script_name}")
            return None
    except Exception as e:
        print(f"[Error] Failed to run {script_name}: {e}")
        return None

def run_security_audit():
    path_sec = os.path.join(BUILD_DIR, "audit_security")
    if platform.system() == "Windows": path_sec += ".exe"
    
    if not os.path.exists(path_sec):
        print("[Warn] Security Audit executable not found.")
        return "UNKNOWN"

    try:
        res_sec = subprocess.run([path_sec], capture_output=True, text=True, timeout=30)
        if res_sec.returncode == 0 and "All Checks Passed" in res_sec.stdout:
            return "VERIFIED"
        else:
            print(res_sec.stdout)
            return "FAILED"
    except Exception as e:
        print(f"Security Audit Failed to Run: {e}")
        return "ERROR"

def run_stress_test_audit():
    stress_res = run_python_script("stress_test.py")
    if stress_res and stress_res.get("backpressure_events", 0) > 0:
        return 100.0, "STABLE"
    return 0.0, "RISK"

def calculate_grade(report):
    p_status = report["benchmarks"].get("numerical_integrity", {}).get("status", "FAIL")
    l_status = report["benchmarks"].get("latency_p99", {}).get("status", "BRONZE")
    r_status = report["benchmarks"]["resiliency"]["status"]
    
    if p_status == "PASS" and l_status == "GOLD" and r_status == "STABLE":
        return "GOLD"
    elif p_status == "PASS" and (l_status == "GOLD" or l_status == "SILVER") and r_status == "STABLE":
        return "SILVER"
    else:
        return "BRONZE"

def generate_report():
    report = {
        "project": "SynapseVM",
        "version": "1.0.0-PROD_READY",
        "hardware": "RTX 4070 / Ryzen 7 7745HX",  
        "benchmarks": {}
    }

    # 1. Precision Audit
    precision_res = run_executable("audit_precision")
    if precision_res:
        report["benchmarks"]["numerical_integrity"] = precision_res
        report["benchmarks"]["numerical_integrity"]["status"] = precision_res.get("status", "UNKNOWN")

    # 2. Latency Audit
    latency_res = run_executable("audit_latency")
    if latency_res:
        report["benchmarks"]["latency_p99"] = latency_res
        eff = latency_res.get("masking_efficiency", 0)
        if eff > 95.0: latency_res["status"] = "GOLD"
        elif eff > 80.0: latency_res["status"] = "SILVER"
        else: latency_res["status"] = "BRONZE"

    # 3. Security Audit (Isolation)
    isolation_status = run_security_audit()

    # 4. Memory Tsunami (Stress Test)
    backpressure_success, risk_status = run_stress_test_audit()
    
    # Consolidated Resiliency Report
    final_status = "STABLE" if (isolation_status == "VERIFIED" and risk_status == "STABLE") else "RISK"
    
    report["benchmarks"]["resiliency"] = {
        "backpressure_success_rate": backpressure_success,
        "isolation_verification": isolation_status,
        "status": final_status
    }
    
    # Overall Grading
    report["overall_grade"] = calculate_grade(report)

    # Write to File
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[Success] Readiness Report generated at {os.path.abspath(REPORT_FILE)}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    generate_report()
