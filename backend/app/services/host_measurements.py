"""Sample process RSS separately from timing; this is not model-only RAM."""
import os
import platform
import statistics
import subprocess
import threading
import time


def environment():
    cpu = platform.processor() or 'UNAVAILABLE'
    power = 'UNAVAILABLE'
    if platform.system() == 'Windows':
        try:
            power = subprocess.run(['powercfg', '/getactivescheme'], capture_output=True, text=True, timeout=5,
                                   creationflags=0x08000000).stdout.strip() or 'UNAVAILABLE'
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except (OSError, subprocess.SubprocessError): pass
    elif platform.system() == 'Linux':
        try:
            from pathlib import Path
            cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
            power = 'UNAVAILABLE: Linux power governor not queried'
        except (OSError, StopIteration): pass
    return {'processor': cpu, 'logical_cpus': os.cpu_count(), 'power_plan': power,
            'power_plan_evidence_status': 'MEASURED' if not power.startswith('UNAVAILABLE') else 'UNAVAILABLE'}


def rss_bytes():
    if platform.system() == 'Windows':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(k, ctypes.c_size_t) for k in
                ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True); kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        value = Counters(); value.cb = ctypes.sizeof(value)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb): raise OSError('RSS query failed')
        return int(value.WorkingSetSize)
    if platform.system() == 'Linux':
        from pathlib import Path
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    raise OSError('Process RSS sampling unsupported on this platform')


def measure_memory(infer, sample, runs=30):
    observations, stopped = [], threading.Event()
    def observe():
        while not stopped.is_set():
            try: observations.append(rss_bytes())
            except OSError: return
            stopped.wait(.01)
    try:
        baseline = rss_bytes(); observations.append(baseline)
        worker = threading.Thread(target=observe, daemon=True); worker.start()
        try:
            for _ in range(runs):
                infer(sample); observations.append(rss_bytes())
        finally:
            stopped.set(); worker.join(timeout=2)
        peak = max(observations)
        return {'process_rss_baseline_bytes': baseline, 'process_rss_sampled_peak_bytes': peak,
                'process_rss_increase_bytes': max(0, peak - baseline), 'memory_sample_count': len(observations),
                'memory_evidence_status': 'MEASURED', 'memory_scope': 'Current engine process; all loaded models, datasets and runtimes included.',
                'memory_method': 'Separate 30-inference pass; RSS sampled every 10 ms and after each invocation. Sampled maximum, not guaranteed true peak or model-only allocation.'}
    except OSError as exc:
        return {'memory_evidence_status': 'UNAVAILABLE', 'memory_method': str(exc)}


def spread(values):
    from app.services.metrics import percentile
    return {'latency_min_ms': min(values), 'latency_max_ms': max(values),
            'latency_p25_ms': percentile(values, 25), 'latency_p75_ms': percentile(values, 75),
            'latency_stddev_ms': statistics.stdev(values) if len(values) > 1 else 0,
            'timing_evidence_status': 'MEASURED', 'timing_warning': 'Small differences require repeated independent experiments; no significance claim.'}
