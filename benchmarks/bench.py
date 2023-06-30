"""Bench clear_net with other ways to create nets."""
# TODO not able to use psutil to measure memory, maybe
# issue with using an env or the process ending too fast
import subprocess
import psutil
import os
import time


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
bench_dir = 'benchmarks'


def time_command(command):
    """Time the given command by running it."""
    start = time.time()
    subprocess.call(command, stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    end = time.time()
    return end - start


def measure_memory_usage(command):
    """Run the test and return mem and time used."""
    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               cwd=parent_dir)
    process.wait()
    end_time = time.time()

    pid = process.pid
    process = psutil.Process(pid)
    memory_info = process.memory_info()

    # Memory usage in bytes
    memory_usage = memory_info.rss

    return memory_usage, end_time - start_time


commands = [
    ['./iris', '-b'],
    # pure python is probably also a pretty bad implementation
    ['python', f'./{bench_dir}/pure_python_iris.py'],
    ['python', f'./{bench_dir}/pytorch_iris.py'],
]

with open(f"./{bench_dir}/times", "w") as file:
    for command in commands:
        # memory_usage, time_used = measure_memory_usage(command)
        text = f"Command: {command} "
        print(text)
        used = time_command(command)
        file.write(text)
        # print(f"Memory Usage: {memory_usage:.2f} Bytes")
        text = f"Time taken: {used}"
        print(text)
        file.write(text + '\n')
        print("--------------------------")
