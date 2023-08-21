import os
import subprocess

cwd = os.getcwd()
script_path = os.path.abspath(__file__)
script_name = os.path.splitext(os.path.basename(script_path))[0]
tester = "./" + script_name

def do_test(name):
    runner = tester + " " + name
    output_bytes = subprocess.check_output(runner, shell=True, cwd=cwd)
    output_str = output_bytes.decode('utf-8')
    print(output_str, end='')

do_test("dense")
do_test("conv")
do_test("with_pooling")
