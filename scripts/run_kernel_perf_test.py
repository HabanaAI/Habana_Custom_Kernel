# *****************************************************************************
# Copyright (c) 2021 Habana Labs.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# *   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# *   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************


#
# Sample script to measure performance of custom TPC kernel tests
#
# Usage:
#
# python3 run_kernel_perf_test.py [-s testName] [-t]
#
# optional arguments:
#   -s testName     Test name regexp
#   -t              Run tpc_runner instead of simulator
#
# Script does the following:
#    1) Run test on tpc_runner;
#    2) Extract performance information fron json file generated at step 1) and store to
#       tmp.csv file;
#    3) Parse csv file to print out the average time and total time.
#


import os
import subprocess
import re
import glob
import argparse
import numpy as np
import sys

####################################################################################################
def get_kernel_time(args):

    time_data = []
    subprocess.run('rm tmp.csv', stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)

    # Seek newly created json file to parse timing information
    list_of_json_files = glob.glob('./*.json') # * means all if need specific format then *.csv
    latest_json = max(list_of_json_files, key=os.path.getctime)

    parser_cmd_line = 'python3 ../scripts/parse_trace_event_format_to_csv.py ' \
                       + latest_json + ' -o tmp.csv'

    # Parse profiler json to get profile data
    result_json = subprocess.run(parser_cmd_line,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 shell=True)
    print(result_json.stdout)

    print('SelfTime: There are a few executions on each engine, ')
    print('AvgTime is the average execution time, SelfTime is the sum of all the executions.')
    print('WallTime: Time from the first begin event to the last end event. Regardless of if there are bubbles/overlapping in between.')
    print('')
    base_name, ext = os.path.splitext(latest_json)
    latest_csv = base_name + '\.csv'

    if not result_json.stderr:
        with open('tmp.csv', 'r') as fh:
            lines = fh.readlines()
            regex = r"All measurements in \S+ \((us)\)"
            matches = re.findall(regex, lines[0].strip(), re.MULTILINE)
            scale = matches[0]

            tpcindex = -1
            for i in range(len(lines)-3):
                lineSizeinWord = len(lines[3+i].strip().split())
                self_time = lines[3+i].strip().split()[lineSizeinWord-1].split(',')[-5]
                avg_time, Wall_time = lines[3+i].strip().split()[lineSizeinWord-1].split(',')[-2:]
          
                self_time = float(self_time)
                avg_time = float(avg_time)
                Wall_time = float(Wall_time)

                if scale == 'us':
                    self_time = int(self_time * 1000)
                    avg_time = int(avg_time * 1000)
                    Wall_time = int(Wall_time * 1000)
                elif scale == 'ms':
                    self_time = int(self_time * 1000000)
                    avg_time = int(avg_time * 1000000)
                    Wall_time = int(Wall_time * 1000000)
                    
                if lineSizeinWord < 5:
                    tpcindex = tpcindex + 1
                    print('At TPC%d SelfTime = %d ns, AvgTime = %d ns, WallTime = %d ns' % (float(tpcindex), float(self_time), float(avg_time), float(Wall_time)), flush=True)

    return (self_time, avg_time, Wall_time)


####################################################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""Run TPC tests profiling""")
    parser.add_argument("-s",  action="store", type=str, help="Test name of kernel")
    parser.add_argument("-t",  action="store_true", help="Run tpc_runner instead of simulator")
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)    
    args = parser.parse_args()

    print('---------------------------------------------------------------')
    print('  Sample script to measure performance of custom TPC kernels   ')
    print('---------------------------------------------------------------')
    print('')

    tpc_run_cmd = '../build/tests/tpc_kernel_tests -t '

    tpc_run_cmd += args.s

    _env = dict(ENABLE_CONSOLE = "true",
                LOG_LEVEL_ALL  = str(1),
                SEED           = str(3796447849)
        )

    if args.t == True:
        _env['TPC_RUNNER'] = str(1)
        _env['HABANA_PROFILE'] = str(1)
        print('TPC_RUNNER is running, now in ASIC mode.')
    else:
        _env['TPC_RUNNER'] = str(0)
        _env['HABANA_PROFILE'] = str(0)
        print('TPC_RUNNER is NOT running, now in simulation mode.')


    subprocess.run('rm *.json', stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
    result_test = subprocess.run(tpc_run_cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True,
                                    shell=True,
                                    env={**os.environ, **_env})

    if result_test.returncode == 0:
        status = 'OK'
        #print(result_test)
        if args.t == True:
            self_time, avg_time, Wall_time = get_kernel_time(args)
        else:
            stdOutValue = result_test.stdout

            # split by space
            my_output_list = stdOutValue.split(" ")
            cyc_index = my_output_list.index('cycles')
            print('Program executed in %d cycles' % float(my_output_list[cyc_index-1]))
    else:
        status = 'FAILED'
        print('Failed to run the test, make sure the kernel test is supoorts by the ASIC!')
    