#!/usr/bin/env python3
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
import sys
import os
import json
import argparse
import csv
from typing import List
from typing import Dict
from typing import TextIO


# some globals
include = ["mme", "tpc"]
exclude = []
metadata = {}
metadata['units'] = {}
metadata['engines'] = {}


def get_trace_events(profiling_file: str) -> List[Dict]:
    if not os.path.isfile(profiling_file):
        sys.exit()

    with open(profiling_file) as f:
        data = json.load(f)

    return data["traceEvents"]


def format_and_writerow(out_csv: TextIO, vals: list) -> None:
    # round floats to 3 decimal places
    row = []
    for f in vals:
        val = round(f, 3) if isinstance(f, float) else f
        row.append(val)
    out_csv.writerow(row)


# Take a Trace Event Format file and create a csv file with the total time of each node in descending order
# All times in are in microseconds
def do_parse(json_file_path: str, csv_file_path: str, group_by_name: bool = False) -> None:

    trace_events = get_trace_events(json_file_path)

    for event in trace_events:
        if event['ph'] == 'M':
            collect_metadata(event)

    total_times = {}
    start_times = {}
    wall_begin_times = {}
    wall_end_times = {}
    guids = {}
    context_id_to_name = {}
    for event in trace_events:
        if event['cat'] == 'Hardware event' or event['cat'].startswith('HE') or event['cat'].startswith('SP'):
            record_event(event, start_times, total_times, wall_begin_times, wall_end_times, guids, context_id_to_name, group_by_name)

    if len(total_times) == 0:
        return

    with open(csv_file_path, 'w') as csv_file:
        out = csv.writer(csv_file, dialect='excel')
        format_and_writerow(out, ['All measurements in microseconds (us)'])
        format_and_writerow(out, ['Node', 'Guid', 'Total count', 'Self time', 'Min', 'Max', 'Avg', 'Wall time'])
        first_begin = min(wall_begin_times[_id] for _id in wall_begin_times)
        last_end = max(wall_end_times[_id] for _id in wall_end_times)
        format_and_writerow(out, ['ALL_NODES', '-', '-', '-', '-', '-', '-', last_end - first_begin])
        for _id in total_times:
            count = len(total_times[_id])
            self_time = sum(total_times[_id])
            min_time = min(total_times[_id])
            max_time = max(total_times[_id])
            avg_time = 0 if count == 0 else self_time/count
            guid = guids[_id] if _id in guids else ""
            if _id in wall_begin_times and _id in wall_end_times:
                wall_time = wall_end_times[_id] - wall_begin_times[_id]
            else:
                wall_time = 0
            row = [_id if group_by_name else context_id_to_name[_id], guid, count, self_time, min_time, max_time, avg_time, wall_time]
            format_and_writerow(out, row)


def collect_metadata(event: Dict) -> None:
    name = event['name']
    ph = event['ph']
    if name != "null" and name != '':
        if ph == 'M':
            if name == "process_name":
                # add to unit dict
                metadata['units'][event['pid']] = event['args']['name']
            elif name == "thread_name":
                # add to engine dict
                metadata['engines'][event['tid']] = event['args']['name']

def is_recording_unit(tid: int) -> bool:
    if (len(include) == 0 or any(expr in metadata['engines'][tid].lower() for expr in include)) and \
        not any(expr in metadata['engines'][tid].lower() for expr in exclude):
            return True

    return False

def record_event(event: Dict, start_times: Dict, total_times: Dict, wall_begin_times: Dict, wall_end_times: Dict, guids: Dict, context_id_to_name: Dict, group_by_name: bool = False):
    tid = event['tid']
    name = event['name']
    context_id = event['id'] if 'id' in event else 0
    ts = event['ts']
    ph = event['ph']
    unit_id = event['pid']

    _id = name if group_by_name else context_id

    guid = ""
    if ("args" in event) and ("op" in event["args"]):
        guid = event['args']['op']
    if guid != '' and _id != 0:
        guids[_id] = guid

    if name != "null" and is_recording_unit(tid):
        if tid not in start_times:
            start_times[tid] = dict()
        if ph == 'B':
            start_times[tid][_id] = ts
            if name != "" or (not group_by_name and context_id not in context_id_to_name):
                context_id_to_name[context_id] = name
            # record the first begin event of a given name for wall time
            if _id not in wall_begin_times or wall_begin_times[_id] > ts:
                wall_begin_times[_id] = ts
        elif ph == 'E':
            if _id in start_times[tid]:
                time = ts - start_times[tid][_id]
                start_times[tid].pop(_id)
                if _id in total_times:
                    total_times[_id].append(time)
                else:
                    total_times[_id] = [time]
            # record the last end event of a given _id for wall time
            if _id not in wall_end_times or wall_end_times[_id] < ts:
                wall_end_times[_id] = ts


def main():
    parser = argparse.ArgumentParser(description="""Convert profiler json output from Trace Event Format to CSV""")
    parser.add_argument("filename", type=str, default="", help="Input file - a trace event format json. Mandatory arg")
    parser.add_argument("-o", type=str, default="", help="Output file - default is the input file with replaced suffix")
    parser.add_argument('--log_level', type=int, default=None, choices=range(0, 7), help="0 - TRACE, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, 5 - CRITICAL, 6 - OFF")
    parser.add_argument("-dma",  action="store_true", help="Include DMA engines in calcuations (default is MME and TPC only). ignored if 'include'/'exclude' flags are used")
    parser.add_argument("-exclude", type=str, default="", help="Exclude engines that contains this expression (seperate multiple expression with ',')")
    parser.add_argument("-include", type=str, default="", help="Include only engines that contains this expression (seperate multiple expression with ',')")
    parser.add_argument("-group_by_name", action="store_true", help="Group events by event name instead of by context-id (recommended for Goya)")
    args = parser.parse_args()

    global include
    global exclude
    if args.include or args.exclude:
        include = args.include.lower().split(",")
        if args.exclude:
            exclude = args.exclude.lower().split(",")
    elif args.dma == True:
        include.append("dma")

    json_file_path = args.filename

    if json_file_path == "":
        exit(1)
    if json_file_path.rfind(".json") < 0:
        exit(1)

    csv_file_path = json_file_path.replace(".json", ".csv") if args.o == "" else args.o

    json_file_path = args.filename
    do_parse(json_file_path, csv_file_path, args.group_by_name)


if __name__ == '__main__':
    main()
