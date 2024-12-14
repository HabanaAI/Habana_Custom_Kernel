#!/bin/env python3
import pathlib
import torch, logging
import os
file_path = pathlib.Path(__file__).parent.resolve()
os.environ["GC_KERNEL_PATH"] += f":{file_path}/libcustom_tpc_perf_lib.so"

from mpi4py import MPI
os.environ['MASTER_ADDR'] = 'localhost'  # server with rank=0 (master)
os.environ['MASTER_PORT'] = '12355'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
os.environ['RANK'] = f"{rank}"
os.environ['WORLD_SIZE'] = f"{world_size}"
os.environ['LOCAL_RANK'] = f"{rank}"

os.environ["PT_HPU_LAZY_MODE"] = "1"


print(f"world_size {world_size}")
print(f"rank {rank}")
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as htorch

class TensorChecker:
    def __init__(self, gold=None, device=None):
        self.device = device
        self.aggregate_check = torch.empty(size=(0,))
        self.gold = gold

    @property
    def gold(self):
        return self._gold

    @gold.setter
    def gold(self, value):
        self._gold = value if value is not None else torch.empty(size=(0,))
        self.init_aggregate_check()

    def init_aggregate_check(self):
        self.aggregate_check = torch.ones_like(self._gold, dtype=torch.bool, device=self.device)

    def check(self, answer):
        answer = answer.to(torch.device(self.device))
        self._gold = self._gold.to(torch.device(self.device))
        self.aggregate_check &= torch.eq(self._gold, answer)

    def passed(self):
        return torch.all(self.aggregate_check)

    def failed(self):
        return not self.passed()

def log(msg):
    print(f"Rank{rank}: {msg}", flush=True)

def run(input_values):
        tin = torch.tensor(input_values, dtype=torch.float32, device="hpu")
        # tin = tin.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # print(f"tin.shape = {tin.shape}")
        kernel_path = f"{file_path}/hpu_custom_reinterpret.cpython-310-x86_64-linux-gnu.so"
        torch.ops.load_library(kernel_path)

        # If you un-comment this print:
        # it will start working, but resutls are incorrect particularly, when running on multiple gaudis
        # log(tin)

        tout = torch.ops.custom_op.reinterpret_float(tin)

        # If you uncomment this print:
        # All the answers returned are zero

        # If you uncomment both prints, then we will pass,
        # though we have seen some data mismatches on rare occasions
        # log(tout)
        # tout = tout.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
        # print(f"tout.shape = {tout.shape}")

        return tout



def print_tensor_as_hex(tensor):
    return [f"0x{value.item():08x}" for value in tensor.view(-1)]
        # print(f"0x{value.item():08x}")


if __name__ == '__main__':
    iterations = 10

    # We will convert 1.0, 2,0 and 3.0 from float to raw binary value (reinterpreted_cast)
    # And the expected values are listed in the gold tensor below
    input_values = [1.0, 2.0, 3.0]
    #                        1.0          2.0       3.0
    gold = torch.tensor([0x3f800000, 0x40000000, 0x40400000], dtype=torch.int32)
    checker = TensorChecker(gold, device="cpu")
    answers = []
    for _ in range(iterations):
        ans = run(input_values)
        checker.check(ans)
        # answers.append(ans)

    print("Answers:")
    for i, v in enumerate(answers):
        log(f" Iteration {i} : {print_tensor_as_hex(v)}")


    if checker.failed():
        log(f"Mismatches detected on data during local checking on rank {rank} ")
    else:
        log(f"No mismatches detected on rank {rank}")
