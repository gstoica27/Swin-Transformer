# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import main as swin
import submitit

# Please change path before use - one time set up
LOGS_PATH = "/nethome/bdevnani3/raid/Swin-Transformer/logs"


def parse_args():
    parent_parser = swin.parse_option()
    parser = argparse.ArgumentParser("Submitit for swin", parents=[parent_parser])
    parser.add_argument(
        "--ngpus", default=2, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes to request"
    )
    parser.add_argument(
        "--cpus_per_task", default=4, type=int, help="Number of nodes to request"
    )
    parser.add_argument("--timeout", default=60, type=int, help="Duration of the job")
    parser.add_argument(
        "--job_dir", default="", type=str, help="Job dir. Leave empty for automatic."
    )
    parser.add_argument(
        "-slurm_partition", type=str, default="overcap", help="slurm partition"
    )
    parser.add_argument("-submitit_run", type=bool, default=True)
    args, _ = parser.parse_known_args()
    return args


def get_shared_folder() -> Path:

    p = Path(LOGS_PATH)
    p.mkdir(exist_ok=True)
    return p


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):

        # lazy imports because we have no guarantees on order of imports
        import main as swin

        self._setup_gpu_args()
        swin.run(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):

        # lazy imports because we have no guarantees on order of imports
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(
            str(self.args.output_dir).replace("%j", str(job_env.job_id))
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()

    # folder = Path("logs/")

    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    # basically, any slurm parameters (exclude, pus_per_task) etc can be added here
    executor.update_parameters(
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus,  # one task per GPU
        nodes=args.nodes,
        cpus_per_task=args.cpus_per_task,
        timeout_min=args.timeout,  # max is 60 * 72
        slurm_partition=args.slurm_partition,
    )

    if args.slurm_partition == "overcap":
        executor.update_parameters(slurm_account=args.slurm_partition)

    executor.update_parameters(name="SWIN")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)
    return job


if __name__ == "__main__":
    job = main()
    # import pdb; pdb.set_trace()
    # job._interrupt(timeout=(False,True))
