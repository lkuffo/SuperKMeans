#!/usr/bin/env python3

import os
import sys

import argparse
import logging

import itertools
import inspect
import types
import subprocess
from pathlib import Path
from typing import Any, Optional
import time


class Stopwatch:
    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self):
        assert self._start is None, "Stopwatch already started"
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        assert self._start is not None, "Stopwatch not started"
        assert self._end is None, "Stopwatch already stopped"
        self._end = time.perf_counter()
        return self._end - self._start


def has_root_privileges() -> bool:
    return os.geteuid() == 0


def directory_exists(path: str) -> bool:
    return Path(path).is_dir()


def get_all_files_in_dir(dir: str) -> list[str]:
    file_paths = []

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    return file_paths


def get_benchmarking_functions() -> list[tuple[str, types.FunctionType]]:
    current_module = inspect.getmodule(inspect.currentframe())
    functions = inspect.getmembers(current_module, inspect.isfunction)

    benchmarking_function_prefix = "bench_"
    benchmarking_functions = filter(
        lambda x: x[0].startswith(benchmarking_function_prefix), functions
    )
    stripped_prefixes_from_name = map(
        lambda x: (x[0].replace(benchmarking_function_prefix, ""), x[1]),
        benchmarking_functions,
    )
    return list(stripped_prefixes_from_name)


def does_file_exist(file_path: str) -> bool:
    return os.path.isfile(file_path)


def run_command(
    command: str,
) -> str:
    if args.dry_run:
        return

    stopwatch: Stopwatch = Stopwatch().start()
    logging.debug(f"Starting: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    logging.info(f"Executed in {stopwatch.stop():.03f}s: {command}")

    if result.returncode != 0:
        logging.critical(f"STDOUT: {result.stdout}")
        logging.critical(f"STDERR: {result.stderr}")
        logging.critical(f"Exited with code {result.returncode}: {command}")

        exit(0)

    return result.stdout


def execute_command(
    command: str,
    output_path: str,
    save_stderr: bool = False,
):
    if args.only_new_runs and does_file_exist(output_path):
        logging.debug(f"Skipping command, output file exists already: {output_path}")
        return

    if args.dry_run:
        print(command, file=sys.stderr)
        return

    stopwatch: Stopwatch = Stopwatch().start()
    logging.debug(f"Starting: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    logging.info(f"Executed in {stopwatch.stop():.03f}s: {command}")

    if result.returncode != 0:
        logging.critical(f"STDOUT: {result.stdout}")
        logging.critical(f"STDERR: {result.stderr}")
        logging.critical(f"Exited with code {result.returncode}: {command}")

        if args.exit_non_zero:
            exit(0)
        else:
            return

    with open(output_path, "w") as f:
        f.write(result.stdout)

        if save_stderr:
            f.write("STDERR:\n")
            f.write(result.stderr)



# ===========================
# benchmarks
# ===========================

def format_to_filename(output_dir: str, command: str, sampling_run: int) -> str:
    return os.path.join(output_dir, f"{command.replace(' ', '_')}_sample-{sampling_run}")


class CommandBuilder:
    command: list

    def __init__(self):
        self.command = []

    def add(self, argument):
        self.command.append(argument)
        return self

    def build(self) -> str:
        return " ".join(self.command)

DATASETS = [
    "sift",
    "wiki",
    "arxiv",
    "contriever",
    "mxbai",
    "openai",
    "cohere",
]

def bench_superkmeans(output_dir: str):
    for dataset in DATASETS:
        for s in range(args.number_sampling_runs):
            filename = format_to_filename(
                output_dir, f"varying_k_superkmeans_{dataset}", s + 1
            )
            command = (
                CommandBuilder()
                .add("./varying_k_superkmeans.out")
                .add(dataset)
                .build()
            )
            execute_command(command, filename)

def bench_gpu_faiss(output_dir: str):
    for dataset in DATASETS:
        for s in range(args.number_sampling_runs):
            filename = format_to_filename(
                output_dir, f"varying_k_gpu_faiss_{dataset}", s + 1
            )
            command = (
                CommandBuilder()
                .add("./gpu_faiss.out")
                .add("0")
                .add(dataset)
                .build()
            )
            execute_command(command, filename)


def bench_cuvs(output_dir: str):
    for dataset in DATASETS:
        for s in range(args.number_sampling_runs):
            filename = format_to_filename(
                output_dir, f"varying_k_cuvs_{dataset}", s + 1
            )
            command = (
                CommandBuilder()
                .add("./end_to_end_cuvs_kmeans_clustering.out")
                .add("1")
                .add(dataset)
                .build()
            )
            execute_command(command, filename)


def main(args):
    assert directory_exists(args.output_dir)
    args.benchmarking_function(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    benchmarking_functions = {func[0]: func[1] for func in get_benchmarking_functions()}
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "benchmarking_function",
        type=str,
        choices=list(benchmarking_functions.keys()) + ["all"],
        help="function to execute",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="directory_to_write_results_to",
    )
    parser.add_argument(
        "-nsr",
        "--number-sampling-runs",
        type=int,
        default=1,
        help="Executes commands multiple times",
    )
    parser.add_argument(
        "-onr",
        "--only-new-runs",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Only runs new runs. If the filename already exists in the output, the run is skipped",
    )
    parser.add_argument(
        "-enz",
        "--exit-non-zero",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Exit when a command return a non zero statuscode",
    )
    parser.add_argument(
        "-dr",
        "--dry-run",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Dry run",
    )

    parser.add_argument(
        "-ll",
        "--logging-level",
        type=int,
        default=logging.INFO,
        choices=[logging.CRITICAL, logging.ERROR, logging.INFO, logging.DEBUG],
        help=f"logging level to use: {logging.CRITICAL}=CRITICAL, {logging.ERROR}=ERROR, {logging.INFO}=INFO, "
        + f"{logging.DEBUG}=DEBUG, higher number means less output",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.logging_level)  # filename='program.log',
    logging.info(
        f"Started {os.path.basename(sys.argv[0])} with the following args: {args}"
    )

    if args.benchmarking_function == "all":
        args.benchmarking_function = lambda out_dir: list(
            func(out_dir) for func in benchmarking_functions.values()
        )
    else:
        args.benchmarking_function = benchmarking_functions[args.benchmarking_function]
    main(args)
