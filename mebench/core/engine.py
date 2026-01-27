"""Benchmark engine core."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np
import yaml
from pathlib import Path

from mebench.core.state import BenchmarkState
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.validate import validate_config
from mebench.core.seed import set_seed
from mebench.oracles.oracle import Oracle
from mebench.oracles.victim_loader import load_victim_from_config
from mebench.attackers.base import BaseAttack
from mebench.attackers.activethief import ActiveThief
from mebench.attackers.dfme import DFME
from mebench.attackers.maze import MAZE
from mebench.attackers.dfms import DFMSHL
from mebench.attackers.game import GAME
from mebench.attackers.es_attack import ESAttack
from mebench.attackers.random_baseline import RandomBaseline
from mebench.attackers.swiftthief import SwiftThief
from mebench.attackers.blackbox_dissector import BlackboxDissector
from mebench.attackers.cloudleak import CloudLeak
from mebench.attackers.blackbox_ripper import BlackboxRipper
from mebench.attackers.copycatcnn import CopycatCNN
from mebench.attackers.inversenet import InverseNet
from mebench.attackers.knockoff_nets import KnockoffNets
from mebench.data.loaders import get_test_dataloader, create_dataloader
from mebench.core.state import BenchmarkState
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.validate import validate_config
from mebench.core.seed import set_seed
from mebench.core.query_storage import create_query_storage
from mebench.eval.evaluator import Evaluator
from mebench.core.logging import ArtifactLogger, create_run_dir
from mebench.models.substitute_factory import create_substitute


def create_attack(
    attack_name: str,
    config: Dict[str, Any],
    state: BenchmarkState,
) -> BaseAttack:
    """Create attack instance from name.

    Args:
        attack_name: Attack name (random, activethief, dfme)
        config: Attack configuration
        state: Global benchmark state

    Returns:
        Attack instance
    """
    if attack_name == "activethief":
        return ActiveThief(config["attack"], state)
    elif attack_name == "dfme":
        return DFME(config["attack"], state)
    elif attack_name == "maze":
        return MAZE(config["attack"], state)
    elif attack_name == "dfms":
        return DFMSHL(config["attack"], state)
    elif attack_name == "game":
        return GAME(config["attack"], state)
    elif attack_name == "es":
        return ESAttack(config["attack"], state)
    elif attack_name == "random":
        return RandomBaseline(config["attack"], state)
    elif attack_name == "swiftthief":
        return SwiftThief(config["attack"], state)
    elif attack_name == "blackbox_dissector":
        return BlackboxDissector(config["attack"], state)
    elif attack_name == "cloudleak":
        return CloudLeak(config["attack"], state)
    elif attack_name == "blackbox_ripper":
        return BlackboxRipper(config["attack"], state)
    elif attack_name == "copycatcnn":
        return CopycatCNN(config["attack"], state)
    elif attack_name == "inversenet":
        return InverseNet(config["attack"], state)
    elif attack_name == "knockoff_nets":
        return KnockoffNets(config["attack"], state)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def run_experiment(
    config: Dict[str, Any],
    device: str = "cpu",
) -> None:
    """Run benchmark experiment.

    Args:
        config: Experiment configuration
        device: Device to use
    """
    # Validate config
    validate_config(config)

    # Run for each seed
    for seed in config["run"]["seeds"]:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")

        # Set seed for reproducibility
        set_seed(seed)

        # Create run directory
        base_dir = Path("runs")
        run_dir = create_run_dir(base_dir, config["run"]["name"], seed)
        print(f"Run directory: {run_dir}")

        # Initialize logger
        logger = ArtifactLogger(run_dir)
        logger.set_run_metadata(config)
        logger.save_config(config)

        # Initialize state
        state = BenchmarkState(
            budget_remaining=config["budget"]["max_budget"],
            metadata={
                "device": device,
                "input_shape": (
                    int(config["victim"]["channels"]),
                    *config["victim"].get("input_size", [32, 32]),
                ),
                "dataset_config": config.get("dataset", {}),
                "substitute_config": config.get("substitute", {}),
                "victim_config": config.get("victim", {}),
                "max_budget": config["budget"]["max_budget"],
            },
        )

        # Load victim model from checkpoint or placeholder
        victim = load_victim_from_config(config["victim"], device)

        # Initialize oracle
        oracle = Oracle(victim, config["victim"], state)

        # Initialize attack
        attack = create_attack(config["attack"]["name"], config, state)

        # Load test dataloader
        test_loader = get_test_dataloader(
            name=config["dataset"]["name"],
            batch_size=128,
        )

        # Initialize query storage for Track A
        query_storage = create_query_storage(
            run_dir,
            output_mode=config["victim"]["output_mode"],
        )

        # Initialize evaluator
        evaluator = Evaluator(config, state, query_storage)

        # Save run config
        logger.save_config(config)

        # Main query collection loop
        checkpoints = config["budget"]["checkpoints"]
        max_budget = config["budget"]["max_budget"]

        print(f"\nStarting query collection (max budget: {max_budget})")

        while state.query_count < max_budget:
            # Determine query batch size
            step_size = min(100, max_budget - state.query_count)

            # Propose queries
            query_batch = attack.propose(step_size, state)

            # Query oracle
            oracle_output = oracle.query(query_batch.x)

            # Observe response
            attack.observe(query_batch, oracle_output, state)

            # Store queries for Track A training
            query_storage.add_batch(query_batch.x, oracle_output.y)

            print(f"\rQueries: {state.query_count}/{max_budget}", end="", flush=True)

            # Check if we've reached a checkpoint
            for checkpoint in checkpoints:
                if state.query_count >= checkpoint and checkpoint not in state.attack_state.get("checkpoint_reached", []):
                    print(f"\n\nCheckpoint reached: {checkpoint} queries")
                    print("Evaluating...")

                    # Evaluate at checkpoint
                    results = evaluator.evaluate(victim, test_loader, checkpoint)

                    # Log results
                    for track in ["track_a", "track_b"]:
                        logger.log_checkpoint(
                            seed=seed,
                            checkpoint=checkpoint,
                            track=track,
                            metrics=results[track],
                        )
                        print(f"{track}: {results[track]}")

                    # Mark checkpoint as reached
                    if "checkpoint_reached" not in state.attack_state:
                        state.attack_state["checkpoint_reached"] = []
                    state.attack_state["checkpoint_reached"].append(checkpoint)

        print(f"\n\nQuery collection complete!")
        print(f"Total queries: {state.query_count}")

        # Finalize logging
        logger.finalize()

        # Save query storage and optionally cleanup
        query_storage.save()

        # Clean up cache if configured
        if config.get("cache", {}).get("delete_on_finish", True):
            query_storage.cleanup()
        else:
            print(f"Query cache preserved at {query_storage.cache_dir}")

    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"{'='*60}")
