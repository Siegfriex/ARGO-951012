"""
ARGO agents module.

This module provides functions to initialize and interact with execution agents.

Example:
    from argo.agents import initialize_hephaestus, process_command

    agent = initialize_hephaestus()
    if agent:
        response = process_command(agent, "What time is it in UTC?")
        print(response)
"""

from .hephaestus import HephaestusAgent, initialize_hephaestus, process_command

__all__ = [
    "HephaestusAgent",
    "initialize_hephaestus",
    "process_command",
]
