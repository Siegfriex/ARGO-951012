# -*- coding: utf-8 -*-
# Copyright 2025 Siegfriex
#
# Licensed under the MIT License.
#
"""Unit tests for the Hephaestus agent module."""

import os
from unittest.mock import MagicMock, patch

from vertexai.preview.reasoning_engines import LangchainAgent

from argo.agents import hephaestus


@patch("argo.agents.hephaestus.load_dotenv")
@patch.dict(os.environ, {}, clear=True)
def test_initialize_hephaestus_fails_without_env_vars(mock_load_dotenv, capsys):
    """
    Tests that agent initialization fails gracefully
    if environment variables are not set.
    """
    # Act
    agent = hephaestus.initialize_hephaestus()

    # Assert
    assert agent is None
    captured = capsys.readouterr()
    assert "ERROR: Please set PROJECT_ID and LOCATION" in captured.err
    mock_load_dotenv.assert_called_once()


@patch("argo.agents.hephaestus.create_hephaestus_agent")
@patch("argo.agents.hephaestus.vertexai.init")
@patch("argo.agents.hephaestus.load_dotenv")
@patch.dict(
    os.environ,
    {"PROJECT_ID": "test-project", "LOCATION": "us-central1"},
    clear=True,
)
def test_initialize_hephaestus_with_exception(
    mock_load_dotenv, mock_vertex_init, mock_create_agent, capsys
):
    """
    Tests that agent initialization handles exceptions gracefully
    during agent creation.
    """
    # Arrange
    mock_create_agent.side_effect = Exception("Creation failed")

    # Act
    agent = hephaestus.initialize_hephaestus()

    # Assert
    assert agent is None
    captured = capsys.readouterr()
    assert "[CRITICAL ERROR] Failed to initialize Hephaestus-Alpha: Creation failed" in captured.err
    mock_load_dotenv.assert_called_once()
    mock_vertex_init.assert_called_once_with(project="test-project", location="us-central1")


@patch("argo.agents.hephaestus.load_dotenv")
@patch("argo.agents.hephaestus.vertexai.init")
@patch("argo.agents.hephaestus.create_hephaestus_agent")
@patch.dict(
    os.environ,
    {"PROJECT_ID": "test-project", "LOCATION": "us-central1"},
    clear=True,
)
def test_initialize_hephaestus_success(
    mock_create_agent, mock_vertex_init, mock_load_dotenv
):
    """
    Tests successful agent initialization, ensuring Vertex AI is initialized
    with the standard operating region.
    """
    # Arrange
    mock_agent_instance = MagicMock(spec=LangchainAgent)
    mock_create_agent.return_value = mock_agent_instance

    # Act
    agent = hephaestus.initialize_hephaestus()

    # Assert
    assert agent is mock_agent_instance
    mock_load_dotenv.assert_called_once()
    mock_vertex_init.assert_called_once_with(project="test-project", location="us-central1")
    mock_create_agent.assert_called_once()


def test_process_command_success():
    """Tests successful command processing."""
    # Arrange
    mock_agent = MagicMock(spec=LangchainAgent)
    mock_agent.query.return_value = {"output": "The time is 10:00 AM."}
    command = "what time is it?"

    # Act
    result = hephaestus.process_command(mock_agent, command)

    # Assert
    mock_agent.query.assert_called_once_with(input=command)
    assert result == "[RESPONSE] The time is 10:00 AM."


def test_process_command_empty_input():
    """Tests that empty commands are handled correctly without calling the agent."""
    # Arrange
    mock_agent = MagicMock(spec=LangchainAgent)
    command = "   "

    # Act
    result = hephaestus.process_command(mock_agent, command)

    # Assert
    assert result == "[INFO] Please enter a command."
    mock_agent.query.assert_not_called()


def test_process_command_agent_exception():
    """Tests that agent exceptions are handled gracefully."""
    # Arrange
    mock_agent = MagicMock(spec=LangchainAgent)
    mock_agent.query.side_effect = Exception("Agent query failed")
    command = "test command"

    # Act
    result = hephaestus.process_command(mock_agent, command)

    # Assert
    mock_agent.query.assert_called_once_with(input=command)
    assert result == "[ERROR] An error occurred during execution: Agent query failed"


def test_process_command_missing_output():
    """Tests handling of responses without 'output' key."""
    # Arrange
    mock_agent = MagicMock(spec=LangchainAgent)
    mock_agent.query.return_value = {"result": "Some other key"}
    command = "test command"

    # Act
    result = hephaestus.process_command(mock_agent, command)

    # Assert
    mock_agent.query.assert_called_once_with(input=command)
    assert result == "[RESPONSE] No output received."
