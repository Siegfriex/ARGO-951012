# -*- coding: utf-8 -*-
# Copyright 2025 Siegfriex
#
# Licensed under the MIT License.
#
"""Unit tests for the Hephaestus agent module."""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

from vertexai.preview.reasoning_engines import LangchainAgent

from argo.agents import hephaestus


@patch("argo.agents.hephaestus.load_dotenv")
def test_initialize_hephaestus_fails_without_env_vars(mock_load_dotenv_unused, capsys):
    """
    Tests that agent initialization fails gracefully
    if environment variables are not set.
    """
    # Act - The mock is active, but we don't need to use its reference.
    agent = hephaestus.initialize_hephaestus()

    # Assert
    assert agent is None
    captured = capsys.readouterr()
    assert "ERROR: Please set PROJECT_ID and LOCATION" in captured.err


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
    mock_vertex_init.assert_called_once_with(
        project="test-project", location="us-central1"
    )
    mock_create_agent.assert_called_once()


def test_process_command_with_tool_usage_simulation(mocker):
    """
    Tests that process_command correctly handles a response that
    simulates a successful tool call, as per Mission 1.3a.
    """
    # 1. Arrange: Mock dependencies to prevent actual API calls.
    mock_agent_instance = MagicMock(spec=LangchainAgent)
    current_year = datetime.now().year
    fake_llm_output = (
        f"The current time in Asia/Seoul is {current_year}-05-22T15:30:00+09:00."
    )
    mock_agent_instance.query.return_value = {"output": fake_llm_output}

    # 2. Act: Initialize the agent and process the command.
    result = hephaestus.process_command(mock_agent_instance, "지금 서울 시간 알려줘")

    # 3. Assert: Verify the agent was queried and the output is correct.
    mock_agent_instance.query.assert_called_once_with(input="지금 서울 시간 알려줘")
    assert "Asia/Seoul" in result
    assert str(current_year) in result


def test_process_command_empty_input(mocker):
    """Tests that empty commands are handled correctly without calling the agent."""
    mock_agent = mocker.MagicMock(spec=LangchainAgent)
    result = hephaestus.process_command(mock_agent, "   ")
    assert result == "[INFO] Please enter a command."
    mock_agent.query.assert_not_called()


@patch("argo.agents.hephaestus.create_hephaestus_agent")
def test_hephaestus_agent_class_query(mock_create_agent):
    """Tests that the deployable HephaestusAgent class correctly calls its inner agent."""
    # Arrange
    mock_inner_agent = MagicMock(spec=LangchainAgent)
    mock_inner_agent.query.return_value = {"output": "Success"}
    mock_create_agent.return_value = mock_inner_agent

    # Act
    deployable_agent = hephaestus.HephaestusAgent()
    response = deployable_agent.query(input="test command")

    # Assert
    mock_create_agent.assert_called_once()
    mock_inner_agent.query.assert_called_once_with(input="test command")
    assert response == {"output": "Success"}
