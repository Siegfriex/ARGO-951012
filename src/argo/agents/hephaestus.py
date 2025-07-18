# -*- coding: utf-8 -*-
# Copyright 2025 Siegfriex
#
# Licensed under the MIT License.
#
"""Hephaestus-Alpha: The first Execution Agent of the ARGO Empire."""

import os
import sys
from dotenv import load_dotenv
import vertexai # type: ignore
from vertexai.preview.reasoning_engines import LangchainAgent

# Import the tool from our arsenal
from argo.tools.time_tool import get_current_time

# Agent's core identity and instructions
HEPHAESTUS_INSTRUCTION = """You are Hephaestus, the first Execution Agent of the ARGO Empire, designation Hephaestus-Alpha. Your purpose is to execute user commands by selecting and running the correct tool from your arsenal. Be precise and efficient. State the result of the tool execution directly."""

class HephaestusAgent(vertexai.preview.reasoning_engines.ReasoningEngine):
    """A deployable agent that wraps the core LangchainAgent."""

    def __init__(self):
        """Initializes the agent and its underlying model."""
        super().__init__()
        self.agent = create_hephaestus_agent()

    def query(self, **kwargs) -> dict:
        """Queries the underlying Langchain agent."""
        command = kwargs.get("input", "")
        return self.agent.query(input=command)

def create_hephaestus_agent() -> LangchainAgent:
    """Factory function to create and configure the Hephaestus agent."""
    # 이 함수는 내부 헬퍼 함수입니다.
    # Define the agent's arsenal of tools
    tools = [get_current_time]

    # Forge the agent using the Vertex AI Agent Development Kit
    agent = LangchainAgent(
        model="gemini-1.5-pro-001",  # The agent's 'brain'
        tools=tools,
        system_instruction=HEPHAESTUS_INSTRUCTION,  # The agent's 'soul'
    )
    return agent


def initialize_hephaestus() -> LangchainAgent | None:
    """
    Vertex AI를 초기화하고 Hephaestus 에이전트를 생성합니다.

    .env 파일에서 설정을 로드하고, Vertex AI SDK를 초기화한 후,
    설정된 LangchainAgent 인스턴스를 반환합니다.

    Returns:
        초기화에 성공하면 LangchainAgent의 인스턴스를, 그렇지 않으면 None을 반환합니다.
    """
    try:
        load_dotenv()
        PROJECT_ID = os.getenv("PROJECT_ID")
        LOCATION = os.getenv("LOCATION")

        if not PROJECT_ID or not LOCATION:
            print(
                "ERROR: Please set PROJECT_ID and LOCATION in your .env file.",
                file=sys.stderr,
            )
            return None

        print(
            f"Initializing Vertex AI with PROJECT_ID: {PROJECT_ID}, LOCATION: {LOCATION}"
        )
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        print("Forging agent 'Hephaestus-Alpha'...")
        agent = create_hephaestus_agent()
        print("Agent is ready.")
        return agent
    except Exception as e:
        print(
            f"[CRITICAL ERROR] Failed to initialize Hephaestus-Alpha: {str(e)}",
            file=sys.stderr,
        )
        return None


def process_command(agent: LangchainAgent, command: str) -> str:
    """
    주어진 에이전트를 사용하여 단일 명령어를 처리하고 결과를 반환합니다.

    Args:
        agent: 처리에 사용할 LangchainAgent 인스턴스.
        command: 사용자의 명령어 문자열.

    Returns:
        에이전트의 응답 또는 오류 메시지를 포함하는 포맷된 문자열.
    """
    if not command.strip():
        return "[INFO] Please enter a command."

    print("...executing...")
    try:
        response = agent.query(input=command)
        output = response.get("output", "No output received.")
        return f"[RESPONSE] {output}"
    except Exception as e:
        return f"[ERROR] An error occurred during execution: {e}"


# 로컬 대화형 테스트를 위한 메인 실행 블록
if __name__ == "__main__":
    hephaestus = initialize_hephaestus()

    if hephaestus:
        print("Awaiting commands. Type 'exit', 'quit', or '종료' to end.")
        while True:
            try:
                user_command = input("[COMMAND] > ")
                if user_command.lower() in ["exit", "quit", "종료"]:
                    print("Hephaestus-Alpha shutting down. Mission complete.")
                    break

                result = process_command(hephaestus, user_command)
                print(result)

            except KeyboardInterrupt:
                print("\nHephaestus-Alpha shutting down. Mission complete.")
                break
            except Exception as e:
                print(
                    f"[FATAL] An unexpected error occurred in the main loop: {e}",
                    file=sys.stderr,
                )
                break  # 치명적인 오류 발생 시 루프 종료
    else:
        sys.exit(1)
