#!/usr/bin/env python
# coding: utf-8
"""
Utility functions for agent operations.
"""

import copy

from agno.agent import Agent


def get_last_agent_reply(agent: Agent) -> str:
    """Extract the content of the last message from an agent's session."""
    return copy.deepcopy(agent.get_messages_for_session()[-1].to_dict()["content"])
