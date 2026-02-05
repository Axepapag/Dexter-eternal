#!/usr/bin/env python3
"""
Agent Collaboration for Dexter
Multi-agent messaging, task delegation, and coordination
Enables Dexter (lead orchestrator) to work with D2, Antigravity, and future agents
"""

import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime

__tool_prefix__ = "agent"

# Agent endpoints configuration
AGENT_ENDPOINTS = {
    "d2": {
        "url": "http://localhost:8010",
        "chat_endpoint": "/chat",
        "status_endpoint": "/status",
        "type": "http",
        "description": "D2 Autonomous Agent - Long-running projects, deep tool integration",
        "capabilities": ["autonomous_execution", "file_operations", "web_research"]
    },
    "antigravity": {
        "url": "http://localhost:8002",
        "chat_endpoint": "/chat", 
        "status_endpoint": "/status",
        "type": "http",
        "description": "Antigravity Assistant - Quick tasks, web search",
        "capabilities": ["quick_queries", "web_search"]
    },
    "max": {
        "url": "http://localhost:8003",
        "chat_endpoint": "/chat",
        "status_endpoint": "/status", 
        "type": "http",
        "description": "Max Agent - Memory operations, learning",
        "capabilities": ["memory_management", "learning"]
    }
}

DEFAULT_TIMEOUT = 60


def agent_list() -> Dict[str, Any]:
    """
    List all available agents for collaboration with their status.
    """
    agents_info = []
    
    for agent_id, config in AGENT_ENDPOINTS.items():
        status = agent_check_status(agent_id)
        agents_info.append({
            "id": agent_id,
            "description": config["description"],
            "endpoint": config["url"],
            "capabilities": config.get("capabilities", []),
            "status": status.get("status", "unknown"),
            "online": status.get("success", False)
        })
    
    online_count = sum(1 for a in agents_info if a["online"])
    
    return {
        "success": True,
        "agents": agents_info,
        "total": len(agents_info),
        "online": online_count,
        "dexter_role": "lead_orchestrator",
        "message": f"{online_count}/{len(agents_info)} agents online"
    }


def agent_check_status(agent: str) -> Dict[str, Any]:
    """
    Check if an agent is online and get its status.
    
    Args:
        agent: Agent ID (d2, antigravity, max)
    """
    if agent not in AGENT_ENDPOINTS:
        return {
            "success": False,
            "error": f"Unknown agent: {agent}",
            "known_agents": list(AGENT_ENDPOINTS.keys())
        }
    
    config = AGENT_ENDPOINTS[agent]
    status_url = config["url"] + config.get("status_endpoint", "/status")
    
    try:
        response = requests.get(status_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "agent": agent,
                "status": "online",
                "details": data
            }
        else:
            return {
                "success": False,
                "agent": agent,
                "status": "error",
                "http_code": response.status_code
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "agent": agent,
            "status": "offline",
            "hint": f"Start {agent} at {config['url']} to enable collaboration"
        }
    except Exception as e:
        return {
            "success": False,
            "agent": agent,
            "status": "error",
            "error": str(e)
        }


def agent_send_message(to_agent: str, message: str, from_agent: str = "dexter",
                      context: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Send a message to another agent.
    
    Args:
        to_agent: Target agent ID (d2, antigravity, max)
        message: Message content
        from_agent: Your agent ID (default: dexter)
        context: Optional context/background information
        timeout: Request timeout in seconds
    """
    if to_agent not in AGENT_ENDPOINTS:
        return {
            "success": False,
            "error": f"Unknown agent: {to_agent}",
            "available_agents": list(AGENT_ENDPOINTS.keys())
        }
    
    # First check if agent is online
    status = agent_check_status(to_agent)
    if not status.get("success"):
        return {
            "success": False,
            "error": f"Agent {to_agent} is not online",
            "status": status.get("status"),
            "hint": f"Start {to_agent} before sending messages"
        }
    
    config = AGENT_ENDPOINTS[to_agent]
    chat_url = config["url"] + config.get("chat_endpoint", "/chat")
    
    # Construct full message
    full_message = message
    if context:
        full_message = f"[Context: {context}]\n\n{message}"
    
    payload = {
        "agent_id": from_agent,
        "message": full_message,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(
            chat_url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "to_agent": to_agent,
                "from_agent": from_agent,
                "response": data.get("response") or data.get("reply") or str(data),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "details": response.text[:500]
            }
    
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Timeout after {timeout}s - {to_agent} may be processing",
            "hint": "The agent may still be working on your request"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def agent_delegate_task(to_agent: str, task_description: str, from_agent: str = "dexter",
                       priority: str = "normal", context: Optional[str] = None) -> Dict[str, Any]:
    """
    Delegate a task to another agent for autonomous execution.
    
    Args:
        to_agent: Agent to delegate to (typically d2 for complex work)
        task_description: Complete task description with requirements
        from_agent: Your agent ID
        priority: Task priority (low, normal, high, urgent)
        context: Additional context
    """
    delegation_message = f"""TASK DELEGATION FROM {from_agent.upper()}
Priority: {priority.upper()}
Delegated at: {datetime.now().isoformat()}

TASK DESCRIPTION:
{task_description}

Please execute this task autonomously and report back with:
1. What you accomplished
2. Any issues encountered
3. Results/success metrics
4. Next steps (if any)

Execute immediately - do not ask clarifying questions unless blocked.
"""
    
    return agent_send_message(
        to_agent=to_agent,
        message=delegation_message,
        from_agent=from_agent,
        context=context,
        timeout=120  # Longer timeout for task delegation
    )


def agent_share_fact(to_agent: str, fact_key: str, fact_value: str, 
                    from_agent: str = "dexter", priority: bool = False) -> Dict[str, Any]:
    """
    Share a fact with another agent for them to remember.
    
    Args:
        to_agent: Agent to share with
        fact_key: Fact identifier/label
        fact_value: Fact content
        from_agent: Your agent ID
        priority: Whether this is a priority fact
    """
    priority_flag = " [PRIORITY]" if priority else ""
    message = f"STORE FACT{priority_flag}: {fact_key} = {fact_value}"
    
    return agent_send_message(
        to_agent=to_agent,
        message=message,
        from_agent=from_agent
    )


def agent_ask_d2(message: str, from_agent: str = "dexter", context: Optional[str] = None) -> Dict[str, Any]:
    """Shortcut to ask D2 a question."""
    return agent_send_message("d2", message, from_agent, context)


def agent_ask_antigravity(message: str, from_agent: str = "dexter", context: Optional[str] = None) -> Dict[str, Any]:
    """Shortcut to ask Antigravity a question."""
    return agent_send_message("antigravity", message, from_agent, context)


def agent_broadcast(message: str, from_agent: str = "dexter", 
                   exclude: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Broadcast a message to all online agents.
    
    Args:
        message: Message to broadcast
        from_agent: Sending agent ID
        exclude: List of agent IDs to exclude
    """
    exclude = exclude or []
    results = {}
    
    for agent_id in AGENT_ENDPOINTS.keys():
        if agent_id not in exclude:
            result = agent_send_message(agent_id, message, from_agent)
            results[agent_id] = result
    
    successful = sum(1 for r in results.values() if r.get("success"))
    
    return {
        "success": successful > 0,
        "broadcast_to": list(results.keys()),
        "successful_deliveries": successful,
        "results": results
    }


def agent_coordination_meeting(topic: str, agenda: List[str], from_agent: str = "dexter") -> Dict[str, Any]:
    """
    Schedule/request a coordination meeting with all agents.
    
    Args:
        topic: Meeting topic
        agenda: List of agenda items
        from_agent: Calling agent
    """
    meeting_message = f"""COORDINATION MEETING REQUEST
Topic: {topic}
Called by: {from_agent}
Time: {datetime.now().isoformat()}

AGENDA:
"""
    for i, item in enumerate(agenda, 1):
        meeting_message += f"{i}. {item}\n"
    
    meeting_message += "\nPlease acknowledge receipt and prepare your input."
    
    return agent_broadcast(meeting_message, from_agent)


# Backwards compatibility aliases
send_message = agent_send_message
delegate_task = agent_delegate_task
share_fact = agent_share_fact
check_status = agent_check_status
list_agents = agent_list
