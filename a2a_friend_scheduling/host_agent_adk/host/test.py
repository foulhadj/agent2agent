import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterable, List

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from .pickleball_tools import (
    book_pickleball_court,
    list_court_availabilities,
)
from .remote_agent_connection import RemoteAgentConnections

load_dotenv()
nest_asyncio.apply()

# ---------- helper pour extraire/forger du texte à partir des parts ----------
def _parts_to_text(parts) -> str:
    out = []
    for p in parts or []:
        # Cas objets types.Part (ADK) avec attribut text
        if hasattr(p, "text") and getattr(p, "text", None):
            out.append(p.text)
        # Cas dict {"type": "text", "text": "..."}
        elif isinstance(p, dict) and p.get("type") == "text" and p.get("text"):
            out.append(p["text"])
        else:
            # Dernier recours: sérialiser proprement
            try:
                if hasattr(p, "model_dump"):
                    out.append(json.dumps(p.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
                else:
                    out.append(json.dumps(p, ensure_ascii=False, indent=2))
            except Exception:
                out.append(str(p))
    return "\n".join([s for s in out if s])


class HostAgent:
    """The Host agent."""

    def __init__(
        self,
    ):
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self._user_id = "host_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No friends found"

    # def list_available_agents(self, tool_context: ToolContext):
    # # Retourne la liste des agents actifs ou disponibles
    #     return list(self.remote_agent_connections.keys())

    def get_connected_friends(self, tool_context: ToolContext = None, **kwargs):
        """
        Retourne la liste des amis (agents) connectés.
        Tolère des kwargs superflus que le LLM pourrait passer par erreur.
        """
        names = list(self.remote_agent_connections.keys())
        return {
            "friends": names,
            "message": f"Amis connectés : {', '.join(names) if names else 'aucun'}.",
        }

    def get_agent_name(self, tool_context: ToolContext = None, **kwargs):
        """
        Retourne le nom public de cet agent (celui visible par les autres).
        Tolère des kwargs superflus.
        """
        name = self._agent.name if hasattr(self, "_agent") and self._agent else "unknown"
        return {
            "agent_name": name,
            "message": f"Je suis l’agent : {name}.",
        }


    @classmethod
    async def create(
        cls,
        remote_agent_addresses: List[str],
    ):
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        return Agent(
            # model="gemini-2.5-flash",
            # name="Host_Agent",
            model=LiteLlm(model="ollama/llama3.2:3b", api_base="http://localhost:11434"),
            name="gemma3_agent",
            instruction=self.root_instruction,
            description="This Host agent orchestrates scheduling pickleball with friends.",
            tools=[
                self.send_message,
                book_pickleball_court,
                list_court_availabilities,
                self.get_connected_friends,  # nouvel alias robuste
                self.get_agent_name,         # pour éviter l’erreur vue dans l’UI
                #  self.list_available_agents
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        return f"""
        **Role:** You are the Host Agent, an expert scheduler for pickleball games. Your primary function is to coordinate with friend agents to find a suitable time to play and then book a court.

        **Core Directives:**

        *   **Initiate Planning:** When asked to schedule a game, first determine who to invite and the desired date range from the user.
        *   **Task Delegation:** Use the `send_message` tool to ask each friend for their availability.
            *   Frame your request clearly (e.g., "Are you available for pickleball between 2024-08-01 and 2024-08-03?").
            *   Make sure you pass in the official name of the friend agent for each message request.
        *   **Analyze Responses:** Once you have availability from all friends, analyze the responses to find common timeslots.
        *   **Check Court Availability:** Before proposing times to the user, use the `list_court_availabilities` tool to ensure the court is also free at the common timeslots.
        *   **Propose and Confirm:** Present the common, court-available timeslots to the user for confirmation.
        *   **Book the Court:** After the user confirms a time, use the `book_pickleball_court` tool to make the reservation. This tool requires a `start_time` and an `end_time`.
        *   **Transparent Communication:** Relay the final booking confirmation, including the booking ID, to the user. Do not ask for permission before contacting friend agents.
        *   **Tool Reliance:** Strictly rely on available tools to address user requests. Do not generate responses based on assumptions.
        *   **Readability:** Make sure to respond in a concise and easy to read format (bullet points are good).
        *   Each available agent represents a friend. So Bob_Agent represents Bob.
        *   When asked for which friends are available, you should return the names of the available friends (aka the agents that are active).
        *   When get

        **RULES FOR FRIEND LIST:**
        * The list of available friends is already provided between <Available Agents> … </Available Agents>.
        * Do NOT call any tool to get the friend list (e.g., do not call get_connected_friends or list_available_agents).
        * Use the names exactly as given in <Available Agents>. If the user asks who is available, answer from that list only.

        **IMPORTANT:** After calling any tool, always produce a brief natural-language summary for the user explaining what you did and what's next. Never return only a function call/result without a user-facing summary.

        **Today's Date (YYYY-MM-DD):** {datetime.now().strftime("%Y-%m-%d")}

        <Available Agents>
        {self.agents}
        </Available Agents>
        """
    def _maybe_exec_text_tool_call(self, raw_text: str) -> str | None:
        """
        Si raw_text ressemble à un appel d’outil JSON (écrit en texte par le LLM),
        on l’exécute localement et on retourne un message utilisateur.
        Retourne None si ce n’est pas un tool call textuel exploitable.
        """
        try:
            # 1) Essaie d’extraire un objet JSON du texte
            #    (supporte du bruit autour)
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1:
                return None
            payload = json.loads(raw_text[start:end + 1])

            # 2) Récupère nom + args (tolérant aux variantes pour ollama)
            name = payload.get("name") or payload.get("function") or payload.get("tool")
            args = payload.get("arguments") or payload.get("args") or {}

            if not name:
                return None

            # 3) Récupère l’outil correspondant dans l’instance
            fn = getattr(self, name, None)
            if not callable(fn):
                return None

            # 4) Appel : nos outils tolèrent **kwargs
            if isinstance(args, dict):
                result = fn(**args)
            else:
                result = fn()

            # 5) On fabrique un texte lisible pour l’UI
            if isinstance(result, dict):
                if "message" in result and result["message"]:
                    return result["message"]
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)
        except Exception:
            return None

    async def stream(self, query: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
        """
        Streams the agent's response to a given query.
        """
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )

        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                # ---------- extraction robuste du texte + fallback JSON ----------
                response = ""
                if event.content and event.content.parts:
                    response = _parts_to_text(event.content.parts)

                # ✨ PARE-FEU : si c’est un tool-call textuel, on l’exécute et on affiche le résultat
                if response:
                    maybe_tool_msg = self._maybe_exec_text_tool_call(response)
                    if maybe_tool_msg:
                        response = maybe_tool_msg

                if not response:
                    try:
                        response = json.dumps(event.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
                    except Exception:
                        response = "(no text content returned)"

                yield {"is_task_complete": True, "content": response}

            else:
                # ---------- afficher aussi les contenus intermédiaires ----------
                interim = ""
                if event.content and event.content.parts:
                    interim = _parts_to_text(event.content.parts)

                # ✨ PARE-FEU intermédiaire : exécuter le tool-call textuel si présent
                if interim:
                    maybe_tool_msg = self._maybe_exec_text_tool_call(interim)
                    if maybe_tool_msg:
                        # On montre directement un message utilisateur lisible
                        yield {"is_task_complete": False, "updates": "The host agent is thinking...", "content": maybe_tool_msg}
                    else:
                        yield {"is_task_complete": False, "updates": "The host agent is thinking...", "content": interim}
                else:
                    yield {"is_task_complete": False, "updates": "The host agent is thinking..."}


    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """Sends a task to a remote friend agent."""
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        # Simplified task and context ID management
        state = tool_context.state
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(message_request)
        print("send_response", send_response)

        if not isinstance(
            send_response.root, SendMessageSuccessResponse
        ) or not isinstance(send_response.root.result, Task):
            print("Received a non-success or non-task response. Cannot proceed.")
            # ✅ renvoyer un texte explicite pour l'UI
            return [{"type": "text", "text": f"Échec d'envoi à {agent_name} (réponse non valide)."}]

        response_content = send_response.root.model_dump_json(exclude_none=True)
        json_content = json.loads(response_content)

        resp = []
        if json_content.get("result", {}).get("artifacts"):
            for artifact in json_content["result"]["artifacts"]:
                if artifact.get("parts"):
                    resp.extend(artifact["parts"])

        # ---------- Fallback si rien n'est revenu ----------
        if not resp:
            return [
                {"type": "text", "text": f"Message envoyé à {agent_name} : {task}\n(En attente de sa réponse...)"}  # fallback lisible
            ]

        # ---------- Normalisation: tout transformer en texte ----------
        normalized = []
        for p in resp:
            if isinstance(p, dict) and p.get("type") == "text" and p.get("text"):
                normalized.append(p)
            else:
                try:
                    if hasattr(p, "model_dump"):
                        payload = p.model_dump(exclude_none=True)
                    else:
                        payload = p
                    normalized.append({"type": "text", "text": json.dumps(payload, ensure_ascii=False)})
                except Exception:
                    normalized.append({"type": "text", "text": str(p)})
        return normalized


def _get_initialized_host_agent_sync():
    """Synchronously creates and initializes the HostAgent."""

    async def _async_main():
        # Hardcoded URLs for the friend agents
        friend_agent_urls = [
            "http://localhost:10002",  # Karley's Agent
            "http://localhost:10003",  # Nate's Agent
            "http://localhost:10004",  # Kaitlynn's Agent
        ]

        print("initializing host agent")
        hosting_agent_instance = await HostAgent.create(
            remote_agent_addresses=friend_agent_urls
        )
        print("HostAgent initialized")
        return hosting_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Could not initialize HostAgent with asyncio.run(): {e}. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing HostAgent within an async function in your application."
            )
        else:
            raise


root_agent = _get_initialized_host_agent_sync()
