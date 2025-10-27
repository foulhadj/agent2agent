import logging
import traceback

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    TaskState,
    TextPart,
)
from app.agent import KaitlynAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KaitlynAgentExecutor(AgentExecutor):
    """Kaitlyn's Scheduling AgentExecutor."""

    def __init__(self):
        self.agent = KaitlynAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if not context.task_id or not context.context_id:
            # On publie une erreur lisible dans le Task plutôt que lever une exception
            await self._submit_failed_task(
                event_queue,
                task_id=context.task_id or "unknown-task",
                context_id=context.context_id or "unknown-context",
                error_msg="Missing task_id or context_id",
            )
            return

        if not context.message:
            await self._submit_failed_task(
                event_queue,
                task_id=context.task_id,
                context_id=context.context_id,
                error_msg="Missing message in RequestContext",
            )
            return

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # ✅ Toujours SUBMIT pour garantir l’existence du Task (idempotent côté store)
        await updater.submit()
        await updater.start_work()

        query = context.get_user_input()

        try:
            async for item in self.agent.stream(query, context.context_id):
                is_task_complete = item.get("is_task_complete", False)
                require_user_input = item.get("require_user_input", False)
                content = item.get("content", "")
                parts = [Part(root=TextPart(text=content or ""))]

                if not is_task_complete and not require_user_input:
                    # statut "working" intermédiaire avec message agent
                    await updater.update_status(
                        TaskState.working,
                        message=updater.new_agent_message(parts),
                    )
                elif require_user_input:
                    # statut "input_required" + dernier message
                    await updater.update_status(
                        TaskState.input_required,
                        message=updater.new_agent_message(parts),
                    )
                    # On s’arrête : le client côté Host verra quand même un Task
                    return
                else:
                    # ✅ FIN NORMALE : on joint un artifact textuel + complete
                    await updater.add_artifact(
                        parts,
                        name="scheduling_result",
                    )
                    await updater.complete()
                    return

            # Si le stream se termine sans brancher "complete" ni "input_required"
            await updater.update_status(
                TaskState.failed,
                message=updater.new_agent_message(
                    [Part(root=TextPart(text="No final response produced by agent."))]
                ),
            )
            return

        except Exception as e:
            tb = traceback.format_exc()
            err_text = f"Agent execution error: {e}\n{tb}"
            # ❌ Ne PAS raise → ✅ publier un Task en état failed, avec texte
            await updater.update_status(
                TaskState.failed,
                message=updater.new_agent_message([Part(root=TextPart(text=err_text))]),
            )
            return

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # On peut publier un état failed/cancelled ici si nécessaire
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.update_status(
            TaskState.failed,
            message=updater.new_agent_message([Part(root=TextPart(text="Cancelled."))]),
        )

    async def _submit_failed_task(
        self,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        error_msg: str,
    ) -> None:
        """Crée un Task en erreur, lisible côté client, au lieu de lever une exception."""
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.submit()
        await updater.update_status(
            TaskState.failed,
            message=updater.new_agent_message([Part(root=TextPart(text=error_msg))]),
        )
