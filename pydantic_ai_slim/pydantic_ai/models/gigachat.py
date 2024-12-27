from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Generator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import Literal, Union

from typing_extensions import assert_never

from .. import result
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..result import Usage
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    AgentModel,
    EitherStreamedResponse,
    Model,
)

try:
    import gigachat.models as giga_chat
    from gigachat.client import GigaChatAsyncClient
    from gigachat.models.chat_function_call import ChatFunctionCall
except ImportError as _import_error:
    raise ImportError(
        'Please install `gigachat` to use the GigaChat model, '
        "you can use the `gigachat` optional group — `pip install 'pydantic-ai-slim[gigachat]'`"
    ) from _import_error


CommonGigaChatModelNames = Literal['GigaChat', 'GigaChat-Max', 'GigaChat-Plus', 'GigaChat-Pro']
GigaChatModelName = Union[CommonGigaChatModelNames, str]


@dataclass(init=False)
class GigaChatModel(Model):
    """GigaChat model impl."""

    model_name: GigaChatModelName
    client: GigaChatAsyncClient

    def __init__(self, model_name: GigaChatModelName, *, api_key: str | None = None):
        self.model_name = model_name
        self.client = GigaChatAsyncClient(credentials=api_key, verify_ssl_certs=False, scope='GIGACHAT_API_PERS')

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return GigaChatAgentModel(self.client, self.model_name, allow_text_result, tools)

    def name(self) -> str:
        return f'sber:{self.model_name}'

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> giga_chat.Function:
        return giga_chat.Function(
            name=f.name,
            description=f.description,
            parameters=giga_chat.FunctionParameters(
                type='object',
                properties=f.parameters_json_schema.get('properties'),
                required=f.parameters_json_schema.get('required'),
            ),
        )


@dataclass
class GigaChatAgentModel(AgentModel):
    """Implementation of `AgentModel` for GigaChat models."""

    client: GigaChatAsyncClient
    model_name: GigaChatModelName
    allow_text_result: bool
    tools: list[giga_chat.Function]

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, Usage]:
        """Make a request to the model."""
        response = await self._completions_create(messages, False)
        return self._process_response(response), _map_cost(response)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncIterator[EitherStreamedResponse]:
        """Make a request to the model and return a streaming response."""
        raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
        # yield is required to make this a generator for type checking
        # noinspection PyUnreachableCode
        yield  # pragma: no cover

    async def _completions_create(self, messages: list[ModelMessage], stream: bool) -> giga_chat.ChatCompletion:
        # standalone function to make it easier to override
        if not self.tools:
            function_call: Literal['none', 'auto'] | ChatFunctionCall | None = None
        elif not self.allow_text_result:
            function_call = ChatFunctionCall(name='final_result')
        else:
            function_call = 'auto'

        gigachat_messages = list(chain(*(self._map_message(m) for m in messages)))
        return await self.client.achat(
            giga_chat.Chat(
                model=self.model_name,
                messages=gigachat_messages,
                n=1,
                functions=self.tools,
                function_call=function_call,
                stream=stream,
            )
        )

    @staticmethod
    def _process_response(response: giga_chat.ChatCompletion) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []

        if len(choice.message.content) != 0:
            items.append(TextPart(choice.message.content))
        if choice.message.function_call is not None and choice.message.function_call.arguments is not None:
            items.append(
                ToolCallPart.from_raw_args(
                    tool_name=choice.message.function_call.name, args=choice.message.function_call.arguments
                )
            )
        return ModelResponse(items, timestamp=timestamp)

    @classmethod
    def _map_message(cls, message: ModelMessage) -> Generator[giga_chat.Messages, None, None]:
        """Just maps a `pydantic_ai.Message` to a `gigachat.models.Messages`."""
        if isinstance(message, ModelRequest):
            yield from cls._map_user_message(message)
        elif isinstance(message, ModelResponse):
            for item in message.parts:
                if isinstance(item, TextPart):
                    yield giga_chat.Messages(role=giga_chat.MessagesRole.ASSISTANT, content=item.content)
                elif isinstance(item, ToolCallPart):
                    yield giga_chat.Messages(role=giga_chat.MessagesRole.ASSISTANT, function_call=_map_tool_call(item))
                else:
                    assert_never(item)
        else:
            assert_never(message)

    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Iterable[giga_chat.Messages]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield giga_chat.Messages(role=giga_chat.MessagesRole.SYSTEM, content=part.content)
            elif isinstance(part, UserPromptPart):
                yield giga_chat.Messages(role=giga_chat.MessagesRole.USER, content=part.content)
            elif isinstance(part, ToolReturnPart):
                yield giga_chat.Messages(
                    role=giga_chat.MessagesRole.FUNCTION,
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield giga_chat.Messages(role=giga_chat.MessagesRole.USER, content=part.model_response())
                else:
                    yield giga_chat.Messages(
                        role=giga_chat.MessagesRole.FUNCTION,
                        content=part.model_response(),
                    )
            else:
                assert_never(part)


def _map_tool_call(t: ToolCallPart) -> giga_chat.FunctionCall:
    return giga_chat.FunctionCall(name=t.tool_name, arguments=t.args_as_dict())


def _map_cost(response: giga_chat.ChatCompletion) -> result.Usage:
    usage = response.usage
    details: dict[str, int] = {}
    return result.Usage(
        request_tokens=usage.prompt_tokens,
        response_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        details=details,
    )
