#!/usr/bin/env python3
import inspect
from typing import List, Union
import logging
from enum import Enum
from collections import defaultdict

class EventAlreadyExistsException(Exception):
    def __init__(self, msg='This event already exists. This is likely an error. You can silence this with `register_event(\'event\', allow_duplicate=True)`', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class RuntimeInterfaceMixin:
    """
    Allows a consistent and DRY mixin between the
    main Trainer runtime and Plugin.

    This is what allows our decorator based interface.
    """
    registered_components = None
    registered_systems = None
    registered_commands = None
    registered_events = None
    registered_overrides = None
    event_index = None
    hyperparams = None

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.registered_components = dict()
        self.registered_systems = dict()
        self.registered_commands = dict()
        self.registered_events = set()
        self.registered_overrides = dict()
        self.event_index = defaultdict(list)
        self.hyperparams = dict()

    def system(self, events: Union[str | List[str]]):
        """
        Decorator that allows tying algorithms into the training (or generally the event) loop
        """
        if isinstance(events, str):
            events = [events]

        def decorator(func):
            params = inspect.signature(func).parameters
            self.registered_systems[func.__name__] = {
                'func': func,
                'sigs': params,
            }
            for ev in events:
                if ev not in self.event_index:
                    self.log.warning(f'Event `{ev}` doesn\'t exist yet. Calls to this system will be dropped. Either register this event, or use the plugin that does register this event.')
                self._add_default_args_to_ctx(params)
                self.event_index[ev].append(func.__name__)
            return func
        return decorator

    def _add_default_args_to_ctx(self, params):
        for name, param in params.items():
            if param.default is not inspect.Parameter.empty:
                self.hyperparams[name] = param.default

    def event(self):
        """
        Decorator that registers new custom events. Must be uniquely named.
        """
        def decorator(obj):
            self.log.debug('Register events')
            if isinstance(obj, type) and issubclass(obj, Enum):
                for event in obj:
                    event_val = event.value
                    self.register_event(event_val)
                    self.register_event(f'{event_val}_BEFORE')
                    self.register_event(f'{event_val}_AFTER')
            elif isinstance(obj, str):
                self.register_event(obj)
            else:
                raise Exception(f"Invalid use of event decorator, unkown object: {obj}")
            return obj
        return decorator


    def command(self):
        """
        Decorator that allows custom commands, these often run smaller subsets
        of the training loop, e.g this can be used to implement inference
        or to run the eval phase on a single dataset with a saved checkpoint
        """
        def decorator(func):
            self.log.debug('Registering command', extra={'function_name': func.__name__, 'func': func})
            params = inspect.signature(func).parameters
            self.registered_commands[func.__name__] = {
                'func': func,
                'sigs': params,
            }
            self._add_default_args_to_ctx(params)

            return func
        return decorator

    def register_event(self, event: str, allow_duplicate=False):
        if not allow_duplicate and event in self.registered_events:
            raise EventAlreadyExistsException()
        self.log.debug(f'Registering event: {event}')
        self.registered_events.add(event)

    def execution_block(self, event, new_state=None, requires=None, call: bool = False):
        """
        Returns an async context manager class that can be used to
        auto-fire BEFORE, AFTER, and during events. Allows initializing
        state as though it happened in the first "BEFORE" system.

        If another plugin registered an `override_block` for this event, the
        override replaces the block body: the BEFORE/event systems still run,
        then the override runs (with dependencies injected from context),
        and the body is skipped via `SkipExecutionBlock`.
        """
        if new_state:
            self.set_state(new_state)
        event_key = event.value if hasattr(event, 'value') else event
        override = self.registered_overrides.get(event_key, None)
        if isinstance(requires, str):
            requires = [requires]
        get_state = getattr(self, 'get_state', None)
        block = ExecutionBlock(
            event_key, self.run_event,
            override=override,
            invoke_override=self._invoke_override,
            requires=requires,
            get_state=get_state,
        )
        if call:
            return block.execute()
        return block

    async def _invoke_override(self, func):
        """Run an override block, injecting dependencies from context.

        The base mixin has no context, so it calls the function as-is.
        `Trainer` overrides this to resolve parameters like a system.
        """
        if inspect.iscoroutinefunction(func):
            return await func()
        return func()

    @staticmethod
    def _check_requires(event, requires, get_state):
        """Warn when `requires` state keys are missing after the main event."""
        if not requires or get_state is None:
            return
        try:
            event_name = event.value if hasattr(event, 'value') else event
        except Exception:
            event_name = event
        for key in requires:
            if get_state(key) is None:
                raise RuntimeError(
                    f"execution_block('{event_name}') requires state `{key}`, "
                    "but it is missing or None after the event ran."
                )

    def override_block(self, event):
        """
        Allows a different plugin to register to override the
        entire execution of another's block.
        """
        def decorator(func):
            event_key = event.value if hasattr(event, 'value') else event
            if event_key in self.registered_overrides:
                raise Exception('Only one plugin may override another block at a time.')
            self.registered_overrides[event_key] = func
            return func

        return decorator

class SkipExecutionBlock(Exception):
    """Internal exception to skip the content of an execution_block."""

class ExecutionBlock:
    def __init__(self, event, run_event, override=None, invoke_override=None,
                 requires=None, get_state=None):
        self.event = event
        self.run_event = run_event
        self.override = override
        self.invoke_override = invoke_override
        self.requires = requires
        self.get_state = get_state
        self._after_ran = False

    def __enter__(self):
        raise NotImplementedError("We haven't implemented synchronous context for the execution_block yet, did you forget to put `async` on your with?")

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError(
            "Use `async with` for execution_block."
        )

    async def execute(self):
        """Execute the complete block without a user-supplied body.

        This is the path used by ``call=True`` and by loops that need a
        plugin override to replace their body. An override replaces the
        event's main systems; BEFORE and AFTER hooks still run exactly once.
        """
        await self.run_event(self.event, before=True)
        try:
            if self.override is not None:
                if self.invoke_override is not None:
                    await self.invoke_override(self.override)
                else:
                    result = self.override()
                    if inspect.isawaitable(result):
                        await result
            else:
                await self.run_event(self.event)
            RuntimeInterfaceMixin._check_requires(
                self.event, self.requires, self.get_state
            )
        finally:
            await self.run_event(self.event, after=True)

    async def __aenter__(self):
        self._after_ran = False
        if self.override is not None:
            # Python cannot skip an async-with body from __aenter__ while
            # still entering __aexit__. Execute the replacement and expose a
            # sentinel for callers that need to catch the skipped body.
            await self.execute()
            self._after_ran = True
            raise SkipExecutionBlock()

        await self.run_event(self.event, before=True)
        await self.run_event(self.event)
        RuntimeInterfaceMixin._check_requires(self.event, self.requires, self.get_state)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, '_after_ran', False):
            return exc_type is SkipExecutionBlock
        await self.run_event(self.event, after=True)
        if exc_type is SkipExecutionBlock:
            return True # Suppress our internal exception
        return False # propagate real errors
