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
            if issubclass(obj, Enum):
                for event in obj:
                    event_val = event.value
                    self.register_event(event_val)
                    self.register_event(f'{event_val}_BEFORE')
                    self.register_event(f'{event_val}_AFTER')
            elif isinstance(obj, str):
                self.register_event(event.value)
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
        # Doesn't matter if it already exists, but if two plugins add the same
        # event, they probably also want to call that event as well which
        # could lead to unintended behavior.
        # We can early terminate since it's likely a bug (maybe we add a way to skip
        # this if the user knows what they're doing)
        if not allow_duplicate:
            if event in self.registered_events:
                raise EventAlreadyExistsException()
        self.log.debug(f'Registering event: {event}')
        self.registered_events.add(event)

    def execution_block(self, event, new_state=None, requires=None, call: bool = False):
        """
        Returns an async context manager class that can be used to
        auto-fire BEFORE, AFTER, and during events. Allows initializing
        state as though it happened in the first "BEFORE" system.
        """
        # TODO if requires, we need to assert that the state was changed
        # before we call the AFTER event
        if new_state:
            self.set_state(new_state)
        override = self.registered_overrides.get(event, None)
        if call:
            async def call_execution_block():
                await self.run_event(event, before=True)
                await self.run_event(event)
                if override is not None:
                    override()
                await self.run_event(event, after=True)
            return call_execution_block()
        return ExecutionBlock(event, self.run_event, override=override)

    def override_block(self):
        """
        Allows a different plugin to register to override the
        entire execution of another's block.
        """
        def decorator(func, event):
            if event in self.registered_overrides:
                raise Exception('Only one plugin may override another block at a time.')
            self.registered_overrides[event] = func
            return OverrideBlock()

        return decorator

class SkipExecutionBlock(Exception):
    """Internal exception to skip the content of an execution_block."""

class ExecutionBlock:
    def __init__(self, event, run_event, override=None):
        self.event = event
        self.run_event = run_event
        self.override = override

    def __enter__(self):
        raise NotImplementedError("We haven't implemented synchronous context for the execution_block yet, did you forget to put `async` on your with?")

    def __exit__(self):
        ...

    async def __aenter__(self):
        await self.run_event(self.event, before=True)
        await self.run_event(self.event)
        if self.override is not None:
            self.override()
            raise SkipBlock()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.run_event(self.event, after=True)
        if exc_type is SkipExecutionBlock:
            return True # Suppress our internal exception
        return False # propagate real errors


class OverrideBlock:
    """
    Basically a no-op (context manager wise), this lets it'
    block of context run, the parent execution block is
    instructed to skip it's execution.
    """
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
