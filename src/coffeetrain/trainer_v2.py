#!/usr/bin/env python3
import os
import asyncio
import functools
import sys
import inspect
import logging
import signal
from datetime import datetime
from typing import List, Union, get_type_hints, get_origin, get_args, Literal
from pathlib import Path
from enum import Enum


from textwrap import dedent # Can use d""" ... """ in py 3.15
from coffeetrain.runtime import RuntimeInterfaceMixin
from coffeetrain.plugin import Plugin

__ALL__ = ['TrainerV2']

log = logging.getLogger('TrainerV2')

class PluginMergeConflict(Exception):
    ...

class ColoredExtraFormatter(logging.Formatter):
    # ANSI Escape Codes for colors
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    CYAN = "\x1b[36;20m"
    RESET = "\x1b[0m"

    # Map levels to colors
    COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    # Dynamically find all standard attributes for THIS version of Python
    _dummy_record = logging.LogRecord("name", logging.INFO, "path", 1, "msg", None, None)
    STANDARD_ATTRS = set(_dummy_record.__dict__.keys())

    def format(self, record):
        # 1. Identify and format the 'extra' dict
        extra_data = {
            k: v for k, v in record.__dict__.items()
            if k not in self.STANDARD_ATTRS
        }

        # We'll make the extra dictionary bright Green for visibility
        red = self.RED
        green = "\x1b[32;20m"
        record.extra_str = '' if not extra_data else f" {red}extra={green}{str(extra_data)}{self.RESET}" if extra_data else "{}"

        # 2. Apply color to the main log level
        log_fmt = self.COLORS.get(record.levelno)
        format_str = f"{log_fmt}%(levelname)s{self.RESET} - %(message)s%(extra_str)s"

        # 3. Create a temporary formatter to handle the actual string interpolation
        formatter = logging.Formatter(format_str)
        return formatter.format(record)


class TrainerV2(RuntimeInterfaceMixin):
    """
    This iterations jumps up a step beyond callback driven composable training
    into more of a full runtime with an ECS style for composable training.

    # TODO with async, callbacks can run sort of concurrently more/less, e.g. plots run while
    # loop gets back to the train batch

    # TODO support a dry-run arg, that plays the events without
    # executing systems

    ```
    from coffeetrain import TrainerV2
    trainer = TrainerV2()

    @trainer.component
    def train_dataset():
        load_dataset('mnist')

    @trainer.component
    def model():
        return MnistModel()

    @trainer.system('FIT_START')
    def start_messages(model: MnistModel):
        log.info(f'''
        Beginning trainer
        {model}
        ''')

    if __name__ = '__main__':
        trainer()
    ```
    """

    context = None

    # TODO should this be changed to __await__? full async class & stuff?
    def __init__(self):
        """
        For now we implement "The Default Loop", where init starts
        us with the core components and systems that are the most common
        for a standard training loop. This gets extended by the user
        to implement their training algorithm. This can be disabled with
        a blank profile to get a no-op training loop.

        NOTE It's very likely that plugins will always assume that
        the events from our default plugin set exist in your trainer
        instance. If you want to implement your own custom training
        loop, it's best practice to match our event names.
        """
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredExtraFormatter())
        self.log = log
        self.log.propagate = False
        self.log.addHandler(handler)
        self.log.setLevel(
            os.environ.get('LOGLEVEL', 'INFO').upper()
        )

        super().__init__()

        self.context = {}
        self.register_event('INIT')

        from coffeetrain.plugins import default_plugins
        self.register_plugin(default_plugins)

        # Is this going to be the only baked in event? sure.. why not
        self.sync_run_event('INIT')

    def __call__(self):
        """
        The "main" method
        Usage:
        ```
        trainer = Trainer()
        if __name__ == '__main__':
            trainer()
        ```
        """
        asyncio.run(self.__async_call__())

    async def __async_call__(self):
        await self.dispatch()

    async def _execute(self, func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # TODO allow enabling warnings for calling out non async fns
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                functools.partial(func, *args, **kwargs)
            )

    async def display_help(self, command=None):
        # TODO get current python script name

        # generates COMMANDS
        commands = [
            f'{name}: {details["func"].__doc__}'
            for name, details in self.registered_commands.items()
        ]
        commands = "\n".join(commands)

        # TODO generates ARGS

        print(dedent(f"""
        Usage: python {{script.py}} [COMMAND] [ARGS]

        Commands:
        {commands}
        """))

    async def dispatch(self):
        self.log.debug(f'Dispatch start', extra={'argv': sys.argv})
        if len(sys.argv) < 2:
            await self.display_help()
            sys.exit(1)

        cmd_name = sys.argv[1]
        if len(sys.argv) < 3:
            # no args
            args = []
        else:
            args = sys.argv[2]

        if cmd_name in self.registered_commands:
            cmd_info = self.registered_commands[cmd_name]
            func = cmd_info["func"]
            params = cmd_info["sigs"]
            parsed_args = self._collect_kwargs_from_ctx(func, params)
            self.log.debug(f'Dispatching command: `{cmd_name}`', extra={'parsed_args': parsed_args})
            return await self._execute(func, **parsed_args)
        else:
            print(f"Unknown command: {cmd_name}")

    def register_plugin(self, plugins: Union[Plugin | List[Plugin]]):
        if isinstance(plugins, Plugin):
            plugins = [plugins]

        def merge(plugin, merge_name, a, b):
            """
            A merge that errors on conflict.
            """
            # Check for common keys using set intersection
            if hasattr(a, 'keys'):
                conflicts = a.keys() & b.keys()
            else:
                conflicts = a & b

            if not conflicts:
                self.log.debug(f'Merging {merge_name} from plugin {plugin.name}, new keys: {b}')
                a |= b
            else:
                raise PluginMergeConflict(f"Registering plugin `{plugin.name}` failed: Duplicate {merge_name} names found: {conflicts}.")

        def deepmerge_defaultlists(plugin, merge_name, a, b):
            names = set(a.keys()) | set(b.keys())
            self.log.debug(f'Merging event index', extra={'event_index': b})
            for name in names:
                a[name] += b[name]

        for plugin in plugins:
            self.log.debug(f'Registering plugin: {plugin.name}')
            merge(plugin, 'components', self.registered_components, plugin.registered_components)
            merge(plugin, 'systems', self.registered_systems, plugin.registered_systems)
            merge(plugin, 'commands', self.registered_commands, plugin.registered_commands)
            merge(plugin, 'events', self.registered_events, plugin.registered_events)
            merge(plugin, 'overrides', self.registered_overrides, plugin.registered_overrides)
            deepmerge_defaultlists(plugin, 'event_index', self.event_index, plugin.event_index)
            merge(plugin, 'hyperparams', self.hyperparams, plugin.hyperparams)

    def get_state(self, name):
        """
        Allows callbacks to access up to date state values in the middle of execution. A
        callback may have triggered events or other side effects could have occurred that
        prevent you from accessing this value in the callback args. Most often this happens
        in a command.
        """
        return self.context.get(name)

    def set_state(self, *args, **kwargs):
        """
        Allows
        """
        if 'name' in kwargs and 'value' in kwargs:
            self.set_state(name, value)
        elif len(args) == 2:
            k, v = args
            self.log.debug(f'Set state: `{k}`')
            self.context[k] = v
        elif isinstance(args[0], dict):
            for k,v in args[0].items():
                self.set_state(k, v)
        elif len(kwargs):
            for k,v in kwargs.items():
                self.set_state(k, v)
        else:
            raise Exception(f"Unknown values passed to `set_state`: `{args}` & `{kwargs}`. Either pass a key & value to set_state, or a dictionary of state values to change.")
        return

    def sync_run_event(self, event: str):
        asyncio.run(self.run_event(event))

    def _collect_kwargs_from_ctx(self, func, params):
        local_context = {
            'log': self.log.getChild(func.__name__),
            'run_event': self.run_event,
            'get_state': self.get_state,
            'set_state': self.set_state,
            'execution_block': self.execution_block,
            **self.hyperparams,
            **self.context,
        }
        hints = get_type_hints(func)
        # Basic type casting logic
        parsed_args = {}
        for i, (name, param) in enumerate(params.items()):
            if name == 'kwargs':
                parsed_args['kwargs'] = local_context
            elif name == 'args':
                self.log.warning(f'We don\'t support commands with the `*args` argument. Remove `*args` from the command {name}')
            elif name in local_context:
                raw_val = local_context[name]
                ann = hints.get(name)
                if ann:
                    cast_val = custom_coerce(raw_val, ann)
                    parsed_args[name] = cast_val
                else:
                    parsed_args[name] = raw_val
            elif param.default is not inspect.Parameter.empty:
                parsed_args[name] = param.default
            else:
                self.log.error(f'Argument not found in the context: `{name}`')
        return parsed_args


    async def run_event(self, event: str, before: bool = False, after: bool = False):
        """
        Triggers all the systems tied to this hook
        """
        if hasattr(event, 'value'):
            event = event.value
        if before:
            event = f'{event}_BEFORE'
        elif after:
            event = f'{event}_AFTER'
        self.log.debug(f'Running event `{event}`')
        system_names = self.event_index[event]
        for system_name in system_names:
            system = self.registered_systems[system_name]
            func = system['func']
            params = system['sigs']
            kwargs = self._collect_kwargs_from_ctx(func, params)
            self.log.debug(f'Running system `{system_name}`', extra={'kwargs': kwargs})
            updated_state = await self._execute(func, **kwargs)
            if updated_state is not None:
                self.set_state(updated_state)


def _coerce_bool(raw: str) -> bool:
    if isinstance(raw, bool):
        return raw
    low = raw.lower()
    if low in ("1", "true", "yes", "y", "on"):
        return True
    if low in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{raw!r} is not a valid bool")


COERCERS = {
    str: str,
    int: int,
    float: float,
    bool: _coerce_bool,
    Path: Path,
    datetime: datetime.fromisoformat,
}


def custom_coerce(value, annotation):
    # Unwrap Optional[X] / Union[X, None]
    origin = get_origin(annotation)

    if origin is Literal:
        allowed = get_args(annotation)

        # Fast path: raw value already matches one of the literals
        if value in allowed:
            return value

        # Try coercing the raw string to the type of each literal
        # (handles Literal[1, 2] where value comes in as "1")
        for lit in allowed:
            try:
                cast = type(lit)(value)
            except (TypeError, ValueError):
                continue
            if cast == lit:
                return cast

        raise ValueError(
            f"{value!r} is not one of {allowed!r}"
        )

    if origin is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            if value is None:
                return None
            return coerce(value, non_none[0])

    # list[X]  -- expects value to already be a list of raw strings
    if origin in (list,):
        (inner,) = get_args(annotation)
        return [coerce(v, inner) for v in value]

    # Enums: match by member name
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)          # match by value
        except ValueError:
            return annotation[value]          # match by name

    # Plain registered types
    if annotation in COERCERS:
        return COERCERS[annotation](value)

    # Fallback: just call the type
    return annotation(value)
