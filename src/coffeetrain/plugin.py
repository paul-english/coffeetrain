#!/usr/bin/env python3
import logging

from coffeetrain.runtime import RuntimeInterfaceMixin

log = logging.getLogger(__name__)

class Plugin(RuntimeInterfaceMixin):
    """
    A plugin is a class based way to extend coffeetrain.Trainer. It allows
    registering events, commands, components, and systems just like you
    can with the decorators.

    Usage:
    ```
    class SomePlugin(Plugin):
        ...

    trainer = Trainer(
        plugins=[SomePlugin(),]
    )
    # Or alternatively
    # trainer.register_plugin(SomePlugin())
    ```

    Alternatively if you prefer the decorator style (I do):
    ```
    some_plugin = Plugin()

    @some_plugin.command()
    def some_command():
        ...

    @some_plugin.component()
    def some_component():
        ...

    trainer = Trainer(
        plugins=[some_plugin,]
    )
    # Or alternatively
    # trainer.register_plugin(some_plugin)
    ```

    """

    name: str
    description: str

    def __init__(self, name, description=None):
        self.name = name
        self.description = description
        super().__init__()
