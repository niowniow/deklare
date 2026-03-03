"""
Copyright 2024 Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import functools
import inspect
import warnings
from copy import copy, deepcopy
from types import TracebackType
from typing import Any, Callable
from uuid import uuid4

import dask
import dask.delayed
from dask.delayed import Delayed
from dask.typing import Graph

from .descriptor import Descriptor

KEY_SEP = "+"
PROTECTED_DESCRIPTOR_KEYS = ["self", "config"]
PROTECTED_CONFIG_KEYS = ["global", "types", "keys"]


class FlowContext:
    __context: dict[str, Callable] = {}
    __enabled: bool = False

    @classmethod
    def get(cls, name: str) -> Callable:
        return cls.__context[name]

    @staticmethod
    def exists(name: str) -> bool:
        return name in FlowContext.__context

    @staticmethod
    def set(name: str, value: Callable) -> None:
        FlowContext.__context[name] = value

    @staticmethod
    def is_enabled() -> bool:
        return FlowContext.__enabled

    @staticmethod
    def set_enabled(value: bool) -> None:
        FlowContext.__enabled = value

    @staticmethod
    def reset() -> None:
        FlowContext.__enabled = False
        FlowContext.__context = {}


class TaskGraphCreator:
    prev: bool

    def __init__(self) -> None:
        self.prev = False

    def __enter__(self) -> None:
        global is_enabled
        FlowContext.reset()
        FlowContext.set_enabled(True)
        is_enabled = True

    def __exit__(
        self, _type: BaseException | None, _value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        global is_enabled
        is_enabled = False
        FlowContext.reset()


is_enabled = False


class Node(object):
    """node of dask task graph

    Attributes:
        _name (str | None): name of Node
        config (Descriptor): configuration of Node
    """

    _name: str | None
    config: Descriptor

    def __init__(self, config: Descriptor | None = None) -> None:  # noqa: ANN401
        self.config = config or Descriptor()
        self._name = None

    def merge_config(self, descriptor: Descriptor) -> Descriptor:
        """Each descriptor contains configuration which may apply to different
        node instances. This function collects all information that apply to _this_
        node (including it's preset configs) and adds a `self` keyword to the descriptor.

        Args:
            descriptor (Descriptor): The descriptor and configuration options.

        Returns:
            Descriptor: A new descriptor which specific to this node.
        """
        new_descriptor = self._copy_descriptor(descriptor)

        self._update_descriptor_config(descriptor, new_descriptor)

        return new_descriptor

    def configure(self, descriptor: Descriptor) -> Descriptor:
        """Before a task graph is executed each node is configured.
        The descriptor is propagated from the end to the beginning
        of the DAG and each nodes "configure" routine is called.
        The descriptor can be updated to reflect additional requirements,
        The return value gets passed to predecessors.

        Essentially the following question must be answered within the
        nodes configure function:
        What do I need to fulfil the descriptor of my successor? Either the node
        can provide what is required or the descriptor is passed through to
        predecessors in hope they can fulfil the descriptor.

        Here, you must not configure the internal parameters of the
        Node otherwise it would not be thread-safe. You can however
        introduce a new key 'requires_descriptor' in the descriptor being
        returned. This descriptor will then be passed as an argument
        to the __call__ function.

        Best practice is to configure the Node on initialization with
        runtime independent configurations and define all runtime
        dependant configurations here.

        Args:
            descriptor (Descriptor): descriptor to merge with own config.

        Returns:
            Descriptor: The (updated) descriptor. If updated, modifications
                  must be made on a copy of the input. The return value
                  must be a Descriptor.
                  If multiple descriptors are input to this function they
                  must be merged.
                  If nothing needs to be descriptored an empty Descriptor
                  can be returned. This removes all dependencies of this
                  node from the task graph.
        """
        merged_descriptor = self.merge_config(descriptor)

        # set default
        merged_descriptor.config["requires_descriptor"] = True

        return merged_descriptor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """
        execute flow, possibly within a dask delayed context

        Args:
            *args (Any): positional arguments to pass to the underlying compute function
            **kwargs (Any): keyword arguments, may optionally include special args:
                - name (str): name for the task. must not contain `KEY_SEP`
                - context (FlowContext): execution context
                - descriptor (Descriptor) : configuration dictionary to merge with the instance's config

        Returns:
            Any: result of computation. if `context.is_enabled()`, returns corresponding `dask.delayed` object
                 else, returns the immediate result of `self.compute(*args, **kwargs)`

        Raises:
            RuntimeError: If kwargs["name"] contains the reserved separator `KEY_SEP`
        """

        name = kwargs.get("name", None)
        context = kwargs.get("context", None)
        if name is not None and KEY_SEP in name:
            raise RuntimeError(f"Do not use a `{KEY_SEP}` character in your {name=}")
        if name is None:
            name = self._name

        new_kwargs = copy(kwargs)
        if kwargs.get("descriptor", None) is not None:
            new_kwargs["descriptor"] = self.merge_config(kwargs["descriptor"])
        elif kwargs.get("descriptor", None) is not None:
            new_kwargs["descriptor"] = self.merge_config(kwargs["descriptor"])
        else:
            new_kwargs["descriptor"] = self.merge_config(Descriptor())

        if context is None:
            context = FlowContext

        forward_func = self.compute

        if context.is_enabled():
            func = dask.delayed(forward_func)(*args, dask_key_name=name, **kwargs)
            self.dask_key_name = func.key
            return func
        else:
            return forward_func(*args, **kwargs)

    def _copy_descriptor(self, descriptor: Descriptor) -> Descriptor:
        """copies descriptor with config of instance under "self" if present
        ignores PROTECTED_DESCRIPTOR_KEYS

        Args:
            descriptor (Descriptor): initial descriptor to copy

        Returns:
            Descriptor: copied and udpated descriptor
        """
        new_descriptor = deepcopy(descriptor)

        if hasattr(self, "config"):
            new_descriptor.config.update(deepcopy(self.config.config))

        new_descriptor._set_internal("self", {})
        new_descriptor._get_internal("self").update(self.config.model_dump(exclude=PROTECTED_DESCRIPTOR_KEYS))

        return new_descriptor

    def _update_descriptor_config(self, old_descriptor: Descriptor, new_descriptor: Descriptor) -> None:
        """Merge configuration from an old descriptor into a new descriptor's `self` parameters.

        Args:
            old_descriptor (Descriptor): old descriptor values to update new descriptor
            new_descriptor (Descriptor): new descriptor to update `self` parameters
        """
        if old_descriptor.config is not None:
            # go through all parameters in the descriptor's config and add them to the self parameters

            # assume anything within 'config' is global
            for key in old_descriptor.config:
                if key in PROTECTED_CONFIG_KEYS:
                    continue

                new_descriptor._get_internal("self")[key] = old_descriptor.config[key]

            # add specific global config entries
            if "global" in old_descriptor.config:
                for key in old_descriptor.config["global"]:
                    new_descriptor._get_internal("self")[key] = old_descriptor.config["global"][key]

            # add type specific configs (overwrites global config)
            if "types" in old_descriptor.config:
                if type(self).__name__ in old_descriptor.config["types"]:
                    new_descriptor._get_internal("self").update(
                        deepcopy(old_descriptor.config["types"][type(self).__name__])
                    )

            # add key specific configs (overwrites global and type config)
            if "keys" in old_descriptor.config:
                if self.dask_key_name in old_descriptor.config["keys"]:
                    new_descriptor._get_internal("self").update(
                        deepcopy(old_descriptor.config["keys"][self.dask_key_name])
                    )

                    # TODO: It should be safe to remove these keys from the new_descriptor!?
                    del new_descriptor.config["keys"][self.dask_key_name]

                # TODO: should we prefer the following way of removing the config?
                # new_descriptor['config']['keys'] = {k:v for k,v in old_descriptor["config"]["keys"].items() if k != self.dask_key_name}  # noqa: E501

            new_descriptor.config = old_descriptor.config


def init_flow_graph(flow: Callable) -> Graph:
    """initialises dask task graph from flow
    parts of the callstack of flow not decorated with @task or of type Node will be treated like normal call
    parts that are decorated with @task or of type node will be turned into dask.Delayed objects

    Args:
        flow (Callable): Callable computing desired result to be wrapped in task graph

    Returns:
        Graph: dask task graph for flow
    """
    if inspect.isclass(flow):
        flow = flow()

    with TaskGraphCreator():
        flow_graph = flow()

    return flow_graph


def task(name: str | None = None, context: FlowContext | None = None) -> Callable:
    """decorator for functions and classes to signalise that it should be made into a dask.Delayed object

    Args:
        name (str | None): name of resulting task graph node. defaults to None
        context (FlowContext | None): context for that flow. defaults to None
    """
    context = context or FlowContext

    def decorator_task(func_or_cls: Callable) -> Callable:
        return (
            _wrap_class(func_or_cls, name)
            if inspect.isclass(func_or_cls)
            else _wrap_function(func_or_cls, name, context)
        )

    return decorator_task


def _wrap_class(cls: type, name: str | None = None) -> type:
    """helper for decorator `task` to wrap class"""
    if cls.__name__ == "DeklareClass":  # Keep this check if "DeklareClass" is still a sentinel
        # don't wrap it twice!
        return cls

    # Create a new class dynamically with the original class's name
    # The new class inherits from the original cls and Node
    new_cls_name = cls.__name__
    bases = (cls, Node)
    new_cls_dict = {}

    # Define __init__ for the new class
    def new_init(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN001, ANN401
        cls.__init__(self, *args, **kwargs)

        self._name = getattr(self, "_name", None) or name
        self.config = getattr(self, "config", None) or Descriptor()

    new_cls_dict["__init__"] = new_init

    # Dynamically create the new class
    new_cls = type(new_cls_name, bases, new_cls_dict)

    # Check if the original class defines configure
    if "configure" in cls.__dict__:
        original_inherit_method = cls.__dict__["configure"]

        def new_configure(self: type, descriptor: Descriptor) -> Descriptor:
            # Ensure Node.configure is called correctly
            descriptor = Node.configure(self, descriptor)
            return original_inherit_method(self, descriptor)

        new_cls.configure = new_configure

    # Rename cls's __call__ method to compute
    if "__call__" in cls.__dict__:
        # Directly set 'compute' to the original __call__ method
        new_cls.compute = cls.__dict__["__call__"]
        # Use Node's __call__ method as NewWrappedClass's __call__ method
        new_cls.__call__ = Node.__call__

    return new_cls


def _wrap_function(func: Callable, name: str | None, context: FlowContext) -> Callable:
    """helper for decorator `task` to wrap function"""
    if isinstance(func, Delayed):
        # don't wrap it twice!
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if context.is_enabled():
            if name is None:
                key_name = func.__name__
            else:
                key_name = name

            ext = ""
            while context.exists(key_name + ext):
                ext = uuid4().hex[-6:]

            key_name = key_name + ext
            context.set(key_name, func)

            if ext != "":
                warnings.warn(f"Duplicate name detected. Name changed to {key_name}", stacklevel=1)

            # make a graph node
            return dask.delayed(func)(*args, dask_key_name=key_name, **kwargs)
        else:
            # compute function and return result
            return func(*args, **kwargs)

    return wrapper
