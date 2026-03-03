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

import importlib
import inspect
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import yaml

from .core import init_flow_graph, task
from .descriptor import Descriptor, PermissiveDescriptor, accept_dict_descriptor
from .graph import Node, compute
from .persist import ChunkPersister, Persister


def deklare_flow(
    flow: Callable,
    template_descriptor: type[Descriptor] | None = None,
    config_path: Path | str | None = None,
) -> Callable:
    """creates a deklare flow to use to query data for given descriptors using dask

    Args:
        flow (Callable): flow to be converted into a dask task graph
        template_descriptor (type[Descriptor] | None): template descriptor class to validate fields
            of query descriptors. defaults to None
        config_path (Path | str | None): path to a config file to replace values in query descriptors
            defaults to None

    Returns:
        Callable: Callable with attribute `query` to execute dask task graph for actual function
    """
    flow_graph = init_flow_graph(flow)

    config_descriptor = None
    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_descriptor = yaml.safe_load(f)

    def query_class(self, descriptor: Descriptor, plan=False) -> Any:  # noqa: ANN001, ANN401, ARG001
        """wrapper for query function for classes with self attribute"""
        return query(descriptor, plan=plan)

    @accept_dict_descriptor(arg_name="descriptor", descriptor_cls=template_descriptor)
    def query(descriptor: Descriptor, plan: bool = False) -> Any:  # noqa: ANN401
        """query function to get data corresponding to descriptor"""
        if config_descriptor:
            descriptor = (template_descriptor or PermissiveDescriptor).update_from_config_dict(
                descriptor, config_descriptor
            )

        if template_descriptor:
            descriptor = template_descriptor.from_dict(descriptor.to_dict())

        return compute(flow_graph, descriptor, plan=plan)

    if hasattr(flow, "__self__") and flow.__self__ is not None:
        flow.query = query_class
    else:
        flow.query = query

    return flow


def deklare_module(
    module: str | ModuleType,
    flows: str | set[str] | None = None,
    names: dict[str, str] | None = None,
    external_tasks: list[str] | None = None,
    ignore: list[str] | None = None,
    no_wrap: bool = False,
) -> tuple[ModuleType, *tuple[Callable, ...]]:
    """
    deklare flows in a module and recursively handle dependencies

    Args:
        module (str | ModuleType): Module object or module name to process
        flows (str | set): name(s) of functions/classes to treat as flows. defaults to None
        names (dict[str, str]): dict mapping member names to task names. defaults to None
        external_tasks (list[str]): list of external task names to include. defaults to None
        ignore (list[str]): list of member names to ignore entirely. defaults to None
        no_wrap (bool): if True, flows are not wrapped with `deklare_flow`. defaults to False

    Returns:
        tuple[ModuleType, *tuple[Callable, ...]]:
            - The module object
            - All flow instances (possibly wrapped) in order
    """
    names = names or {}
    external_tasks = external_tasks or []
    ignore = ignore or []

    flows = flows or set()
    if isinstance(flows, str):
        flows = {flows}

    flows = set(flows)

    if isinstance(module, str):
        # Dynamically import the module
        module = importlib.import_module(module)

    flow_instances, dependency_modules = _init_modules(module, flows, ignore, names, external_tasks)

    for dependency_module, dependency_flows in dependency_modules.items():
        dependency_flows_names = list(dependency_flows.keys())
        module_and_flows = deklare_module(dependency_module, flows=dependency_flows_names, no_wrap=True)
        for _, flow in enumerate(module_and_flows[1:]):
            setattr(module, flow.__name__, flow)
            # # TODO: would the above fail if we use `from module import flow as f`?
            # # maybe better to do, but probably not always in the same order!
            # setattr(module, dependency_flows[i], flow)

    for name, func_or_cls in flow_instances.items():
        if no_wrap:
            flow_instances[name] = func_or_cls
        else:
            flow_instances[name] = deklare_flow(func_or_cls)

    return tuple([module] + list(flow_instances.values()))


def _is_module_function_or_class(_module: ModuleType) -> Callable:
    """returns a predicate function to check if a member of a module is a function or a class

    Args:
        _module (ModuleType): module to check members against

    Returns:
        Callable: returns True if the member is a function or a class, else False
    """

    def predicate(member: object) -> bool:
        # TODO: should we also check for member.__name__ in module.__name__?
        return inspect.isfunction(member) or inspect.isclass(member)

    return predicate


def _init_modules(
    module: ModuleType, flows: set[str], ignore: list[str], names: dict[str, str], external_tasks: list[str]
) -> tuple[dict[str, Callable], dict[str, Callable]]:
    """
    initialize module flows and handle dependencies.

    Iterates over all functions and classes in a module and separates:
    - `flow_instances`: functions/classes that match the `flows` set
    - `dependency_modules`: functions/classes from other modules required as dependencies

    Members not in `flows` but not ignored are wrapped with `task()` decorator.

    Args:
        module (importlib.ModuleType): module to inspect
        flows (set[str]): set of member names to treat as flows
        ignore (list[str]): ist of member names to ignore
        names (dict[str, str]): mapping of member names to task names
        external_tasks (list[str]): list of externally defined tasks to include

    Returns:
        tuple[dict[str, Callable], dict[str, Callable]]:
            - flow_instances: mapping flow names to members
            - dependency_modules: mapping module names to members that are dependencies
    """
    flow_instances, dependency_modules = {}, {}

    # Iterate over all functions defined in the module
    for name, func_or_cls in inspect.getmembers(module, predicate=_is_module_function_or_class(module)):
        if name in flows:
            # if member direct member of the module, we add it to the return flows
            # if not, it needs to be loaded appropriately from it's original module
            if func_or_cls.__module__ == module.__name__:
                flow_instances[name] = func_or_cls
            else:
                # select to load it as a deklare dependency, i.e. load all required members
                # as deklare tasks
                dependency_modules[func_or_cls.__module__] = dependency_modules.get(func_or_cls.__module__, {}) | {
                    name: func_or_cls
                }
        elif name not in ignore and (func_or_cls.__module__ == module.__name__ or name in external_tasks):
            # Decorate the function and add it back to the module's namespace
            key_names = names.get(name, None)
            setattr(module, name, task(name=key_names)(func_or_cls))

    return flow_instances, dependency_modules
