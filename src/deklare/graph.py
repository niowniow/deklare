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

import traceback
from copy import copy, deepcopy
from sqlite3 import InternalError  # ToDo: Why on earth are we importing  a sqllite error?
from typing import Any, Callable, Iterable

import dask
from dask.base import _extract_graph_and_keys, tokenize
from dask.core import flatten, get_dependencies, get_deps
from dask.delayed import Delayed
from dask.optimization import cull, fuse, inline
from dask.typing import Graph
from dask.utils import apply

from .core import KEY_SEP, Node
from .descriptor import Descriptor
from .utils import NodeFailedError, base_name

FUNCTION = 1
DATA = 2
DESCRIPTOR = 3


class App:
    __conf = {
        "fail_mode": "fail",
        "use_delayed": False,
    }
    __setters = ["fail_mode", "use_delayed"]

    @staticmethod
    def config(name: str) -> dict:
        return App.__conf[name]

    @staticmethod
    def exists(name: str) -> bool:
        return name in App.__conf

    @staticmethod
    def set(name: str, value: dict) -> None:
        if name in App.__setters:
            App.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")


def setting_exists(key: str) -> bool:
    return App.exists(key)


def get_setting(key: str) -> dict:
    return App.config(key)


def set_setting(key: str, value: dict) -> None:
    App.set(key, value)


class FailSafeWrapper:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        try:
            return self.func(*args, **kwargs)
        except Exception:
            trace = traceback.format_exc(2)
            return NodeFailedError(trace)


def update_key_in_config(descriptor: Descriptor, old_key: str, new_key: str) -> None:
    if "keys" in descriptor.config:
        if old_key in descriptor.config["keys"]:
            descriptor.config["keys"][new_key] = descriptor.config["keys"].pop(old_key)


def generate_clone_key(current_node_name: str, to_clone_key: str, clone_id: str) -> str:
    return base_name(to_clone_key) + KEY_SEP + tokenize([current_node_name, to_clone_key, "deklare_clone", clone_id])


# @profile
# ToDo: break into multiple smaller functions
def configuration(  # noqa: C901
    delayed: Delayed | list[Delayed],
    descriptors: Descriptor | list[Descriptor],
    keys: Iterable | None = None,
    _default_merge: Callable | None = None,
    optimize_graph: bool = True,
    dependants: dict | None = None,
    clone_instead_merge: bool = True,
) -> Delayed | list[Delayed]:
    """Configures each node of the graph by propagating the descriptor from outputs
    to inputs. Each node checks if it can fulfill the descriptor and what it needs to fulfill
    the descriptor. If a node requires additional configurations to fulfill the descriptor it
    can set the 'requires_descriptor' flag in the returned descriptor and this function will
    add the return descriptor as a a new input to the node's __call__().
    See also Node.configure()

    Args:
        delayed (dask.delayed or list): Delayed object or list of delayed objects
        descriptor (Descriptor or list): descriptor (Descriptor), list of descriptors
        keys (_type_, optional): _description_. Defaults to None.
        default_merge (_type_, optional): _description_. Defaults to None.
        optimize_graph (bool, optional): _description_. Defaults to True.
        dependants (_type_, optional): _description_. Defaults to None.

    Raises:
        RuntimeError: If graph cannot be configured

    Returns:
        dask.delayed: The configured graph
    """

    collections = delayed if delayed is isinstance(delayed, list) else [delayed]

    dsk, dsk_keys = _extract_graph_and_keys(collections)  # ToDo: dont use internal functions
    dependants = dependants or get_deps(dsk)[1]

    dsk_dict = {k: dsk[k] for k in dsk.keys()}

    keys = keys or dsk_keys
    if not isinstance(keys, (list, set)):
        keys = [keys]

    # ToDo: set does not preserve order? then does making the descriptor dict make sense?
    work = list(set(flatten(keys)))
    # create a deepcopy, otherwise we might overwrite descriptors and falsify its usage outside of this function
    # descriptor = deepcopy(descriptor)
    clone_input = False
    if isinstance(descriptors, list) and len(work) == 1:
        descriptors = {k: descriptors for k in work}
        clone_input = True
    elif isinstance(descriptors, list) and len(work) != 1:
        # descriptor = [NestedFrozenDict(r) for r in descriptor if r]
        descriptors = [r for r in descriptors if r]
        if len(descriptors) != len(work):
            raise RuntimeError(
                "When passing multiple descriptor items and the flow has multiple outputs"
                "The number of descriptor items must be same "
                "as the number of keys/outputs"
            )

        descriptors = [d for d in descriptors if d]

        # For each output node different descriptor has been provided
        descriptors = {work[i]: [descriptors[i]] for i in range(len(descriptors))}
    else:
        # Every output node receives the same descriptor
        descriptors = {k: [descriptors] for k in work}

    input_descriptors = {}
    # create a new graph with the configured nodes of the old graph
    out_keys = []  # keep track of configured keys
    work = {k: True for k in work}  # dict for performance and sets dont work?

    if clone_input:
        k = next(iter(work))
        clone_dependencies = descriptors
        current_deps = get_dependencies(dsk_dict, k, as_list=True)
        work = {}
        keys = []
        for clone_id, descriptor in enumerate(clone_dependencies):
            k_in_keys = []
            clone_k = generate_clone_key("fakeit", k, clone_id)
            work[clone_k] = True
            cloned_cd_node = copy(dsk_dict[k])
            dsk_dict[clone_k] = cloned_cd_node
            _normalize_node(clone_k, dsk_dict)

            to_clone_keys = dsk_dict[clone_k][DATA]
            if not isinstance(to_clone_keys, list):
                to_clone_keys = [to_clone_keys]

            for to_clone_key in to_clone_keys:
                if to_clone_key is None:
                    k_in_keys.append(None)
                else:
                    k_in_keys.append(generate_clone_key(clone_k, to_clone_key, clone_id))

            for _, d in enumerate(current_deps):
                clone_work = [d]

                d = generate_clone_key(clone_k, d, clone_id)
                while clone_work:
                    new_clone_work = []
                    for cd in clone_work:
                        clone_d = generate_clone_key(clone_k, cd, clone_id)

                        # update_key_in_config(descriptor,cd,clone_d)
                        # TODO: do we need to reset the dask_key_name of each
                        #       of each cloned node?

                        _normalize_node(cd, dsk_dict)

                        cloned_cd_node = copy(dsk_dict[cd])

                        # if contains data as input
                        to_clone_keys = cloned_cd_node[DATA]
                        if not isinstance(to_clone_keys, list):
                            to_clone_keys = [to_clone_keys]
                        cd_in_keys = []
                        for to_clone_key in to_clone_keys:
                            if to_clone_key is None:
                                cd_in_keys.append(None)
                            else:
                                cd_in_keys.append(generate_clone_key(clone_k, to_clone_key, clone_id))
                        # if len(cd_in_keys) == 1:
                        #     cd_in_keys = cd_in_keys[0]
                        nd = list(cloned_cd_node)
                        nd[DATA] = cd_in_keys
                        cloned_cd_node = tuple(nd)
                        dsk_dict[clone_d] = cloned_cd_node
                        new_deps = get_dependencies(dsk_dict, cd, as_list=True)
                        new_clone_work += new_deps
                    clone_work = new_clone_work
            dsk_k = list(dsk_dict[clone_k])
            dsk_k[DATA] = k_in_keys
            dsk_k[DESCRIPTOR] = (descriptor,)
            dsk_dict[clone_k] = tuple(dsk_k)

            descriptors[clone_k] = (descriptor,)
            keys += [clone_k]

    remove = {k: False for k in work}

    while work:
        new_work = {}

        out_keys += work
        for key in work:
            if key not in descriptors:
                raise InternalError(f"Failed to find descriptor for node {key}")

            # check if we have collected all dependencies so far
            # we will come back to this node another time
            # TODO: make a better check for the case when dependants[k] is a set. why is it a set in the first place..?
            if key in dependants and not isinstance(dependants[key], set):
                continue

            # set configuration for k
            argument_is_node = False
            if isinstance(dsk_dict[key], tuple):
                _normalize_node(key, dsk_dict)

                # now every node will have the function in the second item of the tuple
                if hasattr(dsk_dict[key][FUNCTION], "__self__"):
                    if isinstance(dsk_dict[key][FUNCTION].__self__, Node):
                        argument_is_node = True

            # Check if we get a node of type Node class
            if argument_is_node:
                # have a node class so we can use it's configure function
                assert len(descriptors[key]) == 1
                new_descriptor = dsk_dict[key][1].__self__.configure(
                    descriptors[key][0]
                )  # configure the descriptor for the class
            else:
                # no Node class => no custom configuration function => pass through
                new_descriptor = Descriptor()
                assert len(descriptors[key]) == 1
                r = descriptors[key][0]

                # sanitize descriptor
                if r is not None:
                    ignored_keys = [
                        "requires_descriptor",
                        "insert_predecessor",
                        "clone_dependencies",
                        "remove_dependency",
                        "remove_dependencies",
                    ]
                    new_descriptor = r.__class__.from_dict(
                        {k: v for k, v in r.to_dict().items() if k not in ignored_keys}
                    )
                    new_descriptor._deklare_attrs = {
                        k: v for k, v in new_descriptor._deklare_attrs if k not in ignored_keys
                    }

            # update dependencies
            # we're going to get all dependencies of this node and check if it requires to clone it's input path
            # if so, each cloned path gets a different descriptor from this node (contained in `clone_dependencies`)
            # we are going to introduce new keys and new nodes in the graph
            # so must update this nodes input keys (hacking it from/to dsk_dict[k][DATA]) for each clone

            # for now it's not possible to have predecessors and multiple descriptors
            # User must use one descriptor with `clone_dependencies` and `insert_predecessor` keys
            insert_predecessor = []
            if isinstance(new_descriptor, Descriptor):
                if "insert_predecessor" in new_descriptor.config:
                    insert_predecessor = new_descriptor.config["insert_predecessor"]

                if insert_predecessor:
                    del new_descriptor.config["insert_predecessor"]

            current_deps = get_dependencies(dsk_dict, key, as_list=True)

            k_in_keys = None
            if len(dsk_dict[key]) > DATA:
                k_in_keys = deepcopy(dsk_dict[key][DATA])  # [DATA] equals in_keys in dict

            clone_dependencies = new_descriptor if isinstance(new_descriptor, list) else (new_descriptor,)

            # check if any of our current dependencies already has to fulfil a descriptor
            # since the descriptor's might collide we should just duplicate it
            # in this run it gets a new name, and the existing one is left untouched until it's its turn.
            clone = False
            if clone_instead_merge:
                if len(clone_dependencies) > 1:
                    clone = True
                    k_in_keys = []
                else:
                    for dep in current_deps:
                        if descriptors.get(dep, []):
                            clone = True
                            k_in_keys = []

            # if it's a list it automatically clones it, else the user could use the clone_dependencies to clone
            if isinstance(new_descriptor, Descriptor):
                if new_descriptor.get_config("clone_dependencies", False):
                    clone = True
                    k_in_keys = []
                    clone_dependencies = new_descriptor.config["clone_dependencies"]
                    del new_descriptor.config["clone_dependencies"]

                elif new_descriptor.get_config("requires_descriptor", False):
                    del new_descriptor.config["requires_descriptor"]
                    input_descriptors[key] = new_descriptor

            clone_dependencies = tuple([dep for dep in clone_dependencies if dep])

            for clone_id, descriptor in enumerate(clone_dependencies):
                if clone:
                    to_clone_keys = dsk_dict[key][DATA]
                    if not isinstance(to_clone_keys, list):
                        to_clone_keys = [to_clone_keys]

                    # create new node in graph containing k_in_keys as input
                    if insert_predecessor:
                        pre_function = insert_predecessor[clone_id]
                        pre_descriptor = clone_dependencies[clone_id]

                        pre_k = tokenize([key, "deklare_pre", clone_id])
                        if hasattr(pre_function, "__self__") and hasattr(pre_function.__self__, "dask_key_name"):
                            pre_k = pre_function.__self__.dask_key_name + KEY_SEP + pre_k
                        descriptors[pre_k] = (pre_descriptor,)
                        dsk_dict[pre_k] = [apply, pre_function, [], {}]
                        pre_in_keys = []

                        for to_clone_key in to_clone_keys:
                            if to_clone_key is None:
                                pre_in_keys.append(None)
                            else:
                                pre_in_keys.append(generate_clone_key(key, to_clone_key, clone_id))

                        dsk_dict[pre_k][DATA] = pre_in_keys
                        dsk_dict[pre_k] = tuple(dsk_dict[pre_k])

                        k_in_keys += [pre_k]
                        remove[pre_k] = False
                        new_work[pre_k] = True
                    else:
                        for to_clone_key in to_clone_keys:
                            if to_clone_key is None:
                                k_in_keys.append(None)
                            else:
                                k_in_keys.append(generate_clone_key(key, to_clone_key, clone_id))

                for dep in current_deps:
                    if clone:
                        clone_work = [dep]

                        dep = generate_clone_key(key, dep, clone_id)
                        while clone_work:
                            new_clone_work = []
                            for cd in clone_work:
                                clone_key = generate_clone_key(key, cd, clone_id)

                                _normalize_node(cd, dsk_dict)

                                cloned_node = list(copy(dsk_dict[cd]))

                                # if contains data as input
                                to_clone_keys = cloned_node[DATA]
                                if not isinstance(to_clone_keys, list):
                                    to_clone_keys = [to_clone_keys]

                                cloned_node[DATA] = [
                                    None if k is None else generate_clone_key(key, k, clone_id) for k in to_clone_keys
                                ]

                                dsk_dict[clone_key] = tuple(cloned_node)
                                new_deps = get_dependencies(dsk_dict, cd, as_list=True)
                                new_clone_work += new_deps

                            clone_work = new_clone_work

                    # determine what needs to be removed
                    if not insert_predecessor:
                        # we are not going to remove anything if we inserted a predecessor node before current node k
                        # we are also not updating the descriptors of dependencies of the original node k
                        # since it will be done in the next interaction by configuring the inserted predecessor

                        to_be_removed = False
                        if key in remove:
                            to_be_removed = remove[key]

                        if descriptor is None:
                            to_be_removed = True
                        elif "remove_dependencies" in descriptor._deklare_attrs:
                            to_be_removed = descriptor._deklare_attrs["remove_dependencies"]
                            del descriptor._deklare_attrs["remove_dependencies"]

                        # TODO: so far this doesn't allow to clone dependencies and delete only one of them.
                        #       it might be irrelevant.
                        if descriptor._deklare_attrs.get("remove_dependency", {}).get(base_name(dep), False):
                            to_be_removed = True
                            del descriptor._deklare_attrs["remove_dependency"][base_name(dep)]

                        if not descriptor._deklare_attrs.get("remove_dependency", True):
                            # clean up if an empty dict still exists
                            del descriptor._deklare_attrs["remove_dependency"]
                        if dep in descriptors:
                            if clone_instead_merge:
                                raise InternalError(
                                    f"A duplicate descriptor was found for {dep} with the descriptor \
{descriptor[dep]}, set clone_instead_merge=False to allow this"
                                )
                            if not to_be_removed:
                                descriptors[dep] += [descriptor]
                            remove[dep] = remove[dep] and to_be_removed
                        else:
                            if not to_be_removed:
                                descriptors[dep] = [descriptor]
                            # if we received None
                            remove[dep] = to_be_removed

                        # only configure each node once in a round!
                        # if d not in new_work and d not in work:
                        #     new_work.append(
                        #         d
                        #     )  # TODO: Do we need to configure dependency if we'll remove it?
                        # we should also add `d`` only to the work list if we did not insert a
                        # a predecessor. The predecessor will take care of adding it in the next round
                        # otherwise it could happen that the predecessor changes the name of the node
                        # by cloning it. Then we'd have a deprecated node name in the work list
                        if dep not in work and (dep not in remove or not remove[dep]):
                            new_work[dep] = True

            dsk_k = list(dsk_dict[key])

            # loop though all input keys of this node `k` and discard all inputs that have been removed
            if k_in_keys is not None:
                k_in_keys = k_in_keys if isinstance(k_in_keys, list) else [k_in_keys]
                k_in_keys = [key for key in k_in_keys if (not remove.get(key, False))]

                dsk_k[DATA] = k_in_keys

            dsk_dict[key] = tuple(dsk_k)

        work = new_work

    # Assembling the configured new graph
    out = {k: dsk_dict[k] for k in out_keys if not remove[k]}

    # After we have acquired all descriptors we can input the required_descriptors as a input node to the requiring node
    # we assume that the last argument is the descriptor
    for key in input_descriptors:
        if key not in out:
            continue
        # input_descriptors[k] = clean_descriptor(input_descriptors[k])
        # Here we assume that we always receive the same tuple of (bound method, data, descriptor)
        # If the interface changes this will break #TODO: check for all cases
        if isinstance(out[key][DESCRIPTOR], tuple):
            # FIXME: find a better inversion of unpack_collections().
            #        this is very fragile
            # Check if we've already got a descriptor as argument
            # This is the case if our node will make use of a general config
            # Then the present descriptor is updated with the configured one
            # We need to recreate the tuple/list elements though. (dask changed)
            # TODO: use a distinct descriptor class
            if out[key][DESCRIPTOR][0] is dict:
                my_dict = {}
                # FIXME: it does not account for nested structures
                for item in out[key][DESCRIPTOR][1]:
                    if isinstance(item[1], tuple):
                        if item[1][0] is tuple:
                            item[1] = tuple(item[1][1])
                        elif item[1][0] is list:
                            item[1] = list(item[1][1])
                    my_dict[item[0]] = item[1]
                my_dict = {item[0]: item[1] for item in out[key][DESCRIPTOR][1]}
                my_dict.update(input_descriptors[key])
                out[key] = out[key][:DESCRIPTOR] + ({"descriptor": descriptor.__class__(**my_dict)},)
            else:
                # replace the last entry
                out[key] = out[key][:DESCRIPTOR] + ({"descriptor": input_descriptors[key]},)

        # # TODO: verify that we can ignore this case
        # elif isinstance(out[k][DESCRIPTOR], dict):
        #     out[k] = out[k][:DESCRIPTOR] + (copy(out[k][DESCRIPTOR]) | copy(input_descriptors[k]),)
        else:
            # replace the last entry
            out[key] = out[key][:DESCRIPTOR] + ({"descriptor": input_descriptors[key]},)

        # TODO: we might dask.delayed(out[k][DESCRIPTOR]) here

    # convert to delayed object
    in_keys = list(flatten(keys))

    if len(in_keys) > 1:
        collection = Delayed(key=in_keys, dsk=out)
    else:
        collection = Delayed(key=in_keys[0], dsk=out)
        if isinstance(delayed, list):
            collection = [collection]

    if optimize_graph:
        collection = optimize(collection, keys)

    return collection


def optimize(
    delayed: Delayed | list[Delayed], keys: Iterable | None = None, dask_optimize: bool = False
) -> Delayed | list[Delayed]:
    """Optimizes the graph after configuration"""

    collections = delayed if isinstance(delayed, list) else [delayed]

    dsk, dsk_keys = _extract_graph_and_keys(collections)
    dsk_dict = {k: dsk[k] for k in dsk.keys()}

    keys = keys or dsk_keys
    keys = keys if isinstance(keys, (list, set)) else [keys]

    keys = list(set(flatten(keys)))

    # invert the task graph: make a compute graph
    dsk_inv, sources = _invert_task_graph(keys, dsk_dict)

    # traverse the task graph in compute direction
    traversed_graph = _traverse_graph(sources, keys, dsk_dict, dsk_inv, optimize=dask_optimize)

    if len(keys) > 1:
        collection = Delayed(key=keys, dsk=traversed_graph)
    else:
        collection = Delayed(key=keys[0], dsk=traversed_graph)
        if isinstance(delayed, list):
            collection = [collection]

    return collection


def compute(graph: Graph, descriptor: Descriptor, plan: bool = False) -> Any:  # noqa: ANN401
    configured_graph = configuration(graph, descriptor)
    if plan:
        return configured_graph
    computed_result = dask.compute(configured_graph)[0]
    return computed_result


def _normalize_node(key: str, dsk_dict: dict) -> None:
    # any node that does not have the following structure will get it
    # (apply, func, args, kwargs)
    if dsk_dict[key][0] is not apply:
        args = []
        kwargs = {}

        if len(dsk_dict[key]) > 2:
            args = list(dsk_dict[key][1:-1])
            kwargs["descriptor"] = dsk_dict[key][-1]
        else:
            args = list(dsk_dict[key][1:])

        dsk_dict[key] = (apply, dsk_dict[key][0], args, kwargs)


def _invert_task_graph(keys: list, dsk_dict: dict) -> tuple[dict, set]:
    dsk_inv = {k: {} for k in keys}

    in_keys = copy(keys)
    out_keys = []
    sources = set()
    seen = set()
    while in_keys:
        new_work = []

        out_keys += in_keys
        for k in in_keys:
            current_deps = get_dependencies(dsk_dict, k, as_list=True)
            if not current_deps:
                sources.add(k)

            for dep in current_deps:
                if dep not in dsk_inv:
                    dsk_inv[dep] = {k}
                else:
                    dsk_inv[dep].add(k)

                if dep not in seen:
                    new_work.append(dep)
                    seen.add(dep)

        in_keys = new_work

    return dsk_inv, sources


def _traverse_graph(sources: list, keys: list, dsk_dict: dict, dsk_inv: dict, *, optimize: bool) -> dict:
    work = copy(sources)
    out_keys = []
    seen = set()
    rename = {}
    while work:
        new_work = []

        for k in work:
            if k in keys:
                # if we are at a sink node we don't change the name
                out_keys += [k]
                continue

            # rename k
            node_token = ""

            # FIXME: for which cases do we need the following two lines?
            # it breaks the optimization significantly for chunkpersister, because it initializes
            # a new hashpersister instance for each parallel branch dynamically
            # What is the purpose? Do we want to account for internal configurations? Maybe find a different solution
            # if hasattr(dsk_dict[k][0], "__self__"):
            #     node_token = dsk_dict[k][0].__self__

            input_token = list(dsk_dict[k][1:])

            new_k = base_name(k) + KEY_SEP + tokenize([node_token, input_token])

            out_keys += [new_k]
            rename[k] = new_k

            for dep in dsk_inv[k]:
                # TODO: is there a way to not change the dsk_dict in-place?
                if isinstance(dsk_dict[dep][DATA], list):
                    input_data = [_replace(s, k, new_k) for s in dsk_dict[dep][DATA]]
                elif isinstance(dsk_dict[dep][DATA], str):
                    input_data = _replace(dsk_dict[dep][DATA], k, new_k)
                else:
                    input_data = dsk_dict[dep][DATA]

                dsk_dict[dep] = tuple(list(dsk_dict[dep][:DATA]) + [input_data] + list(dsk_dict[dep][DATA + 1 :]))

                if dep not in seen:
                    new_work.append(dep)
                    seen.add(dep)

        work = new_work

    for k in rename:
        dsk_dict[rename[k]] = dsk_dict[k]

    out = {k: dsk_dict[k] for k in out_keys}

    if optimize:
        out = _optimize_functions(out, keys)

    return out


def _replace(s: Any, r: str, n: str) -> str:  # noqa: ANN401
    if isinstance(s, str) and s == r:
        return n

    return s


def _optimize_functions(dsk: Graph, keys: Iterable) -> Graph:
    dsk1, deps = cull(dsk, keys)
    dsk2 = inline(dsk1, dependencies=deps)
    dsk3, deps = fuse(dsk2, fuse_subgraphs=True)
    return dsk3
