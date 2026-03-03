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

import json
from copy import copy, deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Type

import obstore as obs
import pystac
from cachetools import Cache, LRUCache

# TODO: can we implement our own hash function for descriptors to reduce dependency on dask?
from dask.base import tokenize
from obstore.store import LocalStore, ObjectStore, Store

from .core import task
from .data_io import DataContainer, MediaDescription, StacIO
from .descriptor import Descriptor, accept_dict_descriptor
from .utils import (
    NodeFailedError,
    descriptor_update,
    get_segments,
)


def file_in_store(store: Store, data_path: str | Path) -> bool:
    try:
        obs.head(store, data_path)
        return True
    except FileNotFoundError:
        try:
            next(iter(obs.list(store, prefix=data_path)))
            return True
        except StopIteration:
            # The list was empty, so the path truly does not exist
            return False


def make_sub_store(parent_store: Store, sub_path: str) -> Store:
    """
    Creates a new LocalStore instance rooted inside the parent_store's prefix.
    """
    # 1. Get the current prefix (defaulting to empty string if None)
    # obstore stores prefix as a Path object or None
    current_prefix = getattr(parent_store, "prefix", None) or ""

    # 2. Join the old prefix with the new sub-path
    # We use pathlib to handle slash consistency automatically
    new_prefix = Path(current_prefix) / sub_path
    # 3. Return a NEW instance pointing to this deeper folder
    return LocalStore(prefix=new_prefix, mkdir=True)


@task()
class Persister:
    def __init__(
        self,
        data_container: Type[DataContainer],
        store: ObjectStore | None = None,
        stac_io: StacIO | None = None,
        selected_keys: Iterable | None = None,
        force_update: bool = False,
        use_memorycache: bool = True,
        cache: Cache | None = None,
        global_lock: Lock = None,
        save_metadata: bool = False,
        hash_filter: callable | None = None,
    ) -> None:
        super().__init__(Descriptor(config={"force_update": force_update, "use_memorycache": use_memorycache}))
        if isinstance(store, str) or isinstance(store, Path):
            store = LocalStore(prefix=store, mkdir=True)
        self.store = store
        self.data_container = data_container

        if cache is None:
            cache = LRUCache(100)
        self.cache = cache

        if selected_keys is None:
            # use all keys as hash
            pass

        self.stac_io = stac_io
        self._global_lock = global_lock
        self._mutex = Lock()
        self.save_metadata = save_metadata

        if hash_filter is not None and not callable(hash_filter):
            raise RuntimeError("hash_filter has to be a callable")

        self.hash_filter = hash_filter

    def configure(self, descriptor: Descriptor) -> Descriptor:
        descriptor_hash = self.get_hash(descriptor)
        data_path = f"data/{descriptor_hash}"

        # compute action defaults to passthrough
        descriptor.config["action"] = "passthrough"

        if descriptor.get_config("bypass", False):
            # set to passthrough -> nothing will happen
            return descriptor

        # propagate the descriptor_hash to the compute function
        descriptor.config["descriptor_hash"] = descriptor_hash

        # reload and rewrite the chunk if descriptored
        if descriptor.get_config("force_update", False):
            descriptor.config["action"] = "store"
            return descriptor

        with self._mutex:
            if descriptor.get_config("use_memorycache", True) and data_path in self.cache:
                descriptor._deklare_attrs["remove_dependencies"] = True
                # set the compute action to load
                descriptor.config["action"] = "load_from_cache"
                return descriptor

            if self.store is None:
                return descriptor

            # while holding the mutex, we need to check if the file exists
            if file_in_store(self.store, data_path):
                # remove previous node since we are going to load from disk
                descriptor._deklare_attrs["remove_dependencies"] = True

                # set the compute action to load
                descriptor.config["action"] = "load"
                return descriptor
            elif file_in_store(self.store, "fail/" + descriptor_hash):
                # remove previous node since we are going to load the fail info from disk
                descriptor._deklare_attrs["remove_dependencies"] = True
                descriptor.config["descriptor_hash"] = "fail/" + descriptor_hash

                # set the compute action to load
                descriptor.config["action"] = "load"
                return descriptor

            # TODO: check if the file will be written to already?

            descriptor.config["action"] = "store"

        return descriptor

    @accept_dict_descriptor(arg_name="descriptor")
    def compute(self, data: DataContainer | None = None, descriptor: Descriptor | None = None) -> DataContainer:  # noqa: C901
        descriptor = descriptor or Descriptor()

        if descriptor.config["action"] == "passthrough":
            return data

        if self.store is not None:
            # self.store.dirfs.mkdirs("data/", exist_ok=True)
            data_path = f"data/{descriptor.config['descriptor_hash']}"

        # if descriptor.config["action"] == "load_from_cache":
        # with self._mutex:
        #     if data_path in self.cache:
        #         cached = self.cache[data_path]
        #         return cached
        if descriptor.config["action"] == "load" or descriptor.config["action"] == "load_from_cache":
            substore = make_sub_store(self.store, data_path)
            data = self.data_container.read(substore)

            # with self._mutex:
            #     self.cache[data_path] = data

            return data
        elif descriptor.config["action"] == "store":
            with self._mutex:
                self.cache[data_path] = data

            if self.store is None:
                return data

            try:
                # in this case we assume that the second element is additional metadata for the STAC item
                item_metadata = data.get_stac_metadata() or {}

                # write to file
                if isinstance(data, NodeFailedError):
                    # self.store.dirfs.mkdirs("fail/", exist_ok=True)
                    # with self.store.dirfs.open("fail/" + descriptor.config("descriptor_hash"), "wb") as f:
                    #     data.write(f)
                    failed_path = "fail/" + descriptor["descriptor_hash"]
                    substore = make_sub_store(self.store, failed_path)
                    data.write(substore)
                else:
                    if isinstance(data, str):
                        raise RuntimeError(f"something wrong {data}")

                    try:
                        substore = make_sub_store(self.store, data_path)
                        data.write(substore)
                    except Exception as e:
                        # self.store.dirfs.rm(data_path)
                        raise e

                    if self.save_metadata:
                        self._save_metadata(descriptor, item_metadata, data.file_info())

            except Exception as e:
                import traceback

                traceback.print_exception(e.__class__, e, e.__traceback__)
                raise NodeFailedError("Error during Persister") from e

            return data
        else:
            raise NodeFailedError("A bug in Persister. Please report.")

    def is_valid(self, descriptor: Descriptor) -> bool | None:
        """Checks if persisted object for `descriptor`
        exists and is valid (i.e. is not of type NodeFailedException).

        Args:
            descriptor (Descriptor): The descriptor that should be checked

        Returns:
            boolean | None: Returns false if the persisted item is of type NodeFailedException
                             Returns None if the descriptor has not been persisted yet.
        """
        descriptor_hash = self.get_hash(descriptor)

        if "fail/" + descriptor_hash in self.store:
            return False

        if descriptor_hash in self.store:
            return True

        return None

    def get_hash(self, descriptor: Descriptor) -> str:
        """returns the hash of the descriptor

        Args:
            descriptor (Descriptor): descriptor

        Returns:
            str: hash of the request
        """
        r = {k: v for k, v in descriptor.to_dict().items() if k != "self" and k != "config"}
        if self.hash_filter:
            r = self.hash_filter(r)
        s = json.dumps(r, sort_keys=True, skipkeys=True, default=_string_timestamp)
        descriptor_hash = tokenize(s)

        return descriptor_hash

    def _save_metadata(self, descriptor: Descriptor, item_metadata: dict, file_info: MediaDescription) -> None:
        """saves metadata for given chunk using STAC (https://stacspec.org/)

        Args:
            descriptor (Descriptor): the descriptor containing the temporal and spacial boundaries
            item_metadata (dict): additional metadata passed by loader to save in STAC item
            file_info (MediaDescription): information about the saved file
        """

        kwargs = self.stac_io.gen_stac_item_kwargs(descriptor, item_metadata)

        item = pystac.Item(**kwargs)

        asset = pystac.Asset(
            href=f"./../../data/{descriptor.config['descriptor_hash']}",
            description=file_info.description,
            media_type=file_info.media_type,
            roles=["data"],
        )
        item.add_asset(key="data", asset=asset)

        # save in collection and avoid concurrency issues
        with self._global_lock:
            collection = pystac.Collection.from_file(
                "/stac/collection.json", self.stac_io
            )  # pretending location is absolute to stop stac from changing path
            collection.add_item(item)
            collection.save(
                catalog_type=pystac.CatalogType.SELF_CONTAINED,
                dest_href="/stac",  # pretend path is absolute so pystac doesnt try and change it
                stac_io=self.stac_io,
            )


def _string_timestamp(o: object) -> str:
    if hasattr(o, "isoformat"):
        return o.isoformat()
    else:
        return str(o)


@task()
class ChunkPersister:
    def __init__(
        self,
        data_container: Type[DataContainer],
        store: ObjectStore | None = None,
        dim: str = "time",
        segment_slice: dict | Callable[..., dict] | None = None,
        segment_stride: dict | Callable[..., dict] | None = None,
        dataset_scope: dict | Callable[..., dict] | None = None,
        mode: str = "overlap",
        reference: dict | None = None,
        force_update: bool = False,
        collection_metadata: dict | None = None,
        save_metadata: bool = False,
        use_memorycache: bool = True,
        cache: Cache | None = None,
        hash_filter: callable | None = None,
        pre_filter: callable | None = lambda x: x,
    ) -> None:
        """Chunks every incoming descriptor into subchunks if descriptor is larger than segment_slice
         or extends the descriptor to the respective chunksize if descriptor is smaller than segment_slice

        Args:
            data_container (Type[DataContainer]): Type of DataContainer used
            store (obstore.ObjectStore): store used for caching
            dim (str, optional): _description_. Defaults to "time".
            segment_slice (dict | Callable[..., dict], optional): dict containing an entry for each chunked dimension
                each entry is the respective chunk size given in the units of the expected dimension of the descriptor.
                e.g. for a time dimension you can use pd.Timedelta. Defaults to None.
            dataset_scope (dict | Callable[...,dict], optional): The extend of the chunking.
                If None, the incoming descriptor will be used as the scope. If only select dimensions are given,
                the scope for the other dimensions will be choosen from the incoming descriptor. Defaults to None.
            mode (str, optional): _description_. Defaults to "overlap".
            reference (dict, optional): _description_. Defaults to None.
            force_update (bool, optional): _description_. Defaults to False.
            collection_metadata (dict, optional): Further kwargs for STAC collection.
                May contain keys ['id', 'title', 'keywords', 'license', 'links', 'providers'].
                For 'links' and 'providers' lists of either corresponding STAC objects or dicts to construct them.
                Defaults to {}.
        """
        self.data_container = data_container
        self.pre_filter = pre_filter

        self.use_memorycache = use_memorycache
        self.cache = cache or LRUCache(10)

        if callable(segment_slice):
            self.segment_slice = segment_slice
            segment_slice = None
        else:
            self.segment_slice = None

        if callable(segment_stride):
            self.segment_stride = segment_stride
            segment_stride = None
        else:
            self.segment_stride = None

        self.save_metadata = save_metadata

        super().__init__(
            Descriptor(
                config={
                    "dim": dim,
                    "segment_slice": segment_slice,
                    "segment_stride": segment_stride,
                    "dataset_scope": dataset_scope,
                    "mode": mode,
                    "reference": reference,
                    "force_update": force_update,
                }
            )
        )

        if store is None:
            raise RuntimeError("Either filesystem or store must be provided")

        if isinstance(store, str) or isinstance(store, Path):
            store = LocalStore(prefix=store, mkdir=True)

        self.store = store

        self.stac_io = StacIO(store=store)

        if self.save_metadata:
            # create collection if doesn't exist
            try:
                collection = pystac.Collection.from_file("/stac/collection.json", self.stac_io)
            except Exception:
                collection_metadata = ChunkPersister._process_collection_metadata(collection_metadata or {})
                collection = pystac.Collection(**collection_metadata[0])

                for link in collection_metadata[1]:
                    collection.add_link(link)

                collection.normalize_and_save(
                    root_href="/stac",  # pretend path is absolute so pystac doesnt try and change it
                    catalog_type=pystac.CatalogType.SELF_CONTAINED,
                    stac_io=self.stac_io,
                )

        self.mutex = Lock()

        if hash_filter is not None and not callable(hash_filter):
            raise RuntimeError("hash_filter has to be a callable")
        self.hash_filter = hash_filter

    def __dask_tokenize__(self) -> tuple:
        return (ChunkPersister,)

    def configure(self, descriptor: Descriptor) -> Descriptor:  # noqa: C901
        rs = descriptor.config
        rs.update(self.config.config)
        if rs.get("bypass", False):
            return descriptor

        def get_value(attr_name: str) -> Any:  # noqa: ANN401
            # decide if we use the attribute provided in the descriptor or
            # from a callback provided at initialization
            value = None
            if rs.get(attr_name, None) is None:
                # there is no attribute in the descriptor, check for callback
                callback = self.config.config.get(attr_name, None)
                if callback is not None and callable(callback):
                    value = callback(descriptor)
                else:
                    # not passing segment_stride is okay
                    if attr_name == "segment_stride":
                        return None
                    raise RuntimeError(f"No valid {attr_name} provided")
            else:
                value = rs[attr_name]
            return value

        dataset_scope = copy(rs)
        if rs.get("dataset_scope", None) is not None:
            dataset_scope.update(rs["dataset_scope"])
        segment_slice = get_value("segment_slice")
        segment_stride = get_value("segment_stride")
        segments = get_segments(
            dataset_scope,
            segment_slice,
            segment_stride,
            reference=rs["reference"],
            mode=rs["mode"],
            timestamps_as_strings=True,
            minimal_number_of_segments=1,
        )
        cloned_descriptors = []
        cloned_persisters = []
        for i, segment in enumerate(segments):
            segment_descriptor = deepcopy(descriptor)
            if "self" in segment_descriptor._deklare_attrs:
                del segment_descriptor._deklare_attrs["self"]
            descriptor_update(segment_descriptor, segment)
            processed_descriptors = self.pre_filter(segment_descriptor)
            if not isinstance(processed_descriptors, list):
                if processed_descriptors:
                    processed_descriptors = [processed_descriptors]
                else:
                    processed_descriptors = []

            for j, segment_descriptor in enumerate(processed_descriptors):
                cloned_descriptors += [segment_descriptor]
                cloned_persister = Persister(
                    data_container=self.data_container,
                    store=self.store,
                    stac_io=self.stac_io,
                    global_lock=self.mutex,
                    save_metadata=self.save_metadata,
                    cache=self.cache,
                    use_memorycache=self.use_memorycache,
                )
                cloned_persister.dask_key_name = f"{self.dask_key_name}_persister_{i}_{j}"
                descriptor_update(
                    segment_descriptor,
                    {
                        "config": {
                            "keys": {cloned_persister.dask_key_name: {"force_update": rs.get("force_update", False)}}
                        }
                    },
                )
                cloned_persisters += [cloned_persister.compute]

        # Insert predecessor
        # new_descriptor = {}
        descriptor.config["clone_dependencies"] = cloned_descriptors
        descriptor.config["insert_predecessor"] = cloned_persisters

        return descriptor

    def compute(self, *data: DataContainer | NodeFailedError, descriptor: Descriptor | None = None) -> DataContainer:  # noqa: ARG002
        def unpack_list(inputlist: Iterable[NodeFailedError | DataContainer] | NodeFailedError | DataContainer) -> list:
            new_list = []
            for item in inputlist:
                if isinstance(item, Iterable):
                    new_list += unpack_list(item)
                else:
                    new_list += [item]
            return new_list

        data = unpack_list(data)
        success = [d for d in data if not isinstance(d, NodeFailedError)]

        if not success:
            failed = [str(d) for d in data if isinstance(d, NodeFailedError)]
            raise RuntimeError(f"Failed to load data. Reason: {failed}")

        if self.save_metadata:
            # update extents
            with self.mutex:
                collection = pystac.Collection.from_file(
                    "/stac/collection.json", self.stac_io
                )  # pretend path is absolute so pystac doesnt try and change it
                collection.update_extent_from_items()
                collection.save(
                    catalog_type=pystac.CatalogType.SELF_CONTAINED,
                    dest_href="/stac",  # pretend path is absolute so pystac doesnt try and change it
                    stac_io=self.stac_io,
                )

        section = self.data_container.merge(*success)
        return section

    @staticmethod
    def _process_collection_metadata(collection_metadata: dict | None = None) -> tuple[dict, list]:
        """return dict with arguments to create pystac.Collection

        Args:
            collection_metadata (dict, optional): Dict containing keyword arguments for pystac.Collection constructor

        Returns:
            (dict, list): dict with list of keyword arguments, list of pystac.Link objects to add to collection
        """

        collection_metadata = collection_metadata or {}

        links = []
        for link in collection_metadata.get("links", []):
            if isinstance(link, dict):
                link = pystac.Link(**link)
            elif not isinstance(link, pystac.Link):
                continue
            links.append(link)

        kwargs = copy(DEFAULT_KWARGS)
        blocked_kwargs = ["extent", "catalog_type", "links"]
        for k in collection_metadata:
            if k not in blocked_kwargs:
                kwargs[k] = collection_metadata[k]

        if isinstance(kwargs["providers"], list):
            providers = []
            for p in kwargs["providers"]:
                if isinstance(p, dict):
                    p = pystac.Provider(**p)
                elif not isinstance(p, pystac.Provider):
                    continue
                providers.append(p)
            kwargs["providers"] = providers

        return (kwargs, links)


DEFAULT_KWARGS = {
    "id": "",
    "description": "",
    "extent": pystac.Extent(
        spatial=pystac.SpatialExtent([None]),
        temporal=pystac.TemporalExtent([[None, None]]),
    ),
    "title": "",
    "catalog_type": pystac.CatalogType.SELF_CONTAINED,
    "license": "",
    "keywords": None,
    "providers": None,
}
