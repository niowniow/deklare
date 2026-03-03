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

from abc import ABC, abstractmethod
from typing import IO, Any, Iterable, NamedTuple, Self, TypeVar

import numpy as np
import obstore
import pandas as pd
import pystac
import xarray as xr
from shapely.geometry import Polygon, mapping

from .descriptor import DatetimeRange, Descriptor, Range

Index = TypeVar(
    "Index",
    int,
    slice,
    float,
    str,
    np.datetime64,
    Iterable[int],
    Iterable[float],
    Iterable[str],
    Iterable[np.datetime64],
    dict[str, Any],
)


class MediaDescription(NamedTuple):
    """more precise return type to avoid confusion

    Attributes:
        media_type (str): should be in https://www.iana.org/assignments/media-types/media-types.xhtml, e.g. `image/tiff`
        description (str): information needed to read file in human readable format
    """

    media_type: str
    description: str


class StacIO(pystac.StacIO):
    """subclass of pystac.StacIO to properly handle metadata from descriptors and work with FSMap

    Attributes:
        store (obstore): obstore Store instance used to save the data with object storage API
    """

    store: obstore.store.ObjectStore

    def __init__(self, store: obstore.store.ObjectStore) -> None:
        self.store = store

    def read_text(self, source: pystac.utils.HREF, *args, **kwargs) -> str:  # noqa: ANN002, ANN003, ARG002
        """reads the data at `source` stored in the store

        Args:
            source : source to read from

        Returns:
            str: text contained in file at location specified by the URI
        """
        str_src = str(source)
        # needed to avoid issues with absolute paths that are not properly handled in pystac
        if str_src.startswith("/") and len(str_src) > 1:
            str_src = str_src[1:]

        response = self.store.get(str_src)
        return bytes(response.bytes()).decode()

    def write_text(self, dest: pystac.utils.HREF, txt: str, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG002
        """writes the data of `txt` into the store to `dest`

        Args:
            dest :destination to write to
            txt : the text to write into destination
        """
        str_dest = str(dest)
        if str_dest.startswith("/") and len(str_dest) > 1:
            str_dest = str_dest[1:]

        # self.store[str_dest] = txt.encode()
        self.store.put(str_dest, txt.encode())

    def gen_stac_item_kwargs(self, descriptor: Descriptor, item_metadata: dict) -> dict:
        """generates metadata for the stac item for a descriptor

        Args:
            descriptor (Descriptor): descriptor describing range of data
            item_metadata (dict): additional metadata for item

        Raises:
            RuntimeError: if descriptor does not match expected format to generate metadata

        Returns:
            dict: stac metadata from descriptor and item_metadata
        """
        if (
            "longitude" not in descriptor
            or not isinstance(descriptor["longitude"], Range)
            or "latitude" not in descriptor
            or not isinstance(descriptor["latitude"], Range)
            or "time" not in descriptor
            or not isinstance(descriptor["time"], DatetimeRange)
            or "variable" not in descriptor
            or "descriptor_hash" not in descriptor.config
        ):
            raise RuntimeError("Given descriptor does not match required metadata format")

        bbox = [
            descriptor["longitude"].start,
            descriptor["latitude"].end,
            descriptor["longitude"].end,
            descriptor["latitude"].start,
        ]
        footprint = mapping(
            Polygon(
                [
                    [bbox[0], bbox[1]],  # lower left corner
                    [bbox[0], bbox[3]],  # upper left corner
                    [bbox[2], bbox[3]],  # upper right corner
                    [bbox[2], bbox[1]],  # lower right corner
                    [bbox[0], bbox[1]],  # lower left corner
                ]
            )
        )
        start_time = descriptor["time"].start.to_pydatetime()
        end_time = descriptor["time"].end.to_pydatetime()
        variables = descriptor["variable"]

        kwargs = {
            "id": descriptor.config["descriptor_hash"],
            "geometry": footprint,
            "bbox": bbox,
            "datetime": None,
            "start_datetime": start_time,
            "end_datetime": end_time,
            "properties": {
                "description": self._gen_description(descriptor),
                "variables": variables,
            },
        }

        blocked_keys = kwargs.keys()
        for k, v in item_metadata.items():
            if k not in blocked_keys:
                if k == "assets":
                    kwargs[k] = {k_a: pystac.Asset(**a) for k_a, a in v.items()}
                else:
                    kwargs[k] = v
            elif k == "properties" and isinstance(v, dict):
                for k_p, v_p in v.items():
                    if k_p != "variables":
                        kwargs[k][k_p] = v_p

        return kwargs

    def _gen_description(self, descriptor: Descriptor) -> str:
        """generate human readable description for STAC Item of chunk

        Args:
            descriptor (Descriptor): descriptor defining range of data

        Returns:
            str: STAC Item description
        """
        start_time = descriptor["time"].start.isoformat()
        end_time = descriptor["time"].end.isoformat()
        variable_string = ", ".join(descriptor["variable"])
        latitude_string = f"{descriptor['latitude'].start} to {descriptor['latitude'].end} latitude"
        longitude_string = f"{descriptor['longitude'].start} to {descriptor['longitude'].end} longitude"

        description = (
            f"This chunk contains data for the variable(s) {variable_string}, "
            f"collected from {start_time} to {end_time}, "
            f"covering the geographic region defined by {latitude_string} and {longitude_string}"
        )

        return description


class DataContainer(ABC):
    @abstractmethod
    def write(self, file: IO) -> None:
        """takes the data and writes it to the file

        Args:
            file (IO): file-like object to save data from itself to
        """
        pass

    @classmethod
    @abstractmethod
    def read(cls, file: IO) -> Self:
        """reads the data in file and creates new data container from it

        Args:
            file (IO): file-like object containing saved data

        Returns:
            DataContainer: container containing data saved in file
        """
        pass

    @abstractmethod
    def get_stac_metadata(self) -> dict | None:
        """optionally returns STAC metadata for the contained data

        returns:
            dict | None: dict containing STAC metadata or None for no metadata
        """
        pass

    @classmethod
    @abstractmethod
    def merge(cls, *elements: Self) -> Self:
        """merge the elements into a single DataContainer instance
        if no elements are passed return an empty container, if only one is passed acts as the identity function

        Args:
            elements (DataContainer): list of data containers to merge

        Returns:
            DataContainer: DataContainer resulting from merging elements
        """
        pass

    @abstractmethod
    def file_info(self) -> MediaDescription:
        """return media type and text describtion of file saved in `write` function

        Returns:
            MediaDescription: description of saved file
        """
        pass


class XArrayContainer(DataContainer):
    """example DataContainer to handle XArray Datasets

    Attributes:
        data (xr.Dataset): data wrapped in container
    """

    data: xr.Dataset

    def __init__(self, data: xr.Dataset) -> None:
        super().__init__()
        self.data = data.copy()

    def __getitem__(self, index: Index) -> xr.DataArray | xr.Dataset:
        """function to make XArrayContainer indexable

        Args:
            index (Index): index to retrieve data from container

        Returns:
            xr.DataArray | xr.Dataset: data corresponding to index
        """
        return self.data[index]

    def write(self, file: IO) -> None:
        """takes the data and writes it to the given file in netCDF format

        Args:
            file (IO): file-like object to which netCDF data is saved
        """
        self.data.to_netcdf(file)

    @classmethod
    def read(cls, file: IO) -> Self:
        """reads the data in file and creates new data container from it

        Args:
            file (IO): file-like object containing saved data in netCDF format

        Returns:
            XArrayContainer: container containing data saved in file
        """
        return cls(xr.open_dataset(file))

    def get_stac_metadata(self) -> dict | None:
        """optionally returns STAC metadata for the contained data

        returns:
            dict | None: dict containing STAC metadata for xarray dataset
        """
        bbox = None
        if {"lat", "lon"}.issubset(self.data):
            lats = self.data["lat"].values
            lons = self.data["lon"].values
            bbox = [
                float(lons.min()),
                float(lats.min()),
                float(lons.max()),
                float(lats.max()),
            ]

        time = None
        if "time" in self.data.coords:
            time = pd.Timestamp(self.data["time"].values[0], tz="UTC").isoformat()

        item = {
            "type": "Feature",
            "bbox": bbox,
            "geometry": None
            if bbox is None
            else {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ],
            },
            "properties": {
                "datetime": time,
            },
            "links": [],
        }

        return item

    @classmethod
    def merge(cls, *elements: Self) -> Self:
        """merge the elements into a single XArrayContainer instance
        if no elements are passed return an empty container, if only one is passed acts as the identity function

        Args:
            elements (XArrayContainer): list of xarray containers to merge

        Returns:
            XArrayContainer: XArrayContainer resulting from merging elements
        """
        ds = xr.merge([e.data for e in elements])

        return cls(ds)

    def file_info(self) -> MediaDescription:
        """returns media type and text description of file saved in `write` function, which is netCDF file

        Returns:
            MediaDescription: description of saved format
        """
        return MediaDescription(
            "application/octet-stream",
            "NetCDF of Data",
        )
