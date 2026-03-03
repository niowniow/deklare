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

from __future__ import annotations

import datetime
import inspect
from typing import Annotated, Any, Callable, Generic, Iterable, Iterator, Self, TypeVar, get_args

import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import DatetimeScalar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from pydantic.functional_validators import AfterValidator

BLOCKED_DESCRIPTOR_ATTRS = ["_deklare_attrs"]


def _to_datetime(v: DatetimeScalar) -> pd.Timestamp:
    """helper function to create DateTimeType"""
    return pd.to_datetime(v, utc=True).tz_localize(None)


DatetimeTT = TypeVar("DatetimeTT", int, float, str, datetime.date, datetime.datetime, type(np.datetime64))  # type: ignore


DateTimeType = Annotated[DatetimeTT, AfterValidator(_to_datetime)]


T = TypeVar("T", int, float, DateTimeType)


class Range(BaseModel, Generic[T]):
    start: T
    end: T


DatetimeRange = Range[DateTimeType]


def _transform_to_nested(input_dict: dict, separator: str = ".") -> dict:
    """transform the flat JSON keys with dots (split) into nested JSON keys

    Args:
        input_dict (dict): dict with possibly flattened JSON keys
        separator (str): seperator used to flatten keys
    """
    transformed = {}
    for key, value in input_dict.items():
        parts = key.split(separator)
        current = transformed
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return transformed


class Descriptor(BaseModel, validate_assignment=True):
    """Descriptor to represent queries for data

    Attributes:
        config (dict[str, Any]): config of descriptor
        _deklare_attrs (dict[str, Any]): allows private attributes to be added in deklare, e.g. when configuring graph
            are ignored by __getitem__, __setitem__ and __contain__
    """

    config: dict[str, Any] = Field(default_factory=dict)

    _deklare_attrs: dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    @model_validator(mode="before")
    def dynamic_validator(cls, values: dict[str, Any]) -> dict[str, Any]:
        """validate and transform input data before model instantiation
        converts nested dictionaries into `DatetimeRange` or `Range` instances if required

        Args:
            values (dict[str, Any]): raw input values to validate

        Returns:
            dict[str, Any]: validated data
        """

        kwargs = {}
        values = _transform_to_nested(values)

        for key, value in values.items():
            if key in cls.__pydantic_fields__:
                field_info = cls.__pydantic_fields__[key]
                if DatetimeRange in get_args(field_info.annotation) and isinstance(value, dict):
                    kwargs[key] = DatetimeRange(**value)
                elif Range in get_args(field_info.annotation) and isinstance(value, dict):
                    kwargs[key] = Range(**value)
                else:
                    kwargs[key] = value

        return kwargs

    def to_dict(self, *, remove_none: bool = True) -> dict[str, Any]:
        """transforms instance into dict representation

        Args:
            remove_none (bool): ignores None values if True. defaults to True

        Returns:
            dict[str, Any]: dictionary containing data in instance
        """
        result = {}
        for field_name in self:
            field_value = self[field_name]
            if remove_none and field_value is None:
                continue
            elif isinstance(field_value, (DatetimeRange, Range)):
                result[field_name] = dict(field_value)
            else:
                result[field_name] = field_value
        return result

    def update(self, other: Descriptor) -> None:
        """updates instance given other descriptor

        Args:
            other (Descriptor): descriptor to use to update instance
        """
        for k, v in other.model_dump().items():
            setattr(self, k, v)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """returns the attribute of the descriptor

        Args:
            key (str): key of attribute from descriptor

        Returns:
            Any: value corresponding to key

        Raises:
            KeyError: if key does not exist
        """

        if key in self.__pydantic_fields__:
            return getattr(self, key)

        raise KeyError(key)

    def _get_internal(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        """returns the internal attribute corresponding to key

        Args:
            key (str): key of arg
            default (Any): (defaults to None) default value to return if key is not present

        Returns:
            Any: value corresponding to key or None of not present
        """
        return self._deklare_attrs.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """sets the attribute of the descriptor

        Args:
            key (str): key of attribute from descriptor
            value (Any): value to set

        Raises:
            KeyError: if key does not exist
        """

        if key in self.__pydantic_fields__:
            setattr(self, key, value)
        else:
            raise KeyError(key)

    def _set_internal(self, key: str, value: Any) -> None:  # noqa: ANN401
        """sets the internal attribute to corresponding value

        Args:
            key (str): key of arg
            value (Any): value of arg
        """
        self._deklare_attrs[key] = value

    def __contains__(self, key: str) -> bool:
        """check if key is part of descriptor

        Args:
            key (str): key to check

        Returns:
            (bool): True if key is part of descriptor
        """

        if key in self.__pydantic_fields__:
            return True

        return False

    def __iter__(self) -> Iterator[str]:
        """iterate over the field names

        Returns:
            Iterator[str]: iterator over field names
        """
        yield from self.__pydantic_fields__

    def get_config(self, key: str, default: Any) -> Any:  # noqa: ANN401
        return self.config.get(key, default)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """create descriptor from data

        Args:
            data (dict[str, Any]): data to use to create descriptor

        Returns:
            Self: containing data
        """
        kwargs = {}

        data = _transform_to_nested(data)

        for key, value in data.items():
            if key in cls.model_fields:
                field_info = cls.model_fields[key]
                if DatetimeRange in get_args(field_info.annotation) and isinstance(value, dict):
                    kwargs[key] = DatetimeRange(**value)
                elif Range in get_args(field_info.annotation) and isinstance(value, dict):
                    kwargs[key] = Range(**value)
                else:
                    kwargs[key] = value

        return cls(**kwargs)

    # ToDo: find a cleaner method than converting to dict and back
    @classmethod
    def update_from_config_dict(cls, descriptor: Self | dict, config: Self | dict) -> Self:
        """checks if descriptor has config defaults defined in descriptor['config']
        replaces values in config if not present in descriptor

        Args:
            descriptor (Self| dict): original descriptor
            config (Self | dict): config with default values

        Returns:
            Self: updated descriptor
        """
        if isinstance(descriptor, Descriptor):
            descriptor = descriptor.to_dict()

        if isinstance(config, Descriptor):
            config = config.to_dict()

        if "config" not in descriptor:
            return cls.from_dict(descriptor)

        config_keys = descriptor["config"]

        for key, entry in config_keys.items():
            if key not in config:
                raise KeyError(f"Invalid config key '{key}' (not present)")
            if entry not in config[key]:
                raise KeyError(f"Invalid config key '{key}'/'{entry}' (not present)")

            # dont replace if already in descriptor
            if key not in descriptor:
                descriptor[key] = config[key][entry]

        return cls.from_dict(descriptor)


class PermissiveDescriptor(Descriptor):
    """Descriptor subclass that allows to use extra fields
    can be used e.g. to instantiate with additional fields in from_dict
    """

    model_config = ConfigDict(extra="allow")

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """returns the attribute of the descriptor

        Args:
            key (str): key of attribute from descriptor

        Returns:
            Any: value corresponding to key or None if not present
        """

        if key in BLOCKED_DESCRIPTOR_ATTRS:
            return None

        return getattr(self, key, None)

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """sets the attribute of the descriptor or creates it if it doesnt exist

        Args:
            key (str): key of attribute from descriptor
            value (Any): value to set

        Raises:
            KeyError: if key is in blocked attributes
        """

        if key in BLOCKED_DESCRIPTOR_ATTRS:
            raise KeyError(f"{key} is reserved for deklare internal use")

        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """check if key is part of descriptor

        Args:
            key (str): key to check

        Returns:
            (bool): True if key is part of descriptor
        """

        if key in BLOCKED_DESCRIPTOR_ATTRS:
            return False

        return hasattr(self, key)

    def __iter__(self) -> Iterator[str]:
        """iterate over the field names

        Returns:
            Iterator[str]: iterator over field names
        """
        yield from list(self.__pydantic_fields__.keys()) + list(self.__pydantic_extra__.keys())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """create descriptor from data

        Args:
            data (dict[str, Any]): data to use to create descriptor

        Returns:
            Self: containing data
        """
        kwargs = {}

        data = _transform_to_nested(data)

        for key, value in data.items():
            field_info = cls.model_fields.get(key)
            if field_info and DatetimeRange in get_args(field_info.annotation) and isinstance(value, dict):
                kwargs[key] = DatetimeRange(**value)
            elif field_info and Range in get_args(field_info.annotation) and isinstance(value, dict):
                kwargs[key] = Range(**value)
            else:
                kwargs[key] = value

        return cls(**kwargs)


def accept_dict_descriptor(
    arg_name: str | Iterable[str] = "descriptor", descriptor_cls: type[Descriptor] | None = PermissiveDescriptor
) -> Callable:
    """decorator to allow accepting dicts instead of Descriptor instances and handles instantiation for you

    Args:
        arg_name (str | Iterable[str]): argument names in the function that should be converted to Descriptor if dict
            defaults to "descriptor"
        descriptor_cls (type[Descriptor] | None): Descriptor class used for validation. defaults to PermissiveDescriptor

    Returns:
        Callable: decorated function
    """
    if descriptor_cls is None:
        descriptor_cls = PermissiveDescriptor

    if isinstance(arg_name, str):
        arg_name = [arg_name]

    def outer_wrapper(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        def inner_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            bound = sig.bind_partial(*args, **kwargs)
            for arg in arg_name:
                if arg in bound.arguments:
                    val = bound.arguments[arg]
                    if isinstance(val, dict):
                        bound.arguments[arg] = descriptor_cls.from_dict(val)

            return fn(**bound.arguments)

        return inner_wrapper

    return outer_wrapper
