import traceback
from typing import Any, Callable, Iterable, TypeVar

import numpy as np

try:
    import torch
except Exception:
    RuntimeWarning("Failed to load torch. Install it to use threaded (pre-)loading of datasets")


from tqdm import tqdm

from .descriptor import Descriptor
from .utils import NodeFailedError

DatasetIdx = TypeVar("DatasetIdx", int, tuple[int, int], tuple[int, Iterable[int]])


class Dataset:
    """dataset class to gather data from multiple flows and descriptors

    Attributes:
        singleton (bool): True if only one flow
        flows (list[Callable]): list of flows to gather data from
        transforms (list[Callable]): list of additional tranforms to apply to data from flow
        dataset_descriptors (list[Descriptor]): descriptors that can be used to query the data
        indices (list[int]): potentially valid indices of descriptors
        valid_indices (set[int]): cache which indices are valid
        invalid_indices (set[int]): cache which indices are invalid
    """

    singleton: bool
    flows: list[Callable]
    transforms: list[Callable]
    dataset_descriptors: list[Descriptor]
    indices: list[int]
    valid_indices: set[int]
    invalid_indices: set[int]

    def __init__(
        self,
        descriptors: list[Descriptor],
        flows: list[Callable] | Callable,
        transforms: list[Callable] | Callable | None = None,
    ) -> None:
        self.singleton = False
        if not isinstance(flows, list):
            self.singleton = True
            flows = [flows]

        self.flows = flows

        if not isinstance(transforms, list):  # assume same transform for all flows
            transforms = [transforms] * len(self.flows)

        self.transforms = transforms

        self.dataset_descriptors = descriptors
        self.indices = range(len(descriptors))

        self.invalid_indices = set()
        self.valid_indices = set()

    @property
    def descriptors(self) -> Descriptor:
        return self.dataset_descriptors[self.indices]

    def mask_invalid(self) -> None:
        """remove certainly invalid indices"""
        self.indices = [x for x in self.indices if x not in self.invalid_indices]

    # ToDo: check if this needs to be so complicated
    def valid(self, idx: int) -> bool:
        """check validity of index for dataset

        Args:
            idx (int): index to check

        Returns:
            bool: True if index is valid, False if invalid
        """
        if idx in self.valid_indices:
            return True
        if idx in self.invalid_indices:
            return False

        # If we get here the idx was never tested for validity
        # so let's do it
        try:
            result = self.__getitem__(idx, only_validity=True)
            if isinstance(result, NodeFailedError):
                self.invalid_indices.add(idx)
                return False
            elif isinstance(result, tuple):
                return all([not isinstance(item, NodeFailedError) for item in result])

            self.valid_indices.add(idx)
            return True
        except Exception:
            tqdm.write(traceback.format_exc())
            return False

    def __len__(self) -> int:
        """returns length of dataset"""
        return len(self.indices)

    def __getitem__(self, idx: DatasetIdx) -> Any | tuple[Any]:  # noqa: ANN401
        """make dataset indexable

        Args:
            idx (DatasetIdx): index of descriptor, if tuple second element chooses flow(s)
        """
        singleton = self.singleton

        stream_select = np.arange(len(self.flows))
        if isinstance(idx, tuple):
            idx, stream_select = idx
            if not isinstance(stream_select, (list, np.ndarray)):
                singleton = True
                stream_select = [stream_select]

        internal_idx = self.indices[idx]

        descriptor = self.dataset_descriptors[internal_idx]
        out = []
        for stream in stream_select:
            # check if dataset was persisted before
            values = None
            values = self.flows[stream].query(descriptor)

            if self.transforms[stream] is not None:
                values = self.transforms[stream](values)

            out.append(values)

        if singleton:
            return out[0]

        return tuple(out)

    def check_validity(self, batch_size: int = 1, num_workers: int = 0) -> None:
        """checks which indices are valid for dataset using pytorch

        Args:
            batch_size (int): batch size to use when checking index validity. defaults to 1
            num_workers (int): number of workers to use when checking index validity
                defaults to 0 (as many workers as cores)
        """
        tmp_transforms = self.transforms
        self.transforms = None

        this = self

        class TmpClass:
            def __getitem__(self, idx: int) -> tuple:
                return (idx, this.valid(idx))

            def __len__(self) -> int:
                return len(this)

        for batch in tqdm(
            torch.utils.data.dataloader.DataLoader(
                TmpClass(),
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=False,
                shuffle=False,
                collate_fn=lambda x: x,
            )
        ):
            # We are updating the valid_indices and invalid_indices here in the main thread
            # and not within the possibly parallelized self.valid() calls
            for idx, valid in batch:
                if valid:
                    self.valid_indices.add(idx)
                else:
                    self.invalid_indices.add(idx)

        self.transforms = tmp_transforms

    def preload(self, batch_size: int = 1, num_workers: int = 0) -> None:
        """Using pytorch to preload this dataset, i.e. run through the whole dataset once
        the caching/persisting will happen inside the individual flows

        Args:
            batch_size (int): batch size for loading. defaults to 1
            num_workers (int): number of parallel workers. defaults to 0
        """
        temp_transforms = self.transforms
        self.transforms = None

        for _ in tqdm(
            torch.utils.data.dataloader.DataLoader(
                self,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=False,
                shuffle=False,
                collate_fn=lambda _: [],
            )
        ):
            continue

        self.transforms = temp_transforms
