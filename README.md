# deklare

`deklare` is a lightweight framework to create custom, declarative data processing pipelines. It lets you define **Flows** to process datasets. **Descriptors** describe spatial, temporal, and semantic constraints on the dataset. From these declarations, `deklare` builds a DAG, executing only the necessary steps, and caches intermediate results to avoid redundant computation.

## Installation
The project is managed with **uv**, which can be installed [here](https://docs.astral.sh/uv/getting-started/installation/).

### Setup
Install dependencies using `uv` and sets up pre-commit hooks.
```bash
make setup
```

### Development

Format and lint the codebase before pushing.
```bash
make format
```


## Core Concepts

### Descriptor

A **Descriptor** declaratively specifies *what* data is requested:
- temporal span (time ranges or timestamps)
- spatial extent (latitude / longitude ranges)
- variables, levels, or other dataset-specific dimensions

Descriptors are typed and can be instantiated directly or from dictionaries.

Example descriptor dictionary for `ERA5`:

```JSON
{
    "time": {"start": "2023-02-01T12:00", "end": "2023-02-02T12:00"},
    "latitude": {"start": 48, "end": 47},
    "longitude": {"start": 8, "end": 9},
    "variable": ["2m_temperature"]
}
```

Note that when creating a Descriptor subclass, by default you can only pass arguments during instantiation, that are fields of the Descriptor. In the above example your Descriptor subclass must include a `time`, `latitude`, `longitude`, and `variable` field. If you want to allow any field, no matter if you define them in the Descriptor class, you can use the `PermissiveDescriptor` class.

### Task

A **task** is a processing unit decorated with `@task`. They become nodes in the processing DAG and are used to declare processing flows.

Tasks can be:
- functions
- classes (entry point is `__call__`)

Function-based task:

```python
from deklare import task
from deklar.descriptor import Descriptor

@task()
def processing_function(data: Descriptor):
    ...
```

Class-based task:

```python
@task()
class ProcessingClass:
    def __call__(self, descriptor: Descriptor):
        ...
```

### Descriptor-aware Tasks

If a task requires a descriptor the argument must be named `descriptor`. For class-based tasks, this applies to `__call__`.

To accept plain dictionaries instead of descriptor instances, use `accept_dict_descriptor`. Then the function instantiates a descriptor of class `descriptor_cls` from the kwargs in the dictionary. This is done automatically at runtime.

```python
from deklare.descriptor import Descriptor, accept_dict_descriptor

@accept_dict_descriptor(arg_name="descriptor", descriptor_cls=Descriptor)
def __call__(self, descriptor: Descriptor):
    ...
```

### Flow

A **Flow** is a Python object that orchestrates tasks. Calling a flow builds and executes the DAG. It can combine multiple other classes or functions that may or may not be decorated as tasks, creating nodes. Undecorated functions are executed normally.

Example:

```python
class ExampleFlow:
    def __call__(self, descriptor: Descriptor):
        data = self.loader(descriptor) # may be a @task decorated class
        data = processing_function(data) # may be a @task decorated function
        data = self.persister(data)
        return data
```

### Chunked Processing
`ChunkPersister` enables the chunk-wise execution and caching along declared dimensions. Outputs for the chunks are stored persistently (optionally including STAC metadata). For the final output of the flow the chunk is recombined to include all the data defined in the descriptor. You need to define a subclass of `DataContainer` that is able to store and load your data format. See the `XArrayContainer` as an example implementation.

`segment_slice` defines how to chunk the data. `dataset_scope` potentially limits the global scope of the dataset, limiting descriptors. Pass `None` to not limit. The `reference` defines the points where to align the chunks.

Example configuration:
```python
ChunkPersister(
    store=store,
    data_container=XArrayContainer,
    segment_slice={
        "time": "1 days",
        "variable": 1,
        "latitude": "full",
        "longitude": "full",
    },
    dataset_scope=dataset_scope,
    reference={"time": "1900-01-01"},
    save_metadata=True,
)
```
