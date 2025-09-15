Below is the full contents of the `return_types.py` file, which specifies the different ways that Elysia sends results whenever `tree.async_run()` yields an object. This is primarily intended for interacting with the frontend - giving the frontend a specific 'payload type' and ensuring that all objects sent conform to the same 'mapping'.

Each individual payload type has its own mapping. In this file, the keys of each dictionary are the allowed fields, and the values are descriptions for each field. The `specific_return_types` has broad descriptions of each class.

```python
--8<-- "elysia/util/return_types.py"
```