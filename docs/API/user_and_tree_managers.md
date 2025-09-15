# User Manager and Tree Manager

There exist two manager classes in Elysia designed to help structure and handle multiple users, each interacting with multiple decision trees. These are:
- The **[TreeManager](../Reference/Managers.md#elysia.api.services.tree)**, which tracks and stores multiple decision trees.
- The **[UserManager](../Reference/Managers.md#elysia.api.services.user)**, which handles multiple `TreeManager`s per user, as well as a `[ClientManager](../Reference/Client.md)` object for each user.


## TreeManager Overview

The `TreeManager` is initialised with a `user_id`, which is the identifier for the user this manager is responsible for.

The `TreeManager` includes configs options which is shared amongst all default trees in the `TreeManager`, which is created at initialisation of the object. Since each `TreeManager` instance is unique to each user, this is intended to be the default user configuration for a specific user. These options are 

- `style` - a string defining the 'style' passed to each tree. This can be used to customize the behaviour of the writing of the LLMs within the decision tree.
- `agent_description` - A description of the agent that will be used in the tree. This helps define the agent's capabilities and behaviour.
- `end_goal` - The ultimate objective or goal that the tree should work towards achieving, how it decides when the decision tree ends.
- `branch_initialisation` - What tools are initialised as a default configuration of the tree.
- `settings` - Optional configuration settings, [see the Settings reference page](../Reference/Settings.md).

The class can also be initialised with `tree_timeout`, which controls the length of time before a particular conversation is cleaned up. [See below for more details.](#timeouts)

Methods within `TreeManager` include:

- **[`add_tree`](../Reference/Managers.md#elysia.api.services.tree.TreeManager.add_tree)** - create a tree object with a `conversation_id` parameter, and optionally pass unique configuration options for that specific tree (which will not be overrided if the default `settings` object in the `TreeManager` is changed).
- **[`configure`](../Reference/Managers.md#elysia.api.services.tree.TreeManager.configure)** - a wrapper for the `configure` method for the `TreeManager`'s user `settings` object.
- **[`process_tree`](../Reference/Managers.md#elysia.api.services.tree.TreeManager.process_tree)** - an async function which runs the tree initialised with a `conversation_id` for a given `query` (user prompt).
- **[`check_all_trees_timeout`](../Reference/Managers.md#elysia.api.services.tree.TreeManager.check_all_trees_timeout)** - checks if any trees have been active for longer than `tree_timeout`, and removes them from the `TreeManager` if so.

For a complete view of all the methods, [see the reference page](../Reference/Managers.md#elysia.api.services.tree).

## UserManager Overview

The `UserManager` manages both a `TreeManager` and a `ClientManager` per user. Essentially, it contains a `users` dictionary which is keyed by different `user_id`s and has these managers for each user.

It has initialisations:
- `user_timeout` - controls how many minutes a user needs to be inactive before getting timed out (if you run the `check_all_users_timeout` method)
- `tree_timeout` - initialisation parameter passed down to all `TreeManager` instances for each user.
- `client_timeout` - initialisation parameter passed down to all `ClientManager` instances for each user.

It has methods such as:

- **[`add_user_local`](../Reference/Managers.md#elysia.api.services.user.UserManager.add_user_local)** - creates a user as well as their `TreeManager` and `ClientManager`. Can pass configuration options to create the user with these default configurations.
- **[`get_user_local`](../Reference/Managers.md#elysia.api.services.user.UserManager.get_user_local)** - retrieve a user from the users dictionary, will raise an error if there is no user with that ID.
- **[`initialise_tree`](../Reference/Managers.md#elysia.api.services.user.UserManager.initialise_tree)** - create a tree within the tree manager for a particular user.
- **[`process_tree`](../Reference/Managers.md#elysia.api.services.user.UserManager.process_tree)** - runs the tree for a given `user_id` and `conversation_id` (tree within that user). Automatically sends error payloads if the user or tree has been timed out.
- **[`check_all_users_timeout`](../Reference/Managers.md#elysia.api.services.user.UserManager.check_all_users_timeout)** - loop over all users stored in the user manager and removes any that have timed out.
- **[`check_all_trees_timeout`](../Reference/Managers.md#elysia.api.services.user.UserManager.check_all_trees_timeout)** - loop over all trees for all users in the user manager and removes any that have timed out.
- **[`check_restart_clients`](../Reference/Managers.md#elysia.api.services.user.UserManager.check_restart_clients)** - loop over all clients for all users and call the `restart_client` functions, which only restarts clients if they have exceeded the `client_timeout` parameter since they were last used.

For a complete view of all the methods, [see the reference page](../Reference/Managers.md#elysia.api.services.user).

## Timeouts
 
If you're using the [UserManager](../Reference/Managers.md#elysia.api.services.user), you can set a scheduled `user_manager.check_all_trees_timeout()`, `user_manager.check_all_users_timeout()` and/or `user_manager.check_restart_clients()`; which will remove empty trees/users if they've been inactive for a period of time, or restart the Weaviate clients if they are also inactive. These time periods can be configured on initialisation of the UserManager, i.e.

```python
UserManager(tree_timeout: datetime.timedelta | int, user_timeout: datetime.timedelta | int, client_timeout: datetime.timedelta | int))
```

This defaults to `TREE_TIMEOUT`, `USER_TIMEOUT` and `CLIENT_TIMEOUT` respectively in the environment variables if not set (in minutes), which itself defaults to 10 minutes. If they are set to 0, then no users/trees/clients will be restart. Not restarting the clients is not recommended.

If you're using a [TreeManager](../Reference/Managers.md#elysia.api.services.tree) only (and not a `UserManager`), you can do the same with `tree_manager.check_all_trees_timeout()`, with same defaults.

If you're only using the [ClientManager], you can call `restart_client` and `restart_async_client`, which automatically checks and restarts the clients individually if they have passed the `client_timeout` threshold.

### Example Scheduler with FastAPI lifespan

For example, if using [FastAPI](https://fastapi.tiangolo.com/), you can set an automatic scheduler such as:

```python
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler

async def check_user_timeouts():
    user_manager = get_user_manager()
    await user_manager.check_all_users_timeout()

async def check_tree_timeouts():
    user_manager = get_user_manager()
    await user_manager.check_all_trees_timeout()

async def check_restart_clients():
    user_manager = get_user_manager()
    await user_manager.check_restart_clients()

@asynccontextmanager
async def lifespan(app: FastAPI):
    user_manager = get_user_manager()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_tree_timeouts, "interval", seconds=23)
    scheduler.add_job(check_user_timeouts, "interval", seconds=29)
    scheduler.add_job(check_restart_clients, "interval", seconds=31)

    scheduler.start()
    yield
    scheduler.shutdown()

    await user_manager.close_all_clients()
```
Where the `get_user_manager()` function returns a globally defined `UserManager` (and doesn't create a new one when it's called).

This automatically runs the functions in the user manager every 23, 29 and 31 seconds, respectively.