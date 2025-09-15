import psutil

from elysia.api.services.user import UserManager


def get_total_memory():
    return round(psutil.virtual_memory().total / (1024 * 1024), 2)


def get_free_memory():
    return round(psutil.virtual_memory().available / (1024 * 1024), 2)


def get_cpu_usage():
    return psutil.cpu_percent(interval=1)


def get_memory_usage():
    return psutil.virtual_memory().percent


def get_disk_usage():
    return psutil.disk_usage("/").percent


def get_number_local_users(user_manager: UserManager):
    return len(user_manager.users)


async def get_average_user_memory(user_manager: UserManager):
    if len(user_manager.users) == 0:
        return 0, 0
    avg_user_memory = 0
    avg_tree_memory = 0
    for user in user_manager.users.values():
        user_memory = 0
        for tree in user["tree_manager"].trees.values():
            user_memory += tree["tree"].detailed_memory_usage()["total"] / (1024 * 1024)

        if len(user["tree_manager"].trees) > 0:
            avg_tree_memory += user_memory / len(user["tree_manager"].trees)

        avg_user_memory += user_memory

    return (
        round(avg_user_memory / len(user_manager.users), 2),
        round(avg_tree_memory / len(user_manager.users), 2),
    )


# async def get_average_user_requests(user_manager: UserManager):
#     return await user_manager.get_average_user_requests()


async def print_resources(
    user_manager: UserManager | None = None, save_to_file: bool = False
):
    if user_manager is not None:
        avg_user_memory, avg_tree_memory = await get_average_user_memory(user_manager)
        # avg_user_requests = await get_average_user_requests(user_manager)
        # num_users_db = await get_number_local_users_db(user_manager)

    print("\n\n\n\n")
    print("-" * 100)
    print(f"ELYSIA RESOURCES")
    print("-" * 100)
    print(f"Total memory: {get_total_memory()} MB")
    print(f"Free memory: {get_free_memory()} MB")
    print(f"CPU usage: {get_cpu_usage()}%")
    print(f"Memory usage: {get_memory_usage()}%")
    print(f"Disk usage: {get_disk_usage()}%")
    if user_manager is not None:
        print("\n\n")
        print("-" * 100)
        print(f"USER STATISTICS")
        print("-" * 100)
        print(f"Number of local users: {get_number_local_users(user_manager)}")
        # print(f"Number of unique users in database: {num_users_db}")
        print(f"Average user memory usage: {avg_user_memory} MB")
        print(f"Average tree memory usage: {avg_tree_memory} MB")
        # print(f"Average user requests: {avg_user_requests}")
    print("-" * 100)
    print("\n\n\n\n")

    if save_to_file:
        with open("resources.txt", "w") as f:
            f.write("-" * 100)
            f.write("\n")
            f.write(f"ELYSIA RESOURCES\n")
            f.write("-" * 100)
            f.write("\n")
            f.write(f"Total memory: {get_total_memory()} MB\n")
            f.write(f"Free memory: {get_free_memory()} MB\n")
            f.write(f"CPU usage: {get_cpu_usage()}%\n")
            f.write(f"Memory usage: {get_memory_usage()}%\n")
            f.write(f"Disk usage: {get_disk_usage()}%\n")
            if user_manager is not None:
                f.write("\n\n")
                f.write("-" * 100)
                f.write("\nUSER STATISTICS\n")
                f.write("-" * 100)
                f.write("\n\n")
                f.write(
                    f"Number of local users: {get_number_local_users(user_manager)}\n"
                )
                # f.write(f"Number of unique users in database: {num_users_db}\n")
                f.write(f"Average user memory usage: {avg_user_memory} MB\n")
                f.write(f"Average tree memory usage: {avg_tree_memory} MB\n")
                # f.write(f"Average user requests: {avg_user_requests}\n")


# if __name__ == "__main__":
#     await print_resources()
