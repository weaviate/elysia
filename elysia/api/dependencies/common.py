from elysia.api.services.user import UserManager

# Create the singleton instance at module level
user_manager = UserManager()


def get_user_manager() -> UserManager:
    """Get the singleton UserManager instance."""
    return user_manager
