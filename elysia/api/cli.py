import click
import uvicorn


@click.group()
def cli():
    """Main command group for Elysia."""
    pass


@cli.command()
@click.option(
    "--port",
    default=8000,
    help="FastAPI Port",
)
@click.option(
    "--host",
    default="localhost",
    help="FastAPI Host",
)
def start(port, host):
    """
    Run the FastAPI application.
    """
    uvicorn.run(
        "elysia.api.app:app",
        host=host,
        port=port,
        reload=True,
    )


if __name__ == "__main__":
    cli()
