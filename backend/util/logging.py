from rich import print
import psutil
import os

def backend_print(*args, **kwargs):
    for arg in args:
        print(f"[bold]BACKEND:[/bold]  {arg}", **kwargs)

def print_memory_usage(name=""):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    if name != "":
        name = f" At [italic]{name}[/italic]:"
    backend_print(f"[bold green]Memory usage:[/bold green]{name} {memory_info.rss / 1024 / 1024:.2f} MB")


