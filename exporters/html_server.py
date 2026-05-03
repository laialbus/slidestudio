import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

from rich import print as rprint
from rich.panel import Panel


def serve_and_open(output_path: Path | None, port: int) -> None:
    serve_dir = Path.cwd()
    if output_path is not None:
        try:
            relative = output_path.relative_to(serve_dir)
        except ValueError:
            relative = output_path
        viewer_url = f"http://localhost:{port}/exporters/html/index.html?file=/{relative}"
    else:
        viewer_url = f"http://localhost:{port}/exporters/html/index.html"

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # suppress access logs

    server = socketserver.TCPServer(("localhost", port), _Handler)
    server.allow_reuse_address = True

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    webbrowser.open(viewer_url)

    rprint(Panel(
        f"[bold green]Server running on http://localhost:{port}[/bold green]\n"
        f"[cyan]Viewer:[/cyan]  {viewer_url}\n\n"
        f"[dim]Press [bold]Ctrl+C[/bold] to stop the server.[/dim]",
        title="[bold white]SlideStudio Viewer[/bold white]",
        border_style="bright_blue",
        expand=False,
    ))

    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
