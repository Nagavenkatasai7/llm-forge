"""Rich terminal UI for the LLM Forge conversational assistant."""

from __future__ import annotations

from llm_forge.chat.engine import ChatEngine


def _print_banner() -> None:
    """Print the welcome banner."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        banner = Text()
        banner.append("LLM Forge", style="bold cyan")
        banner.append(" - Build your own AI model\n", style="dim")
        banner.append("Just tell me what you want to build.\n\n", style="")
        banner.append("Type ", style="dim")
        banner.append("quit", style="bold red")
        banner.append(" to exit.", style="dim")

        console.print(Panel(banner, border_style="cyan", padding=(1, 2)))
        console.print()
    except ImportError:
        print("=" * 50)
        print("  LLM Forge - Build your own AI model")
        print("  Just tell me what you want to build.")
        print("  Type 'quit' to exit.")
        print("=" * 50)
        print()


def _print_response(text: str) -> None:
    """Print the assistant's response with Rich formatting."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        console = Console()
        md = Markdown(text)
        console.print(
            Panel(md, border_style="green", title="Forge", title_align="left", padding=(0, 1))
        )
        console.print()
    except ImportError:
        print(f"\nForge: {text}\n")


def _get_input() -> str:
    """Get user input with a styled prompt."""
    try:
        from rich.console import Console

        console = Console()
        console.print("[bold cyan]You:[/bold cyan] ", end="")
        return input()
    except ImportError:
        return input("You: ")


def launch_chat(provider: str | None = None) -> None:
    """Launch the interactive chat session."""
    _print_banner()

    engine = ChatEngine(provider=provider)

    # Check for API key
    if engine.provider == "none":
        try:
            from rich.console import Console

            console = Console()
            console.print(
                "[yellow]No API key found.[/yellow]\n\n"
                "To use LLM Forge, set one of these environment variables:\n\n"
                "  [bold]export ANTHROPIC_API_KEY=your-key-here[/bold]  (recommended)\n"
                "  [bold]export OPENAI_API_KEY=your-key-here[/bold]\n\n"
                "Get a Claude API key at: [link]https://console.anthropic.com/[/link]\n"
            )
        except ImportError:
            print("No API key found.")
            print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
            print("Get a Claude API key at: https://console.anthropic.com/")
        return

    # Send initial greeting
    try:
        greeting = engine.send(
            "The user just launched llm-forge. Greet them warmly and ask what kind of AI model "
            "they want to build. Keep it to 2-3 sentences. Also detect their hardware."
        )
        _print_response(greeting)
    except Exception as e:
        try:
            from rich.console import Console

            Console().print(f"[red]Error connecting to API:[/red] {e}")
        except ImportError:
            print(f"Error connecting to API: {e}")
        return

    # Main conversation loop
    while True:
        try:
            user_input = _get_input()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input.strip():
            continue

        if user_input.strip().lower() in ("quit", "exit", "q", "bye"):
            try:
                from rich.console import Console

                Console().print("\n[cyan]Thanks for using LLM Forge! Happy training![/cyan]\n")
            except ImportError:
                print("\nThanks for using LLM Forge! Happy training!\n")
            break

        try:
            response = engine.send(user_input)
            _print_response(response)
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            continue
        except Exception as e:
            try:
                from rich.console import Console

                Console().print(f"[red]Error:[/red] {e}")
            except ImportError:
                print(f"Error: {e}")
