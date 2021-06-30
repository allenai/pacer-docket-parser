import click
import logging
from . import commands


logger = logging.getLogger(__name__)


@click.group(context_settings={"help_option_names": ["--help", "-h"]})
@click.option("--verbose", "-v", is_flag=True, help="Set log level to DEBUG.")
def docketparser_cli(verbose):
    """An interface for managing PDFs for PAWLS."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)

subcommands = [
    commands.parse_all
]

for subcommand in subcommands:
    docketparser_cli.add_command(subcommand)


if __name__ == "__main__":
    docketparser_cli()