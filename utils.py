import json
import logging

#TODO: Review functionality;
def read_json(filepath: str) -> dict:
    """
    Reads a JSON file and returns its contents as a dictionary.

    Parameters
    ----------
    filepath : str
        Path to the JSON file to be read.

    Returns
    -------
    dict
        Dictionary containing the contents of the JSON file.
    """

    with open(filepath, "r") as file:
        data = json.load(file)

    return data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
