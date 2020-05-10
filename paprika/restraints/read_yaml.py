import logging
import re

import yaml

logger = logging.getLogger(__name__)


def read_yaml(file):
    """
    Read `Taproom <https://github.com/slochower/host-guest-benchmarks>`_ -style YAML-formatted instructions for
    preparing host-guest systems.

    Parameters
    ----------
    file: Path-like
        A YAML-formatted file.
    Returns
    -------
    yaml_data: dict
        Dictionary containing simulation setup parameters.

    """

    with open(file, "r") as f:
        yaml_data = yaml.safe_load(f)
    logger.debug(yaml_data)

    if "aliases" in yaml_data.keys():
        logger.debug("Dealiasing atom masks...")
        yaml_data = de_alias(yaml_data)

    return yaml_data


def read_yaml_aliases(file):
    with open(file, "r") as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    anchor_def = ["D1","D2","D3","H1","H2","H3","G1","G2","G3"]
    int_def = [1,1,1,1,1,1,1,1,1]

    anchor_atoms = {}
    if "aliases" in yaml_data.keys():
        for alias in yaml_data["aliases"]:
            definition = list(alias.items())[0]
            anchor = definition[0]
            atoms = definition[1]
            if anchor in anchor_def:
                anchor_atoms[anchor] = atoms
                int_def[anchor_def.index(anchor)] = 0
        if sum(int_def) != 0:
            for i in range(len(int_def)):
                if int_def[i] != 0:
                    anchor_atoms[anchor_def[i]] = 0

    return anchor_atoms


def multiple_replace(dict, text):
    """
    Create a regular expression to do multiple find and replace.
    """
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def de_alias(yaml_data):
    """
    Replace aliased atoms in a ``taproom`` recipe.
    """
    mapping_list = yaml_data["aliases"]
    mapping_dictionary = {}

    for atom_pair in mapping_list:
        mapping_dictionary.update(atom_pair)
    logger.debug(f"Found mapping: {mapping_dictionary}")

    for restraint_type, restraint_type_list in yaml_data["restraints"].items():
        for restraint in restraint_type_list:
            atoms = restraint["restraint"]["atoms"]
            mapped_atoms = multiple_replace(mapping_dictionary, atoms)
            logger.info(f"{atoms} → {mapped_atoms}")
            restraint["restraint"]["atoms"] = mapped_atoms

    if "symmetry_correction" in yaml_data.keys():
        for restraint in yaml_data["symmetry_correction"]["restraints"]:
            atoms = restraint["atoms"]
            mapped_atoms = multiple_replace(mapping_dictionary, atoms)
            logger.info(f"{atoms} → {mapped_atoms}")
            restraint["atoms"] = mapped_atoms

    return yaml_data
