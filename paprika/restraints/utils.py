def parse_window(window):
    """
    Utility function to use a path to index a :class:`paprika.restraints.DAT_restraint` instance.

    Parameters
    ----------
    window : str
        A string representation of a particular simulation window

    Returns
    -------
    window : int
        The window number
    phase : str
        The calculation phase

    """
    if window[0] == "a":
        phase = "attach"
    elif window[0] == "p":
        phase = "pull"
    elif window[0] == "r":
        phase = "release"
    else:
        raise Exception("Cannot determine the phase for this restraint.")
    window = int(window[1:])

    return window, phase


def read_restraint_from_file(filename):
    """
    Simple function to read in restraints from an ASCII template file

    Parameters
    ----------
    filename : str
        file name of template file

    Returns
    -------
    restraints : dict
        dictionary of restraints containing information of the atoms, target and spring constant

    Examples
    --------
    ASCII file should contain the atoms (2, 3 or 4), the equilibrium target and spring constant

        :1@C12 :2@C4  12.5 50.0
        :1@O11 :1@C12 :1@O13 :1@C14  115.47 6.0
        :1@C12 :1@O13 :1@C14 :1@C15 -121.16 6.0

    """
    restraints = {'atoms': [], 'target': [], 'k': []}

    with open(filename, 'r') as file:
        for line in file:
            if not line.startswith("#"):
                restr = line.split()

                # Distance
                if len(restr) == 4:
                    restraints['atoms'].append([restr[0], restr[1]])
                    restraints['target'].append(float(restr[2]))
                    restraints['k'].append(float(restr[3]))

                # Angle
                elif len(restr) == 5:
                    restraints['atoms'].append([restr[0], restr[1], restr[2]])
                    restraints['target'].append(float(restr[3]))
                    restraints['k'].append(float(restr[4]))

                # Dihedral
                elif len(restr) == 6:
                    restraints['atoms'].append([restr[0], restr[1], restr[2], restr[3]])
                    restraints['target'].append(float(restr[4]))
                    restraints['k'].append(float(restr[5]))

    return restraints
