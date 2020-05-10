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


def extract_guest_restraints(structure, guest_resname, restraints):
    """
    Utility function to extract the guest restraints in the form of:
        [r, theta, phi, alpha, beta, gamma]

    If there is no restraints applied to a particular reaction coordinate
    a None will be inserted.

    This is useful for the free-energy analysis when computing the
    ref_state_work

    Parameters
    ----------
    structure : parmed.Structure
        parmed structure object of the system
    guest_resname : str
        Residue name of the guest molecule
    restraints : list
        list of restraints

    Returns
    -------
    guest_restraints : list
        list of guest-specific restraints

    """
    guest_resname = guest_resname.upper()

    r = None
    theta = None
    phi = None
    alpha = None
    beta = None
    gamma = None

    for restraint in restraints:

        mask2_residue_name = structure[restraint.mask2].residues[0].name

        # Distance
        if "DM1" in restraint.mask1 and guest_resname in mask2_residue_name and not restraint.mask3 and not \
                restraint.mask4:
            r = restraint

        # Angle
        if restraint.mask3 and not restraint.mask4:
            mask3_residue_name = structure[restraint.mask3].residues[0].name

            if "DM2" in restraint.mask1 and "DM1" in restraint.mask2 and guest_resname in mask3_residue_name:
                theta = restraint

            if "DM1" in restraint.mask1 and guest_resname in mask2_residue_name and guest_resname in \
                    mask3_residue_name:
                beta = restraint

        # Dihedral
        if restraint.mask4:
            mask3_residue_name = structure[restraint.mask3].residues[0].name
            mask4_residue_name = structure[restraint.mask4].residues[0].name

            if "DM3" in restraint.mask1 and "DM2" in restraint.mask2 and "DM1" in restraint.mask3 and guest_resname \
                    in mask4_residue_name:
                phi = restraint

            if "DM2" in restraint.mask1 and "DM1" in restraint.mask2 and guest_resname in mask3_residue_name \
                    and guest_resname in mask4_residue_name:
                alpha = restraint

            if "DM1" in restraint.mask1 and guest_resname in mask2_residue_name and guest_resname in \
                    mask3_residue_name and guest_resname in mask4_residue_name:
                gamma = restraint

    guest_restraints = [r, theta, phi, alpha, beta, gamma]

    return guest_restraints
