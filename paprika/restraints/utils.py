PI = 3.1415926535


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
                line = line.split()

                # Distance
                if len(line) == 4:
                    restraints['atoms'].append([line[0], line[1]])
                    restraints['target'].append(float(line[2]))
                    restraints['k'].append(float(line[3]))

                # Angle
                elif len(line) == 5:
                    restraints['atoms'].append([line[0], line[1], line[2]])
                    restraints['target'].append(float(line[3]))
                    restraints['k'].append(float(line[4]))

                # Dihedral
                elif len(line) == 6:
                    restraints['atoms'].append([line[0], line[1], line[2], line[3]])
                    restraints['target'].append(float(line[4]))
                    restraints['k'].append(float(line[5]))

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


def parse_restraints(static=None, host_conf=None, guest=None, wall=None, list_type='tuple'):
    """
    A not so good looking function to parse restraints that is used when writing
    the restraints to file.

    Parameters
    ----------
    static : list
        List of host static DAT_restraint()
    host_conf : list
        List of host conformational DAT_restraint()
    guest : list
        List of guest DAT_restraint()
    wall : list
        List of guest-wall static DAT_restraint()
    list_type : str
        Type of list to return (tuple or dict)

    Returns
    -------
    restraints_list : tuple/dict
        The list of available restraints

    """
    l_s = 0 if static is None else len(static)
    l_h = 0 if host_conf is None else len(host_conf)
    l_g = 0 if guest is None else len(guest)
    l_w = 0 if wall is None else len(wall)

    restraints_list = None

    if l_s != 0 and l_h != 0 and l_g != 0 and l_w != 0:
        if list_type == 'tuple':
            restraints_list = (static + guest + host_conf + wall)
        elif list_type == 'dict':
            restraints_list = {"static": static, "guest": guest, "host": host_conf, "wall": wall}

    if l_s != 0 and l_h != 0 and l_g != 0 and l_w == 0:
        if list_type == 'tuple':
            restraints_list = (static + guest + host_conf)
        elif list_type == 'dict':
            restraints_list = {"static": static, "guest": guest, "host": host_conf}

    if l_s != 0 and l_h == 0 and l_g != 0 and l_w != 0:
        if list_type == 'tuple':
            restraints_list = (static + guest + wall)
        elif list_type == 'dict':
            restraints_list = {"static": static, "guest": guest, "wall": wall}

    if l_s != 0 and l_h != 0 and l_g == 0 and l_w == 0:
        if list_type == 'tuple':
            restraints_list = (static + host_conf)
        elif list_type == 'dict':
            restraints_list = {"static": static, "host": host_conf}

    if l_s != 0 and l_h == 0 and l_g != 0 and l_w == 0:
        if list_type == 'tuple':
            restraints_list = (static + guest)
        elif list_type == 'dict':
            restraints_list = {"static": static, "guest": guest}

    if l_s != 0 and l_h == 0 and l_g == 0 and l_w != 0:
        if list_type == 'tuple':
            restraints_list = (static + wall)
        elif list_type == 'dict':
            restraints_list = {"static": static, "wall": wall}

    if l_s != 0 and l_h == 0 and l_g == 0 and l_w == 0:
        if list_type == 'tuple':
            restraints_list = static
        elif list_type == 'dict':
            restraints_list = {"static": static}

    return restraints_list


def restraint_to_colvar(restraints, phase, window, radians=True, legacy_k=True):
    """
    Extract information about restraints and store in a python dictionary

    Parameters
    ----------
    restraints : list
        List of DAT_restraint() object.
    phase : str
        Which phase of the simulation ('attach', 'pull', 'release').
    window : int
        Current window index
    radians: bool
        Use radians or degrees for angle?
    legacy_k : bool
        Are the restraints based on legacy force constants? Old MD codes
        like AMBER and CHARMM requires the user to half the force constant
        beforehand. New MD codes like GROMACS and NAMD requires the user
        to set the force constant without the 1/2 factor.

    Returns
    -------
    colvar : dict
        A dictionary containing the information of a particular restraint block.

    """
    factor = 1.0
    if legacy_k:
        factor = 2.0

    colvar = {
        "atoms": [],
        "AT": [],
        "KAPPA": [],
        "type": [],
        "factor": factor,
        "ncolvar": len(restraints),
    }
    for restraint in restraints:
        atoms = []
        angle = False

        # Atom indices
        if restraint.index1:
            atoms.append(restraint.index1[0])
        else:
            raise Exception("There must be at least two atoms in a restraint.")

        if restraint.index2:
            atoms.append(restraint.index2[0])
        else:
            raise Exception("There must be at least two atoms in a restraint.")

        if restraint.index3:
            angle = True
            atoms.append(restraint.index3[0])

        if restraint.index4:
            angle = True
            atoms.append(restraint.index4[0])

        # Type of collective variable
        if len(atoms) == 2:
            colvar["type"].append("DISTANCE")
        elif len(atoms) == 3:
            colvar["type"].append("ANGLE")
        elif len(atoms) == 4:
            colvar["type"].append("TORSION")

        # Target and force constant
        target = restraint.phase[phase]["targets"][window]
        force_constant = restraint.phase[phase]["force_constants"][window]
        if angle and radians:
            target *= PI / 180.0

        # Store info to dict
        colvar["atoms"].append(atoms)
        colvar["AT"].append(target)
        colvar["KAPPA"].append(force_constant)

    return colvar
