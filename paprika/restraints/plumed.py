import logging

from paprika.restraints.utils import parse_window

logger = logging.getLogger(__name__)

PI = 3.14159265359


def plumed_colvar_file(file, restraints, window):
    """
    Writes a PLUMED colvar file for a specific window.

    Parameters
    ----------
    file: class '_io.TextIOWrapper'
        The file object handle to save the plumed file
    restraints: dict
        The pAPRika restraint to be used.
    window: str
        The calculation window that will be used to index the restraint values.


    Examples
    --------

    for window in self.window_list:
        with open(f"windows/{window}/plumed.dat", "w") as file:
            if window[0] == 'a':
                restraints = {
                    "static": static_restraints,
                    "guest": guest_restraints,
                    "host": conformational_restraints,
                    "wall": guest_wall_restraints,
                }
            plumed_colvar_file(file, restraints, window)

    """

    window, phase = parse_window(window)
    file.write("UNITS LENGTH=A ENERGY=kcal/mol TIME=ns\n")

    if "static" in restraints.keys():
        colvar = restraint_to_colvar(restraints["static"], phase, window)
        write_to_plumed(file, colvar, "static")

    if "host" in restraints.keys():
        colvar = restraint_to_colvar(restraints["host"], phase, window)
        write_to_plumed(file, colvar, "host")

    if "guest" in restraints.keys():
        colvar = restraint_to_colvar(restraints["guest"], phase, window)
        write_to_plumed(file, colvar, "guest")

    if "wall" in restraints.keys():
        colvar = restraint_to_colvar(restraints["wall"], phase, window)
        write_to_plumed(file, colvar, "wall")


def restraint_to_colvar(restraints, phase, window):
    """
    Extract information about restraints and store in a python dictionary

    Parameters
    ----------
    restraints : list
        List of DAT_restraint() object
    phase : str
        Which phase of the simulation ('attach', 'pull', 'release')
    window : str
        Current window folder (i.e. 'a000', 'a001', ...)

    Returns
    -------
    colvar : dict
        A dictionary containing the information of a particular restraint block

    """
    colvar = {
        "atoms": [],
        "AT": [],
        "KAPPA": [],
        "type": [],
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
        if angle:
            target *= PI / 180.0

        # Store info to dict
        colvar["atoms"].append(atoms)
        colvar["AT"].append(target)
        colvar["KAPPA"].append(force_constant)

    return colvar


def write_to_plumed(file, colvar, block, legacy_k=False):
    """
    Write collective variable and restraints to file

    Parameters
    ----------
    file : class '_io.TextIOWrapper'
        The file object handle to save the plumed file
    colvar : dict
        Dictionary containing information about the collective variable
    block : str
        Restraint type for naming purposes
    legacy_k : bool
        Are the restraints based on legacy force constants? Old MD codes 
        like AMBER and CHARMM requires the user to half the force constant
        beforehand. New MD codes like GROMACS and NAMD requires the user
        to set the force constant without the 1/2 factor.


    Examples
    --------

    # static restraints
    s_1: DISTANCE ATOMS=170,17 NOPBC
    s_2: ANGLE ATOMS=171,170,17 NOPBC
    s_3: ANGLE ATOMS=170,17,43 NOPBC
    s_4: TORSION ATOMS=172,171,170,17 NOPBC
    s_5: TORSION ATOMS=171,170,17,43 NOPBC
    s_6: TORSION ATOMS=170,17,43,94 NOPBC
    RESTRAINT ...
    ARG=s_1,s_2,s_3,s_4,s_5,s_6,
    AT=7.8612,2.4389,1.1843,2.3843,-2.6366,-1.6021,
    KAPPA=10.00,100.00,100.00,100.00,100.00,100.00,
    LABEL=static

    """
    if legacy_k:
        factor = 1.0
    else:
        factor = 2.0

    file.write(f"# {block} restraints\n")
    arg = ""
    at = ""
    kappa = ""

    for ndx in range(colvar["ncolvar"]):
        atoms = f""
        if colvar["type"][ndx] == "DISTANCE":
            atoms = f"{colvar['atoms'][ndx][0]},{colvar['atoms'][ndx][1]}"
        elif colvar["type"][ndx] == "ANGLE":
            atoms = (
                f"{colvar['atoms'][ndx][0]},{colvar['atoms'][ndx][1]},"
                f"{colvar['atoms'][ndx][2]}"
            )
        elif colvar["type"][ndx] == "TORSION":
            atoms = (
                f"{colvar['atoms'][ndx][0]},{colvar['atoms'][ndx][1]},"
                f"{colvar['atoms'][ndx][2]},{colvar['atoms'][ndx][3]}"
            )

        file.write(f"{block[0]}_{ndx + 1}: {colvar['type'][ndx]} ATOMS={atoms} NOPBC\n")
        arg += f"{block[0]}_{ndx + 1} "
        at += f"{colvar['AT'][ndx]:0.4f} "
        kappa += f"{factor * colvar['KAPPA'][ndx]:0.2f} "

    if block == "wall":
        bias = "UPPER_WALLS"
    else:
        bias = "RESTRAINT"

    file.write(f"{bias} ...\n")
    file.write(f"ARG={arg.replace(' ', ',')}\n")
    file.write(f"AT={at.replace(' ', ',')}\n")
    file.write(f"KAPPA={kappa.replace(' ', ',')}\n")
    file.write(f"LABEL={block}\n")
    file.write(f"... {bias}\n")


def write_dummy_restraints(file, dummy_atoms, kpos=50.0, legacy_k=False):
    """
    Append to the plumed.dat file the dummy atoms colvar and restraints

    Parameters
    ----------
    file : class '_io.TextIOWrapper'
        The file object handle to save the plumed file
    dummy_atoms : dict
        Dictionary containing information about the dummy atoms
    kpos : float
        Spring constant used to restrain dummy atoms (kcal/mol/A^2)
    legacy_k : bool
        Are the restraints based on legacy force constants? Old MD codes 
        like AMBER and CHARMM requires the user to half the force constant
        beforehand. New MD codes like GROMACS and NAMD requires the user
        to set the force constant without the 1/2 factor.

    Returns
    -------
    None

    Example
    -------

    dm1: POSITION ATOM = 170 NOPBC
    dm2: POSITION ATOM = 171 NOPBC
    dm3: POSITION ATOM = 172 NOPBC
    RESTRAINT...
    ARG = dm1.x, dm1.y, dm1.z, dm2.x, dm2.y, dm2.z, dm3.x, dm3.y, dm3.z
    AT = 19.68, 20.3, 26.9, 19.68, 20.3, 23.9, 19.68, 22.5, 21.7
    KAPPA = 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0
    ...
    RESTRAINT

    """
    if legacy_k:
        factor = 1.0
    else:
        factor = 2.0

    file.write("# dummy restraints\n")
    file.write(f"dm1: POSITION ATOM={dummy_atoms['DM1']['idx']} NOPBC\n")
    file.write(f"dm2: POSITION ATOM={dummy_atoms['DM2']['idx']} NOPBC\n")
    file.write(f"dm3: POSITION ATOM={dummy_atoms['DM3']['idx']} NOPBC\n")

    arg = "dm1.x,dm1.y,dm1.z,"
          "dm2.x,dm2.y,dm2.z,"
          "dm3.x,dm3.y,dm3.z,"

    at = (
        f"{dummy_atoms['DM1']['pos'][0]:0.3f},"
        f"{dummy_atoms['DM1']['pos'][1]:0.3f},"
        f"{dummy_atoms['DM1']['pos'][2]:0.3f},"
    )
    at += (
        f"{dummy_atoms['DM2']['pos'][0]:0.3f},"
        f"{dummy_atoms['DM2']['pos'][1]:0.3f},"
        f"{dummy_atoms['DM2']['pos'][2]:0.3f},"
    )
    at += (
        f"{dummy_atoms['DM3']['pos'][0]:0.3f},"
        f"{dummy_atoms['DM3']['pos'][1]:0.3f},"
        f"{dummy_atoms['DM3']['pos'][2]:0.3f},"
    )

    kpos *= factor
    kappa = (
        f"{kpos:0.1f},{kpos:0.1f},{kpos:0.1f},"
        f"{kpos:0.1f},{kpos:0.1f},{kpos:0.1f},"
        f"{kpos:0.1f},{kpos:0.1f},{kpos:0.1f},"
    )

    file.write(f"RESTRAINT ...\n")
    file.write(f"ARG={arg}\n")
    file.write(f"AT={at}\n")
    file.write(f"KAPPA={kappa}\n")
    file.write(f"LABEL=dummy\n")
    file.write(f"... RESTRAINT\n")