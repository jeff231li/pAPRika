import os
import logging

from paprika.restraints.utils import parse_window, restraint_to_colvar

logger = logging.getLogger(__name__)

PI = 3.14159265359


def plumed_colvar_file(file, restraints, window, legacy_k=True):
    """
    Writes a PLUMED colvar file for a specific window.

    Parameters
    ----------
    file: class '_io.TextIOWrapper'
        The file object handle to save the plumed file.
    restraints: dict
        The pAPRika restraint to be used.
    window: str
        The calculation window that will be used to index the restraint values.
    legacy_k : bool
        Are the restraints based on legacy force constants? Old MD codes
        like AMBER and CHARMM requires the user to half the force constant
        beforehand. New MD codes like GROMACS and NAMD requires the user
        to set the force constant without the 1/2 factor.


    Examples
    --------
    Below is an example of converting restraints defined by DAT_restraint()
    to a Plumed colvar format.

    for window in window_list:
        with open(f"windows/{window}/plumed.dat", "w") as file:
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
        colvar = restraint_to_colvar(restraints["static"], phase, window, radians=True, legacy_k=legacy_k)
        write_colvar_to_plumed(file, colvar, "static")

    if "host" in restraints.keys():
        colvar = restraint_to_colvar(restraints["host"], phase, window, radians=True, legacy_k=legacy_k)
        write_colvar_to_plumed(file, colvar, "host")

    if "guest" in restraints.keys():
        colvar = restraint_to_colvar(restraints["guest"], phase, window, radians=True, legacy_k=legacy_k)
        write_colvar_to_plumed(file, colvar, "guest")

    if "wall" in restraints.keys():
        colvar = restraint_to_colvar(restraints["wall"], phase, window, radians=True, legacy_k=legacy_k)
        write_colvar_to_plumed(file, colvar, "wall")


def write_colvar_to_plumed(file, colvar, label):
    """
    Write collective variable and restraints to file

    Parameters
    ----------
    file : class '_io.TextIOWrapper'
        The file object handle to save the plumed file.
    colvar : dict
        Dictionary containing information about the collective variable.
    label : str
        Restraint type for naming purposes.


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
    file.write(f"# {label} restraints\n")
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

        file.write(f"{label[0]}_{ndx + 1}: {colvar['type'][ndx]} ATOMS={atoms} NOPBC\n")
        arg += f"{label[0]}_{ndx + 1},"
        at += f"{colvar['AT'][ndx]:0.4f},"
        kappa += f"{colvar['KAPPA'][ndx] * colvar['factor']:0.2f},"

    if label == "wall":
        bias = "UPPER_WALLS"
    else:
        bias = "RESTRAINT"

    file.write(f"{bias} ...\n")
    file.write(f"ARG={arg}\n")
    file.write(f"AT={at}\n")
    file.write(f"KAPPA={kappa}\n")
    file.write(f"LABEL={label}\n")
    file.write(f"... {bias}\n")


def write_dummy_to_plumed(file, dummy_atoms, kpos=100.0):
    """
    Append to the plumed.dat file the dummy atoms colvar and restraints

    Parameters
    ----------
    file : class '_io.TextIOWrapper'
        The file object handle to save the plumed file.
    dummy_atoms : dict
        Dictionary containing information about the dummy atoms.
    kpos : float
        Spring constant used to restrain dummy atoms (kcal/mol/A^2).


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
    file.write("# dummy restraints\n")
    file.write(f"dm1: POSITION ATOM={dummy_atoms['DM1']['idx']} NOPBC\n")
    file.write(f"dm2: POSITION ATOM={dummy_atoms['DM2']['idx']} NOPBC\n")
    file.write(f"dm3: POSITION ATOM={dummy_atoms['DM3']['idx']} NOPBC\n")

    arg = "dm1.x,dm1.y,dm1.z," "dm2.x,dm2.y,dm2.z," "dm3.x,dm3.y,dm3.z,"

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


def add_dummy_to_plumed(structure, file_path='./', plumed='plumed.dat'):
    # Extract dummy atoms
    from paprika.utils import extract_dummy_atoms

    dummy_atoms = extract_dummy_atoms(structure, serial=True)

    # Write dummy atom info to 'plumed.dat'
    plumed_file = os.path.join(file_path, plumed)

    if os.path.isfile(plumed_file):
        with open(plumed_file, "a") as file:
            write_dummy_to_plumed(file, dummy_atoms)
    else:
        raise Exception(
            f"ERROR: '{plumed}' file does not exists, please check your setup script"
        )
