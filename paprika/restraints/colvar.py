import os
import logging

from paprika.restraints.utils import parse_window, restraint_to_colvar

logger = logging.getLogger(__name__)

PI = 3.1415926535


def colvar_module_file(file, restraints, window, legacy_k=True):
    """
    Writes a COLVAR Module file for a specific window.

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
        with open(f"windows/{window}/colvars.tcl", "w") as file:
            restraints = {
                "static": static_restraints,
                "guest": guest_restraints,
                "host": conformational_restraints,
                "wall": guest_wall_restraints,
            }
            colvar_module_file(file, restraints, window)

    """

    window, phase = parse_window(window)
    file.write("ColvarsTrajFrequency    5000\n")
    file.write("ColvarsRestartFrequency 50000\n")

    if "static" in restraints.keys():
        colvar = restraint_to_colvar(restraints["static"], phase, window, radians=False, legacy_k=legacy_k)
        write_colvar_to_file(file, colvar, "static")

    if "host" in restraints.keys():
        colvar = restraint_to_colvar(restraints["host"], phase, window, radians=False, legacy_k=legacy_k)
        write_colvar_to_file(file, colvar, "host")

    if "guest" in restraints.keys():
        colvar = restraint_to_colvar(restraints["guest"], phase, window, radians=False, legacy_k=legacy_k)
        write_colvar_to_file(file, colvar, "guest")

    if "wall" in restraints.keys():
        colvar = restraint_to_colvar(restraints["wall"], phase, window, radians=False, legacy_k=legacy_k)
        write_colvar_to_file(file, colvar, "wall")


def write_colvar_to_file(file, colvar, label, convert_kangle=True):
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
    convert_kangle : bool
        Convert angle force constant from kcal/mol/rad^2 to kcal/mol/deg^2.


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
    conversion = 1.0
    if convert_kangle:
        conversion = (PI/180)**2

    file.write(f"# {label} restraints\n")
    arg = ""
    at = ""
    kappa = ""

    # Colvar definition
    for ndx in range(colvar["ncolvar"]):
        colvar_name = f"{label[0]}_{ndx+1}"
        arg += f"{colvar_name} "
        at += f"{colvar['AT'][ndx]:0.4f} "
        colvar_def = ""

        if colvar["type"][ndx] == "DISTANCE":
            colvar_def = f"colvar {{\n" \
                         f"\tname {colvar_name}\n" \
                         f"\tdistance {{\n" \
                         f"\t\tforceNoPBC yes\n" \
                         f"\t\tgroup1 {{ atomNumbers {{ {colvar['atoms'][ndx][0]} }} }}\n" \
                         f"\t\tgroup2 {{ atomNumbers {{ {colvar['atoms'][ndx][1]} }} }}\n" \
                         f"\t}}\n}}\n"
            kappa += f"{colvar['KAPPA'][ndx] * colvar['factor']:0.5f} "

        elif colvar["type"][ndx] == "ANGLE":
            colvar_def = f"colvar {{\n" \
                         f"\tname {colvar_name}\n" \
                         f"\tangle {{\n" \
                         f"\t\tforceNoPBC yes\n" \
                         f"\t\tgroup1 {{ atomNumbers {{ {colvar['atoms'][ndx][0]} }} }}\n" \
                         f"\t\tgroup2 {{ atomNumbers {{ {colvar['atoms'][ndx][1]} }} }}\n" \
                         f"\t\tgroup3 {{ atomNumbers {{ {colvar['atoms'][ndx][2]} }} }}\n" \
                         f"\t}}\n}}\n"
            kappa += f"{colvar['KAPPA'][ndx] * colvar['factor'] * conversion:0.5f} "

        elif colvar["type"][ndx] == "TORSION":
            colvar_def = f"colvar {{\n" \
                         f"\tname {colvar_name}\n" \
                         f"\tdihedral {{\n" \
                         f"\t\tforceNOPBC yes\n" \
                         f"\t\tgroup1 {{ atomNumbers {{ {colvar['atoms'][ndx][0]} }} }}\n" \
                         f"\t\tgroup2 {{ atomNumbers {{ {colvar['atoms'][ndx][1]} }} }}\n" \
                         f"\t\tgroup3 {{ atomNumbers {{ {colvar['atoms'][ndx][2]} }} }}\n" \
                         f"\t\tgroup4 {{ atomNumbers {{ {colvar['atoms'][ndx][3]} }} }}\n" \
                         f"\t}}\n}}\n"
            kappa += f"{colvar['KAPPA'][ndx] * colvar['factor'] * conversion:0.5f} "

        file.write(colvar_def)

    # Harmonic restraints
    if label == "wall":
        bias = "harmonicWalls"
        target = "upperWalls"
        force = "upperWallConstant"
    else:
        bias = "harmonic"
        target = "centers"
        force = "forceConstant"

    harmonic = f"{bias} {{\n" \
               f"\tcolvars {arg}\n" \
               f"\t{target} {at}\n" \
               f"\t{force} {kappa}\n" \
               f"}}\n"

    file.write(harmonic)
