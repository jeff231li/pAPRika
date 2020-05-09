import json
import logging
import os
import shutil
import sys

import parmed as pmd

from paprika.io import NumpyEncoder, save_restraints
from paprika.restraints import DAT_restraint, static_DAT_restraint
from paprika.restraints.amber import amber_restraint_line
from paprika.restraints.plumed import plumed_colvar_file, write_dummy_to_plumed
from paprika.restraints.restraints import create_window_list

logger = logging.getLogger(__name__)


class HostGuestRestraints(object):
    """
    Class for building host-guest restraints with pAPRika.

    * Right now the protocol only support setting up restraints for cyclodextrins
      host-guest systems. In particular the host conformational restraints is that
      of torsion. For cucurbituril systems, a distance-based restraints is needed
      for the host.

    TODO:
        - implement a distance-based conformational restraint module
        - implement restraints that is not based on Boresch-style anchoring atoms

    Parameters
    ----------
    guest_resname : str
        Residue name of the guest molecule
    host_resname : str
        Residue name of the host molecule
    base_name : str
        Base name of the host-guest system
    anchor_atoms : dict
        Dictionary containing definition of anchor atoms {'G1': '8@C7', 'G2': '8@C1', ...}
    windows_length : list
        List of the length of windows for each phase [15, 45, 15]
    structure : `class`:`parmed.structure.Structure`
        structure of the host-guest system with dummy atoms (vacuum or solvated)

    """

    def __init__(
        self,
        guest_resname,
        host_resname,
        base_name,
        anchor_atoms,
        windows_length,
        structure,
    ):
        self._guest_resname = guest_resname
        self._host_resname = host_resname
        self._base_name = base_name
        self._anchor_atoms = anchor_atoms
        self._windows = windows_length
        self._structure = structure
        self._apr_list = {
            "attach": {"fractions": [], "windows": []},
            "pull": {"fractions": [], "windows": []},
            "release": {"fractions": [], "windows": []},
        }
        self._window_list = []
        self._static_restraints = []
        self._conformational_restraints = []
        self._guest_wall_restraints = []
        self._guest_restraints = []
        self._dummy_atoms = {"DM1": {}, "DM2": {}, "DM3": {}}
        self._continuous_apr = None
        self._auto_apr = None
        self._protocol = None

        # Attach only protocol
        if self._windows[0] != 0 and self._windows[1] == 0 and self._windows[2] == 0:
            print(">>> Setting up attach only protocol")
            self._auto_apr = False
            self._continuous_apr = False
            self._protocol = "a"
        # Pull only protocol
        elif self._windows[0] == 0 and self._windows[1] != 0 and self._windows[2] == 0:
            print(">>> Setting up pull only protocol")
            self._auto_apr = False
            self._continuous_apr = False
            self._protocol = "p"
        # Release only protocol
        elif self._windows[0] == 0 and self._windows[1] == 0 and self._windows[2] != 0:
            print(">>> Setting up release only protocol")
            self._auto_apr = False
            self._continuous_apr = False
            self._protocol = "r"
        # Attach-Pull only protocol
        elif self._windows[0] != 0 and self._windows[1] != 0 and self._windows[2] == 0:
            print(">>> Setting up attach-pull protocol")
            self._auto_apr = True
            self._continuous_apr = True
            self._protocol = "a-p"
        # Attach-Pull-Release protocol
        elif self._windows[0] != 0 and self._windows[1] != 0 and self._windows[2] != 0:
            print(">>> Setting up full attach-pull-release protocol")
            self._auto_apr = True
            self._continuous_apr = True
            self._protocol = "a-p-r"
        else:
            sys.exit("Protocol currently not supported")

    def host_static(
        self, static_atoms=None, distance_fc=5.0, angle_fc=100.0,
    ):
        """
        Method to create static Boresch restraints for the host.

        Parameters
        ----------
        static_atoms : list
            list of atoms that defines the Boresch restraints e.g. [[":D1", ":H1"], [":D2", ":D1", ":H1"], ...]
        distance_fc: float
            The force constant for distance restraints.
        angle_fc: float
            The force constant for angle restraints.

        """
        self._static_restraints = []
        if static_atoms is None:
            static_atoms = [
                [self._anchor_atoms["D1"], self._anchor_atoms["H1"]],
                [
                    self._anchor_atoms["D2"],
                    self._anchor_atoms["D1"],
                    self._anchor_atoms["H1"],
                ],
                [
                    self._anchor_atoms["D1"],
                    self._anchor_atoms["H1"],
                    self._anchor_atoms["H2"],
                ],
                [
                    self._anchor_atoms["D3"],
                    self._anchor_atoms["D2"],
                    self._anchor_atoms["D1"],
                    self._anchor_atoms["H1"],
                ],
                [
                    self._anchor_atoms["D2"],
                    self._anchor_atoms["D1"],
                    self._anchor_atoms["H1"],
                    self._anchor_atoms["H2"],
                ],
                [
                    self._anchor_atoms["D1"],
                    self._anchor_atoms["H1"],
                    self._anchor_atoms["H2"],
                    self._anchor_atoms["H3"],
                ],
            ]

        for _, atoms in enumerate(static_atoms):
            this = static_DAT_restraint(
                restraint_mask_list=atoms,
                num_window_list=self._windows,
                ref_structure=self._structure,
                force_constant=angle_fc if len(atoms) > 2 else distance_fc,
                amber_index=True,
            )

            self._static_restraints.append(this)

        print(f"\t* There are {len(self._static_restraints)} static restraints")

    def host_conformation(
        self, template, targets, torsion_fc=6.0,
    ):
        """
        Method to create conformational restraints based on torsion for the host. This method
        assumes that the atom namings are periodic across the cyclodextrin monomers.

        Parameters
        ----------
        template: list
            List of torsions defined by atom names, e.g. [["O5", "C1", "O1", "C4"], ["C1", "O1", "C4", "C5"]].
        targets: list
            A list of equilibrium angles corresponding to the torsion defined in the ``template``.
        torsion_fc: float
            The force constant for the torsional restraints (kcal/mol/rad^2)

        """
        self._conformational_restraints = []
        host_residues = len(
            self._structure[":{}".format(self._host_resname.upper())].residues
        )
        first_host_residue = (
            self._structure[":{}".format(self._host_resname.upper())].residues[0].number
            + 1
        )

        for n in range(first_host_residue, host_residues + first_host_residue):
            if n + 1 < host_residues + first_host_residue:
                next_residue = n + 1
            else:
                next_residue = first_host_residue

            for (index, atoms), target in zip(enumerate(template), targets):
                conformational_restraint_atoms = []
                if index == 0:
                    conformational_restraint_atoms.append(f":{n}@{atoms[0]}")
                    conformational_restraint_atoms.append(f":{n}@{atoms[1]}")
                    conformational_restraint_atoms.append(f":{n}@{atoms[2]}")
                    conformational_restraint_atoms.append(f":{next_residue}@{atoms[3]}")
                else:
                    conformational_restraint_atoms.append(f":{n}@{atoms[0]}")
                    conformational_restraint_atoms.append(f":{n}@{atoms[1]}")
                    conformational_restraint_atoms.append(f":{next_residue}@{atoms[2]}")
                    conformational_restraint_atoms.append(f":{next_residue}@{atoms[3]}")

                this = DAT_restraint()
                this.auto_apr = self._auto_apr
                this.continuous_apr = self._continuous_apr
                this.amber_index = True
                this.topology = self._structure

                this.mask1 = conformational_restraint_atoms[0]
                this.mask2 = conformational_restraint_atoms[1]
                this.mask3 = conformational_restraint_atoms[2]
                this.mask4 = conformational_restraint_atoms[3]

                if (
                    self._protocol == "a"
                    or self._protocol == "a-p"
                    or self._protocol == "a-p-r"
                ):
                    this.attach["fraction_list"] = self._apr_list["attach"]["fractions"]
                    this.attach["target"] = target
                    this.attach["fc_final"] = torsion_fc

                if (
                    self._protocol == "p"
                    or self._protocol == "a-p"
                    or self._protocol == "a-p-r"
                ):
                    this.pull["target_final"] = target
                    this.pull["num_windows"] = self._windows[1]

                if self._protocol == "r" or self._protocol == "a-p-r":
                    this.release["fraction_list"] = self._apr_list["release"]["fractions"]
                    this.release["target"] = target
                    this.release["fc_final"] = torsion_fc

                this.initialize()

                self._conformational_restraints.append(this)

        print(
            f"\t* There are {len(self._conformational_restraints)} conformational restraints"
        )

    def host_conformation(
        self, template,
    ):
        """
        Method to create conformational restraints based on torsion for the host.

        Parameters
        ----------
        template: dict
            List of torsions for host conformational restraints.

            template = {
                'atoms': [[":1@O5", "1@C1", "1@O1", "1@C4"], ["1@C1", "1@O1", "1@C4", "1@C5"], ...],
                'target': [120.0, -115.0, ...],
                'k': [6.0, 6.0, ...]
            }

        """
        self._conformational_restraints = []

        for atoms, target, force_constant in zip(
            template["atoms"], template["target"], template["k"]
        ):
            this = DAT_restraint()
            this.auto_apr = self._auto_apr
            this.continuous_apr = self._continuous_apr
            this.amber_index = True
            this.topology = self._structure

            this.mask1 = atoms[0]
            this.mask2 = atoms[1]
            this.mask3 = atoms[2]
            this.mask4 = atoms[3]

            if (
                self._protocol == "a"
                or self._protocol == "a-p"
                or self._protocol == "a-p-r"
            ):
                this.attach["fraction_list"] = self._apr_list["attach"]["fractions"]
                this.attach["target"] = target
                this.attach["fc_final"] = force_constant

            if (
                self._protocol == "a-p"
                or self._protocol == "p"
                or self._protocol == "a-p-r"
            ):
                this.pull["target_final"] = target
                this.pull["num_windows"] = self._windows[1]

            if self._protocol == "r" or self._protocol == "a-p-r":
                this.release["fraction_list"] = self._apr_list["release"]["fractions"]
                this.release["target"] = target
                this.release["fc_final"] = force_constant

            this.initialize()

            self._conformational_restraints.append(this)

        print(
            f"\t* There are {len(self._conformational_restraints)} conformational restraints"
        )

    def guest_wall(
        self, template, targets, distance_fc=50.0, angle_fc=500.0,
    ):
        """
        Method to setup wall restraints on the guest molecule so that weakly bound guest will
        not unbind from the host during the attach phase.

        Parameters
        ----------
        template: list
            List of distances/angles defined by atom names, e.g. [["O2", "G1"], ["D2", "G1", "G2"]].
        targets: list
            A list of equilibrium values corresponding to the distances/angles defined in the ``template``.
        distance_fc: float
            The force constant for distance restraints (kcal/mol/A^2).
        angle_fc: float
            The force constant for angle restraints (kcal/mol/rad^2).

        """
        self._guest_wall_restraints = []
        host_residues = len(
            self._structure[":{}".format(self._host_resname.upper())].residues
        )
        first_host_residue = (
            self._structure[":{}".format(self._host_resname.upper())].residues[0].number
            + 1
        )

        for n in range(first_host_residue, host_residues + first_host_residue):
            for (index, atoms), target in zip(enumerate(template[:-1]), targets[:-1]):
                guest_wall_restraint_atoms = [f":{n}@{atoms[0]}", f"{atoms[1]}"]

                this = DAT_restraint()
                this.auto_apr = self._auto_apr
                this.continuous_apr = self._continuous_apr
                this.amber_index = True
                this.topology = self._structure
                this.mask1 = guest_wall_restraint_atoms[0]
                this.mask2 = guest_wall_restraint_atoms[1]
                this.attach["fc_initial"] = distance_fc
                this.attach["fc_final"] = distance_fc
                this.custom_restraint_values["rk2"] = distance_fc
                this.custom_restraint_values["rk3"] = distance_fc
                this.custom_restraint_values["r1"] = 0.0
                this.custom_restraint_values["r2"] = 0.0

                this.attach["target"] = target
                this.attach["num_windows"] = self._windows[0]

                this.initialize()
                self._guest_wall_restraints.append(this)

        # Add a single angle restraint!
        guest_wall_restraint_atoms = [
            f"{template[-1][0]}",
            f"{template[-1][1]}",
            f"{template[-1][2]}",
        ]
        target = targets[-1]

        this = DAT_restraint()
        this.auto_apr = self._auto_apr
        this.continuous_apr = self._continuous_apr
        this.amber_index = True
        this.topology = self._structure
        this.mask1 = guest_wall_restraint_atoms[0]
        this.mask2 = guest_wall_restraint_atoms[1]
        this.mask3 = guest_wall_restraint_atoms[2]
        this.attach["fc_initial"] = angle_fc
        this.attach["fc_final"] = angle_fc
        this.custom_restraint_values["rk2"] = angle_fc
        this.custom_restraint_values["rk3"] = 0.0

        this.attach["target"] = target
        this.attach["num_windows"] = self._windows[0]

        this.initialize()
        self._guest_wall_restraints.append(this)

        print(f"\t* There are {len(self._guest_wall_restraints)} guest wall restraints")

    def guest_wall(
        self, template,
    ):
        """
        Method to setup wall restraints on the guest molecule so that weakly bound guest will
        not unbind from the host during the attach phase.

        Parameters
        ----------
        template: dict
            List of distance/angle definition for wall restraints

            template = {
                'atoms': [[":1@O2", ":8@C1"], [":DM2", ":8@C1", ":8@C7"], ...],
                'target': [12.5, 180.0, ...],
                'k': [50.0, 100.0, ...]
            }

        """
        self._guest_wall_restraints = []

        for atoms, target, force_constant in zip(
            template["atoms"], template["target"], template["k"]
        ):
            this = DAT_restraint()
            this.auto_apr = self._auto_apr
            this.continuous_apr = self._continuous_apr
            this.amber_index = True
            this.topology = self._structure

            this.mask1 = atoms[0]
            this.mask2 = atoms[1]

            this.attach["fc_initial"] = force_constant
            this.attach["fc_final"] = force_constant

            # Bonds
            if len(atoms) == 2:
                this.custom_restraint_values["rk2"] = 50.0
                this.custom_restraint_values["rk3"] = 50.0
                this.custom_restraint_values["r1"] = 0.0
                this.custom_restraint_values["r2"] = 0.0

            # Angles
            elif len(atoms) == 3:
                this.mask3 = atoms[2]

                this.custom_restraint_values["rk2"] = 500.0
                this.custom_restraint_values["rk3"] = 0.0

            this.attach["target"] = target
            this.attach["num_windows"] = self._windows[0]

            this.initialize()
            self._guest_wall_restraints.append(this)

        print(f"\t* There are {len(self._guest_wall_restraints)} guest wall restraints")

    def guest(
        self, guest_atoms=None, guest_targets=None, distance_fc=5.0, angle_fc=100.0,
    ):
        """
        Construct DAT_restraint() objects for the guest restraints

        Parameters
        ----------
        guest_atoms : list
            List of anchoring atoms that defines the guest restraints [[D1, G1], [D2, D1, G1], [D1, G1, G2]]
        guest_targets : dict
            List of equilibrium targets for the corresponding restraints at the initial and final stages
            {'initial': [6.0, 180.0, 180.0], 'final': [24.0, 180.0, 180.0]}
        distance_fc : float
            The force constant for distance restraints (kcal/mol/A^2).
        angle_fc : float
            The force constant for angle restraints (kcal/mol/rad^2).

        """

        self._guest_restraints = []
        if guest_atoms is None:
            guest_atoms = [
                [self._anchor_atoms["D1"], self._anchor_atoms["G1"]],
                [
                    self._anchor_atoms["D2"],
                    self._anchor_atoms["D1"],
                    self._anchor_atoms["G1"],
                ],
                [
                    self._anchor_atoms["D1"],
                    self._anchor_atoms["G1"],
                    self._anchor_atoms["G2"],
                ],
            ]
        if guest_targets is None:
            guest_targets = {
                "initial": [self._apr_list["pull"]["fractions"][0], 180.0, 180.0],
                "final": [self._apr_list["pull"]["fractions"][-1], 180.0, 180.0],
            }

        for index, atoms in enumerate(guest_atoms):
            if len(atoms) > 2:
                angle = True
            else:
                angle = False
            this = DAT_restraint()
            this.auto_apr = self._auto_apr
            this.continuous_apr = self._continuous_apr
            this.amber_index = True
            this.topology = self._structure
            this.mask1 = atoms[0]
            this.mask2 = atoms[1]
            if angle:
                this.mask3 = atoms[2]
                this.attach["fc_final"] = angle_fc
                this.release["fc_final"] = angle_fc
            else:
                this.attach["fc_final"] = distance_fc
                this.release["fc_final"] = distance_fc

            if (
                self._protocol == "a"
                or self._protocol == "a-p"
                or self._protocol == "a-p-r"
            ):
                this.attach["target"] = guest_targets["initial"][index]
                this.attach["fraction_list"] = self._apr_list["attach"]["fractions"]

            if (
                self._protocol == "p"
                or self._protocol == "a-p"
                or self._protocol == "a-p-r"
            ):
                this.pull["target_final"] = guest_targets["final"][index]
                this.pull["num_windows"] = self._windows[1]

            if self._protocol == "r" or self._protocol == "a-p-r":
                this.release["target"] = guest_targets["final"][index]
                # keep the guest restraints on during release.
                this.release["fraction_list"] = [1.0] * self._windows[2]

            this.initialize()
            self._guest_restraints.append(this)

        print(f"\t* There are {len(self._guest_restraints)} guest restraints")

    def _write_apr_windows(self):
        """
        Extract APR windows and write to a json file
        """

        for window in self._window_list:
            if window[0] == "a":
                self._apr_list["attach"]["windows"].append(window)
            if window[0] == "p":
                self._apr_list["pull"]["windows"].append(window)
            if window[0] == "r":
                self._apr_list["release"]["windows"].append(window)

        with open("apr_windows.json", "w") as f:
            dumped = json.dumps(self._apr_list, cls=NumpyEncoder)
            f.write(dumped)

    def set_APR_windows(
        self, a=None, p=None, r=None
    ):
        """
        Sets the APR windows/fractions

        Parameters
        ----------
        a : list
            List of fractions for the 'attach' phase
        p : list
            List of windows for the 'pull' phase
        r : list
            List of fractions for the 'release' phase

        """
        self._apr_list["attach"]["fractions"] = a
        self._apr_list["pull"]["fractions"] = p
        self._apr_list["release"]["fractions"] = r

    def get_window_list(self):
        return self._window_list

    def get_last_pull_window(self):
        return self._apr_list["pull"]["windows"][-1]

    def _calc_target_diff(self, window):
        return (
            self._guest_restraints[0].phase["pull"]["targets"][int(window)]
            - self._guest_restraints[0].pull["target_initial"]
        )

    def _create_folders(self, clean=True):
        """
        Create APR folders as defined by the fractions

        Parameters
        ----------
        clean : bool
            delete current 'windows' folder?

        """
        # Remove folders
        if clean:
            if os.path.exists(f"windows"):
                shutil.rmtree("windows")
            os.makedirs("windows")

        # Create window folders
        for window in self._window_list:
            folder = os.path.join("windows", f"{window}")
            if not os.path.exists(folder):
                os.makedirs(folder)

    def _fetch_dummy_atoms(self, serial=True):
        """
        Extract information about dummy atoms based on the structure given in the constructor

        Parameters
        ---------
        serial : bool
            Use serial (stats a 1) or index (starts at 0) atomic indices

        """

        from paprika.utils import extract_dummy_atoms

        self._dummy_atoms = extract_dummy_atoms(self._structure, serial=serial)

    def translate_guest_molecule(self):
        """
        Translate the guest molecule as defined by the pull windows
        """
        pull_window = os.path.join("windows", self.get_last_pull_window())

        for window in self._window_list:
            sub_folder = os.path.join("windows", window)

            if window[0] == "a":
                shutil.copy(
                    self._structure.name,
                    os.path.join(sub_folder, f"{self._base_name}.prmtop"),
                )
                shutil.copy(
                    self._structure.name.replace("prmtop", "rst7"),
                    os.path.join(sub_folder, f"{self._base_name}.rst7"),
                )

            elif window[0] == "p":
                structure = pmd.load_file(
                    self._structure.name, self._structure.name.replace("prmtop", "rst7")
                )

                target_difference = self._calc_target_diff(window[1:])
                print(
                    f"In window {window} we will translate the guest {target_difference:0.1f} Angstroms."
                )

                for atom in structure.atoms:
                    if atom.residue.name == self._guest_resname:
                        atom.xz += target_difference

                structure.save(
                    os.path.join(sub_folder, f"{self._base_name}.prmtop"),
                    overwrite=True,
                )
                structure.save(
                    os.path.join(sub_folder, f"{self._base_name}.rst7"),
                    overwrite=True,
                )

            elif window[0] == "r":
                shutil.copy(
                    os.path.join(pull_window, f"{self._base_name}.prmtop"),
                    os.path.join(sub_folder, f"{self._base_name}.prmtop"),
                )
                shutil.copy(
                    os.path.join(pull_window, f"{self._base_name}.rst7"),
                    os.path.join(sub_folder, f"{self._base_name}.rst7"),
                )

    def dump_restraint_files(
        self,
        json_file="restraints.json",
        restr_type="amber",
        ref_from_structure=False,
        clean=False,
    ):
        """
        Method to write pAPRika restraints to either Amber NMR-style or Plumed format

        Parameters
        ----------
        json_file : str
            filename for pAPRika restraint file (.json)
        restr_type : str
            restraint format - Amber NMR or Plumed
        ref_from_structure : bool
            take the reference position of dummy atoms from current structure
        clean : bool
            Delete the current folder?

        """
        # dump restraint file for later use in analysis
        restraint_list = self._static_restraints + self._conformational_restraints

        if (
            self._protocol == "a"
            or self._protocol == "p"
            or self._protocol == "a-p"
            or self._protocol == "a-p-r"
        ):
            restraint_list += self._guest_restraints

        save_restraints(restraint_list=restraint_list, filepath=json_file)

        # Create folders and write to json files
        self._window_list = create_window_list(self._conformational_restraints)
        self._create_folders(clean)
        self._write_apr_windows()

        # Write AMBER NMR restraint file
        if restr_type.lower() == "amber":
            for window in self._window_list:
                with open(os.path.join("windows", window, "disang.rest"), "w") as file:
                    if window[0] == "a":
                        restraints = (
                            self._static_restraints
                            + self._guest_restraints
                            + self._conformational_restraints
                            + self._guest_wall_restraints
                        )
                    if window[0] == "p":
                        restraints = (
                            self._static_restraints
                            + self._guest_restraints
                            + self._conformational_restraints
                        )
                    if window[0] == "r" and self._protocol == "a-p-r":
                        restraints = (
                            self._static_restraints
                            + self._guest_restraints
                            + self._conformational_restraints
                        )
                    if window[0] == "r" and self._protocol == "r":
                        restraints = (
                            self._static_restraints + self._conformational_restraints
                        )

                    for restraint in restraints:
                        string = amber_restraint_line(restraint, window)
                        if string is not None:
                            file.write(string)

        # Write Plumed colvar file
        elif restr_type.lower() == "plumed":
            for window in self._window_list:
                with open(os.path.join("windows", window, "plumed.dat"), "w") as file:
                    if window[0] == "a":
                        restraints = {
                            "static": self._static_restraints,
                            "guest": self._guest_restraints,
                            "host": self._conformational_restraints,
                            "wall": self._guest_wall_restraints,
                        }
                    if window[0] == "p":
                        restraints = {
                            "static": self._static_restraints,
                            "guest": self._guest_restraints,
                            "host": self._conformational_restraints,
                        }
                    if window[0] == "r" and self._protocol == "a-p-r":
                        restraints = {
                            "static": self._static_restraints,
                            "guest": self._guest_restraints,
                            "host": self._conformational_restraints,
                        }
                    if window[0] == "r" and self._protocol == "r":
                        restraints = {
                            "static": self._static_restraints,
                            "host": self._conformational_restraints,
                        }

                    plumed_colvar_file(file, restraints, window)

                    if ref_from_structure:
                        self._fetch_dummy_atoms(serial=True)
                        write_dummy_to_plumed(file, self._dummy_atoms)
