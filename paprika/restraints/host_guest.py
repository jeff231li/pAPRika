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
from paprika.restraints.utils import parse_restraints

logger = logging.getLogger(__name__)


class HostGuestRestraints(object):
    """
    Class for building host-guest restraints with pAPRika.

    * Right now the protocol only support setting up restraints for cyclodextrins
      host-guest systems. In particular the host conformational restraints is that
      of torsion. For cucurbituril systems, a distance-based restraints is needed
      for the host.

    TODO:
        - implement an automatic distance-based conformational restraint module
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
    structure : `class`:`parmed.structure.Structure`
        structure of the host-guest system with dummy atoms (vacuum or solvated)
    windows : list
        List of the windows for each phase [[0.0, 0.01, 0.2], [6.0, 6.4, 6.8], [1.0, 0.8 0.0]]

    """

    def __init__(
        self,
        guest_resname,
        host_resname,
        base_name,
        anchor_atoms,
        structure,
        windows=None,
        output_path='.',
        output_prefix='windows',
        engine='amber',
    ):
        self._guest_resname = guest_resname
        self._host_resname = host_resname
        self._base_name = base_name
        self._anchor_atoms = anchor_atoms
        self._windows = None
        self._structure = structure
        self._output_path = output_path
        self._output_prefix = output_prefix
        self._engine = engine
        self._apr_list = {
            "attach": {"fractions": [], "windows": []},
            "pull": {"fractions": [], "windows": []},
            "release": {"fractions": [], "windows": []},
        }
        self._window_list = []
        self._static_restraints = []
        self._conformational_restraints = []
        self._wall_restraints = []
        self._symmetry_restraints = []
        self._guest_restraints = []
        self._dummy_atoms = {"DM1": {}, "DM2": {}, "DM3": {}}
        self._continuous_apr = None
        self._auto_apr = None
        self._protocol = None

        self._cv_host_rmsd = None
        self._cv_guest_rmsd = None

        if windows is not None:
            self.set_APR_windows(a=windows[0], p=windows[1], r=windows[2])

    def _init_protocol(self):
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
                amber_index=False if self._engine == "openmm" else True,
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
                this.amber_index = False if self._engine == "openmm" else True
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
            this.amber_index = False if self._engine == "openmm" else True
            this.topology = self._structure

            # Distance
            this.mask1 = atoms[0]
            this.mask2 = atoms[1]

            # Angle
            if len(atoms) == 3:
                this.mask3 = atoms[2]

            # Torsion
            if len(atoms) == 4:
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

    def host_rmsd(self, atom_types=None, structure=None, target=0.0, krmsd=10.0, scale=True):
        self._cv_host_rmsd = {
            'cv_ni': 0, 'cv_nr': 0, 'cv_i': [], 'cv_r': [], 
            'target': target, 'k': krmsd, 'scale': scale
        }

        if structure is None:
            structure = self._structure

        for atom in structure.atoms:
            if atom_types is not None:
                if atom.residue.name == self._host_resname and atom.type in atom_types:
                    self._cv_host_rmsd['cv_i'].append(atom.idx + 1)
                    self._cv_host_rmsd['cv_r'].append([atom.xx, atom.xy, atom.xz])
            elif atom_types is None:
                if atom.residue.name == self._host_resname and atom.element != 1:
                    self._cv_host_rmsd['cv_i'].append(atom.idx + 1)
                    self._cv_host_rmsd['cv_r'].append([atom.xx, atom.xy, atom.xz])

        self._cv_host_rmsd['cv_ni'] = len(self._cv_host_rmsd['cv_i'])+1
        self._cv_host_rmsd['cv_nr'] = len(self._cv_host_rmsd['cv_r'])*3

        print(f"\t* There are {len(self._cv_host_rmsd['cv_i'])} atoms for the host-RMSD colvar")

    def guest_rmsd(self, atom_types=None, structure=None, target=0.0, krmsd=10.0, scale=True):
        self._cv_guest_rmsd = {
            'cv_ni': 0, 'cv_nr': 0, 'cv_i': [], 'cv_r': [], 
            'target': target, 'k': krmsd, 'scale': scale
        }

        if structure is None:
            structure = self._structure

        for atom in structure.atoms:
            if atom_types is not None:
                if atom.residue.name == self._guest_resname and atom.type in atom_types:
                    self._cv_guest_rmsd['cv_i'].append(atom.idx + 1)
                    self._cv_guest_rmsd['cv_r'].append([atom.xx, atom.xy, atom.xz])
            elif atom_types is None:
                if atom.residue.name == self._guest_resname and atom.element != 1:
                    self._cv_guest_rmsd['cv_i'].append(atom.idx + 1)
                    self._cv_guest_rmsd['cv_r'].append([atom.xx, atom.xy, atom.xz])

        self._cv_guest_rmsd['cv_ni'] = len(self._cv_guest_rmsd['cv_i'])+1
        self._cv_guest_rmsd['cv_nr'] = len(self._cv_guest_rmsd['cv_r'])*3

        print(f"\t* There are {len(self._cv_guest_rmsd['cv_i'])} atoms for the guest-RMSD colvar")

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
        self._wall_restraints = []
        self._symmetry_restraints = []
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
                this.amber_index = False if self._engine == "openmm" else True
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
                self._wall_restraints.append(this)

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
        this.amber_index = False if self._engine == "openmm" else True
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
        self._symmetry_restraints.append(this)

        print(f"\t* There are {len(self._wall_restraints)} guest wall restraints")

    def guest_wall(
        self, template,
    ):
        """
        Method to setup wall restraints (half-harmonic) on the guest molecule so that
        weakly bound guest will not unbind from the host during the attach phase.

        Currently only supports distance and angle based restraints

        Parameters
        ----------
        template: dict
            Dictionary of wall restraints with information about atoms, target
            and force constant

            template = {
                'atoms': [[":1@O2", ":8@C1"], [":DM2", ":8@C1", ":8@C7"], ...],
                'target': [12.5, 180.0, ...],
                'k': [50.0, 100.0, ...]
            }

        """
        self._wall_restraints = []
        self._symmetry_restraints = []

        for atoms, target, force_constant, dat_type in zip(
            template["atoms"], template["target"], template["k"], template["type"], 
        ):
            this = DAT_restraint()
            this.auto_apr = self._auto_apr
            this.continuous_apr = self._continuous_apr
            this.amber_index = False if self._engine == "openmm" else True
            this.topology = self._structure

            this.mask1 = atoms[0]
            this.mask2 = atoms[1]

            this.attach["fc_initial"] = force_constant
            this.attach["fc_final"] = force_constant

            this.custom_restraint_values["rk2"] = force_constant

            # Distance
            if dat_type == "bond":
                this.custom_restraint_values["rk3"] = force_constant
                this.custom_restraint_values["r1"] = 0.0
                this.custom_restraint_values["r2"] = 0.0

            # Angles
            elif dat_type == "angle":
                this.mask3 = atoms[2]
                this.custom_restraint_values["rk3"] = 0.0

            this.attach["target"] = target
            this.attach["num_windows"] = self._windows[0]

            this.initialize()

            if dat_type == "bond":
                self._wall_restraints.append(this)
            elif dat_type == "angle":
                self._symmetry_restraints.append(this)

        print(f"\t* There are {len(self._wall_restraints)} guest wall restraints")
        print(f"\t* There are {len(self._symmetry_restraints)} guest symmetry restraints")

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
        if guest_targets is None and self._protocol != "a":
            guest_targets = {
                "initial": [self._apr_list["pull"]["fractions"][0], 180.0, 180.0],
                "final": [self._apr_list["pull"]["fractions"][-1], 180.0, 180.0],
            }
        elif guest_targets is None:
            guest_targets = {
                "initial": [6.0, 180.0, 180.0],
                "final": [6.0, 180.0, 180.0],
            }

        for index, atoms in enumerate(guest_atoms):
            this = DAT_restraint()
            this.auto_apr = self._auto_apr
            this.continuous_apr = self._continuous_apr
            this.amber_index = False if self._engine == "openmm" else True
            this.topology = self._structure

            # Atoms
            this.mask1 = atoms[0]
            this.mask2 = atoms[1]
            if len(atoms) == 3:
                this.mask3 = atoms[2]
            if len(atoms) == 4:
                this.mask3 = atoms[2]
                this.mask4 = atoms[3]

            # APR protocol
            if (
                self._protocol == "a"
                or self._protocol == "a-p"
                or self._protocol == "a-p-r"
            ):
                this.attach["fraction_list"] = self._apr_list["attach"]["fractions"]
                this.attach["target"] = guest_targets["initial"][index]
                this.attach["fc_final"] = distance_fc if len(atoms) == 2 else angle_fc

            if (
                self._protocol == "p"
                or self._protocol == "a-p"
                or self._protocol == "a-p-r"
            ):
                this.pull["target_final"] = guest_targets["final"][index]
                this.pull["num_windows"] = self._windows[1]

            if self._protocol == "r" or self._protocol == "a-p-r":
                # keep the guest restraints on during release.
                this.release["fraction_list"] = [1.0] * self._windows[2]
                this.release["target"] = guest_targets["final"][index]
                this.release["fc_final"] = distance_fc if len(atoms) == 2 else angle_fc

            this.initialize()
            self._guest_restraints.append(this)

        print(f"\t* There are {len(self._guest_restraints)} guest restraints")

    def set_APR_windows(self, a=None, p=None, r=None):
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

        # Initialize APR protocol
        self._windows = [
            0 if a is None else len(a),
            0 if p is None else len(p),
            0 if r is None else len(r),
        ]
        self._init_protocol()

    def dump_windows_list(self, output="windows.json"):
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

        with open(os.path.join(self._output_path, output), "w") as f:
            dumped = json.dumps(self._apr_list, cls=NumpyEncoder)
            f.write(dumped)

    def get_window_list(self):
        return self._window_list

    def get_last_pull_window(self):
        return self._apr_list["pull"]["windows"][-1]

    def _calc_target_diff(self, window):

        return (
            self._guest_restraints[0].phase["pull"]["targets"][int(window)]
            - self._guest_restraints[0].pull["target_initial"]
        )

    def create_folders(self, clean=True):
        """
        Create APR folders as defined by the fractions

        Parameters
        ----------
        clean : bool
            delete current 'windows' folder

        """
        # Generate list and create APR windows
        if not self._window_list:
            if self._conformational_restraints and self._guest_restraints or \
                    self._conformational_restraints and not self._guest_restraints:
                self._window_list = create_window_list(self._conformational_restraints)

            elif self._guest_restraints and not self._conformational_restraints:
                self._window_list = create_window_list(self._guest_restraints)

            elif not self._guest_restraints and not self._conformational_restraints and self._protocol == 'r':
                self._window_list = []
                for i in range(len(self._apr_list["release"]["fractions"])):
                    self._window_list.append(f'r{i:03d}')

            else:
                raise Exception('No APR windows will be created because both guest '
                                'and host conformational restraints are not defined')

        # Path to folder
        windows_folder = os.path.join(self._output_path, self._output_prefix)

        # Remove folders
        if clean:
            if os.path.exists(windows_folder):
                shutil.rmtree(windows_folder)
            os.makedirs(windows_folder)

        # Create window folders
        for window in self._window_list:
            sub_folder = os.path.join(windows_folder, f"{window}")
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)

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

    def dump_structures(self, add_dummy_to_plumed=False):
        """
        Method to copy structure to each folder ready for solvation with tleap. The
        structures for the pull phase will have the guest molecule translated according to
        the specified pull distances.

        Parameters
        ----------
        add_dummy_to_plumed : bool
            Append plumed.dat file with dummy atom restraints?

        """

        for window in self._window_list:
            sub_folder = os.path.join(self._output_path, self._output_prefix, window)

            if window[0] == "a":
                shutil.copy(
                    self._structure.name,
                    os.path.join(sub_folder, f"{self._base_name}.prmtop"),
                )
                shutil.copy(
                    self._structure.name.replace("prmtop", "rst7"),
                    os.path.join(sub_folder, f"{self._base_name}.rst7"),
                )
                shutil.copy(
                    self._structure.name.replace("prmtop", "pdb"),
                    os.path.join(sub_folder, f"{self._base_name}.pdb"),
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
                    os.path.join(sub_folder, f"{self._base_name}.prmtop"), overwrite=True,
                )
                structure.save(
                    os.path.join(sub_folder, f"{self._base_name}.rst7"), overwrite=True,
                )
                structure.save(
                    os.path.join(sub_folder, f"{self._base_name}.pdb"), overwrite=True,
                )

            elif window[0] == "r":
                if self._protocol == "r":
                    prmtop = self._structure.name
                    inpcrd = self._structure.name.replace("prmtop", "rst7")
                    inppdb = self._structure.name.replace("prmtop", "pdb")

                else:
                    pull_window = os.path.join(self._output_path, self._output_prefix, self.get_last_pull_window())
                    prmtop = os.path.join(pull_window, f"{self._base_name}.prmtop")
                    inpcrd = os.path.join(pull_window, f"{self._base_name}.rst7")
                    inppdb = os.path.join(pull_window, f"{self._base_name}.pdb")

                shutil.copy(
                    prmtop,
                    os.path.join(sub_folder, f"{self._base_name}.prmtop"),
                )
                shutil.copy(
                    inpcrd,
                    os.path.join(sub_folder, f"{self._base_name}.rst7"),
                )
                shutil.copy(
                    inppdb,
                    os.path.join(sub_folder, f"{self._base_name}.pdb"),
                )

            # Add dummy atom restraints to plumed.dat
            if add_dummy_to_plumed:
                structure = pmd.load_file(
                        os.path.join(sub_folder, f"{self._base_name}.prmtop"),
                        os.path.join(sub_folder, f"{self._base_name}.rst7")
                )

                # Append to file
                from paprika.restraints.plumed import add_dummy_to_plumed

                add_dummy_to_plumed(structure, file_path=sub_folder, plumed='plumed.dat')

    def dump_openmm_xml(self, nbMethod='PME', cutoff=9.0, output_xml="system.xml"):
        from paprika.setup import apply_openmm_restraints
        from simtk import unit
        from simtk.openmm import CustomExternalForce, CustomCVForce, RMSDForce, XmlSerializer
        from simtk.openmm.app import AmberInpcrdFile, AmberPrmtopFile, HBonds, NoCutoff, PME

        if nbMethod == 'PME':
            nonbondedMethod = PME
        elif nbMethod == "NoCutoff":
            nonbondedMethod = NoCutoff

        for window in self._window_list:
            sub_folder = os.path.join(self._output_path, self._output_prefix, window)
            if window[0] == 'a':
                phase = 'attach'
            elif window[0] == 'r':
                phase = 'release'

            prmtop = AmberPrmtopFile(os.path.join(sub_folder, f"{self._base_name}.prmtop"))

            system = prmtop.createSystem(
                nonbondedMethod=nonbondedMethod,
                nonbondedCutoff=cutoff * unit.angstrom,
                constraints=HBonds,
            )

            # Add restraints
            for atom in self._structure.atoms:
                if atom.name == "DUM":
                    positional_restraint = CustomExternalForce(
                        "k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
                    )
                    positional_restraint.addPerParticleParameter("k")
                    positional_restraint.addPerParticleParameter("x0")
                    positional_restraint.addPerParticleParameter("y0")
                    positional_restraint.addPerParticleParameter("z0")

                    k = 50.0 * unit.kilocalorie_per_mole / unit.angstrom**2
                    x0 = 0.1 * atom.xx * unit.nanometers
                    y0 = 0.1 * atom.xy * unit.nanometers
                    z0 = 0.1 * atom.xz * unit.nanometers
                    positional_restraint.addParticle(atom.idx, [k, x0, y0, z0])
                    system.addForce(positional_restraint)
                    positional_restraint.setForceGroup(15)

            for restraint in self._static_restraints:
                system = apply_openmm_restraints(system, restraint, window, ForceGroup=10)
            for restraint in self._conformational_restraints:
                system = apply_openmm_restraints(system, restraint, window, ForceGroup=11)
            for restraint in self._guest_restraints:
                system = apply_openmm_restraints(system, restraint, window, ForceGroup=12)
            for restraint in self._symmetry_restraints:
                system = apply_openmm_restraints(system, restraint, window, flat_bottom=True, ForceGroup=13)
            for restraint in self._wall_restraints:
                system = apply_openmm_restraints(system, restraint, window, flat_bottom=True, ForceGroup=14)

            if self._cv_host_rmsd is not None:
                inpcrd = AmberInpcrdFile(self._structure.name.replace("prmtop", "rst7"))
                rmsd_cv = RMSDForce(inpcrd.getPositions(), self._cv_host_rmsd['cv_i'])
                rmsd_cv.setForceGroup(16)
                rmsd_restraint = CustomCVForce('k_rmsd * (RMSD-RMSD0)^2')
                rmsd_restraint.addCollectiveVariable('RMSD', rmsd_cv)
                rmsd_restraint.addGlobalParameter(
                    'k_rmsd', 
                    self._cv_host_rmsd['k']*self._apr_list[phase]['fractions'][self._apr_list[phase]['windows'].index(window)]*unit.kilocalories_per_mole/unit.angstroms**2
                )
                rmsd_restraint.addGlobalParameter('RMSD0', self._cv_host_rmsd['target']*unit.angstroms)
                system.addForce(rmsd_restraint)
                rmsd_restraint.setForceGroup(16)

            if self._cv_guest_rmsd is not None:
                inpcrd = AmberInpcrdFile(self._structure.name.replace("prmtop", "rst7"))
                rmsd_cv = RMSDForce(inpcrd.getPositions(), self._cv_guest_rmsd['cv_i'])
                rmsd_cv.setForceGroup(17)
                rmsd_restraint = CustomCVForce('k_rmsd * (RMSD-RMSD0)^2')
                rmsd_restraint.addCollectiveVariable('RMSD', rmsd_cv)
                rmsd_restraint.addGlobalParameter(
                    'k_rmsd', 
                    self._cv_guest_rmsd['k']*self._apr_list[phase]['fractions'][self._apr_list[phase]['windows'].index(window)]*unit.kilocalories_per_mole/unit.angstroms**2
                )
                rmsd_restraint.addGlobalParameter('RMSD0', self._cv_guest_rmsd['target']*unit.angstroms)
                system.addForce(rmsd_restraint)
                rmsd_restraint.setForceGroup(17)

            # Write XML file
            system_xml = XmlSerializer.serialize(system)

            with open(os.path.join(sub_folder, output_xml), "w") as file:
                file.write(system_xml)

    def dump_restraints_file(
        self,
        json_file="restraints.json",
        restr_type="amber",
        ref_from_structure=False,
    ):
        """
        Method to write pAPRika restraints to either Amber NMR-style or Plumed format

        Parameters
        ----------
        output_prefix : str
            output folder for dumping restraint files
        json_file : str
            filename for pAPRika restraint file (.json)
        restr_type : str
            restraint format - Amber NMR or Plumed
        ref_from_structure : bool
            take the reference position of dummy atoms from current structure

        """
        # dump restraint file for later use in analysis
        restraint_list = self._static_restraints.copy()

        if self._conformational_restraints:
            restraint_list += self._conformational_restraints.copy()

        if self._guest_restraints:
            restraint_list += self._guest_restraints.copy()

        save_restraints(restraint_list=restraint_list, filepath=os.path.join(self._output_path, self._output_prefix, json_file))

        # Write restraints to file
        restraint_file = "disang.rest"
        list_type = "tuple"
        if restr_type.lower() == "plumed":
            restraint_file = "plumed.dat"
            list_type = "dict"

        total_restraints = len(
                self._static_restraints
                + self._guest_restraints
                + self._conformational_restraints
                + self._wall_restraints
                + self._symmetry_restraints)

        for window in self._window_list:
            sub_folder = os.path.join(self._output_path, self._output_prefix, window)
            if total_restraints != 0:
                with open(os.path.join(sub_folder, restraint_file), "w") as file:
                    if window[0] == "a":
                        restraints = parse_restraints(
                            static=self._static_restraints,
                            guest=self._guest_restraints,
                            host=self._conformational_restraints,
                            wall=self._wall_restraints,
                            symmetry=self._symmetry_restraints,
                            list_type=list_type,
                        )
                    if window[0] == "p":
                        restraints = parse_restraints(
                            static=self._static_restraints,
                            guest=self._guest_restraints,
                            host=self._conformational_restraints,
                            list_type=list_type,
                        )
                    if window[0] == "r" and self._protocol == "a-p-r":
                        restraints = parse_restraints(
                            static=self._static_restraints,
                            guest=self._guest_restraints,
                            host=self._conformational_restraints,
                            list_type=list_type,
                        )
                    if window[0] == "r" and self._protocol == "r":
                        restraints = parse_restraints(
                            static=self._static_restraints,
                            host=self._conformational_restraints,
                            list_type=list_type,
                        )

                    if restr_type.lower() == "amber":
                        for restraint in restraints:
                            string = amber_restraint_line(restraint, window)
                            if string is not None:
                                file.write(string)

                    elif restr_type.lower() == "plumed":
                        plumed_colvar_file(file, restraints, window, legacy_k=True)

                        if ref_from_structure:
                            self._fetch_dummy_atoms(serial=True)
                            write_dummy_to_plumed(file, self._dummy_atoms)

            if self._cv_host_rmsd is not None:
                kfactor = 1.0
                if self._cv_host_rmsd['scale']:
                    if window[0] == 'a':
                        kfactor *= self._apr_list["attach"]["fractions"][self._apr_list["attach"]["windows"].index(window)]
                    elif window[0] == 'r':
                        kfactor *= self._apr_list["release"]["fractions"][self._apr_list["release"]["windows"].index(window)]

                with open(os.path.join(sub_folder, 'colvars.in'), 'w') as file:
                    file.writelines('&colvar\n')
                    file.writelines('  cv_type = \'MULTI_RMSD\',\n')
                    file.writelines('  cv_ni = {}, cv_nr = {},\n'.format(self._cv_host_rmsd['cv_ni'], self._cv_host_rmsd['cv_nr']))
                    file.writelines('  cv_i = ')
                    for index in self._cv_host_rmsd['cv_i']:
                        file.writelines('{},'.format(index))
                    file.writelines('0,\n')
                    file.writelines('  cv_r = ')
                    for pos in self._cv_host_rmsd['cv_r']:
                        file.writelines('{},{},{},\n'.format(pos[0], pos[1], pos[2]))
                    file.writelines('  anchor_position = 0.0,{},{},999.0,\n'.format(self._cv_host_rmsd['target'], self._cv_host_rmsd['target']))
                    file.writelines('  anchor_strength = {},{},\n'.format(self._cv_host_rmsd['k']*kfactor, self._cv_host_rmsd['k']*kfactor))
                    file.writelines('/\n')

            if self._cv_guest_rmsd is not None:
                kfactor = 1.0
                if self._cv_guest_rmsd['scale']:
                    if window[0] == 'a':
                        kfactor *= self._apr_list["attach"]["fractions"][self._apr_list["attach"]["windows"].index(window)]
                    elif window[0] == 'r':
                        kfactor *= self._apr_list["release"]["fractions"][self._apr_list["release"]["windows"].index(window)]

                fio = 'w'
                if os.path.exists(os.path.join(sub_folder, 'colvars.in')):
                    fio = 'a'

                with open(os.path.join(sub_folder, 'colvars.in'), fio) as file:
                    file.writelines('&colvar\n')
                    file.writelines('  cv_type = \'MULTI_RMSD\',\n')
                    file.writelines('  cv_ni = {}, cv_nr = {},\n'.format(self._cv_guest_rmsd['cv_ni'], self._cv_guest_rmsd['cv_nr']))
                    file.writelines('  cv_i = ')
                    for index in self._cv_guest_rmsd['cv_i']:
                        file.writelines('{},'.format(index))
                    file.writelines('0,\n')
                    file.writelines('  cv_r = ')
                    for pos in self._cv_guest_rmsd['cv_r']:
                        file.writelines('{},{},{},\n'.format(pos[0], pos[1], pos[2]))
                    file.writelines('  anchor_position = 0.0,{},{},999.0,\n'.format(self._cv_guest_rmsd['target'], self._cv_guest_rmsd['target']))
                    file.writelines('  anchor_strength = {},{},\n'.format(self._cv_guest_rmsd['k']*kfactor, self._cv_guest_rmsd['k']*kfactor))
                    file.writelines('/\n')

