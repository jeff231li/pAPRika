"""
Tests the restraints utilities.
"""

import pytest
import os
import shutil

from paprika.restraints.restraints import *
from paprika.restraints.plumed import *
from paprika.utils import *
from paprika.align import *
from paprika.dummy import *
from paprika.tleap import *


@pytest.fixture
def clean_files(directory=os.path.join(os.path.dirname(__file__), "tmp")):
    # This happens before the test function call
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    yield
    # This happens after the test function call
    shutil.rmtree(directory)


def test_restraint_to_colvar(clean_files):
    rest1 = DAT_restraint()
    rest1.amber_index = True
    rest1.continuous_apr = False
    rest1.auto_apr = False
    rest1.topology = os.path.join(os.path.dirname(__file__), "../data/cb6-but/cb6-but-notcentered.pdb")
    rest1.mask1 = ":CB6@O,O2,O4,O6,O8,O10"
    rest1.mask2 = ":BUT@C3"
    rest1.attach["target"] = 3.0
    rest1.attach["num_windows"] = 4
    rest1.attach["fc_initial"] = 0.0
    rest1.attach["fc_final"] = 3.0
    rest1.pull["fc"] = rest1.attach["fc_final"]
    rest1.pull["num_windows"] = 4
    rest1.pull["target_initial"] = rest1.attach["target"]
    rest1.pull["target_final"] = 6.0
    rest1.release["target"] = rest1.pull["target_final"]
    rest1.release["num_windows"] = rest1.attach["num_windows"]
    rest1.release["fc_initial"] = rest1.attach["fc_initial"]
    rest1.release["fc_final"] = rest1.attach["fc_final"]
    rest1.initialize()

    colvar = restraint_to_colvar([rest1], 'attach', 0)
    assert colvar['AT'][0] == rest1.attach["target"]
    assert colvar['KAPPA'][0] == rest1.attach['fc_initial']

    colvar = restraint_to_colvar([rest1], 'attach', 3)
    assert colvar['AT'][0] == rest1.attach["target"]
    assert colvar['KAPPA'][0] == rest1.attach['fc_final']

    colvar = restraint_to_colvar([rest1], 'pull', 0)
    assert colvar['AT'][0] == rest1.pull["target_initial"]
    assert colvar['KAPPA'][0] == rest1.pull['fc']

    colvar = restraint_to_colvar([rest1], 'pull', 3)
    assert colvar['AT'][0] == rest1.pull["target_final"]
    assert colvar['KAPPA'][0] == rest1.pull['fc']

    colvar = restraint_to_colvar([rest1], 'release', 0)
    assert colvar['AT'][0] == rest1.release["target"]
    assert colvar['KAPPA'][0] == rest1.release['fc_initial']

    colvar = restraint_to_colvar([rest1], 'release', 3)
    assert colvar['AT'][0] == rest1.release["target"]
    assert colvar['KAPPA'][0] == rest1.release['fc_final']


def test_extract_dummy_atoms(clean_files):
    """ Test that dummy atoms get added correctly """
    temporary_directory = os.path.join(os.path.dirname(__file__), "tmp")
    host_guest = pmd.load_file(
        os.path.join(
            os.path.dirname(__file__), "../data/cb6-but/cb6-but-notcentered.pdb"
        ),
        structure=True,
    )
    host_guest = zalign(host_guest, ":BUT@C", ":BUT@C3", save=False)
    host_guest = add_dummy(host_guest, residue_name="DM1", z=-11.000, y=2.000, x=-1.500)
    host_guest = add_dummy(host_guest, residue_name="DM2", z=-13.000, y=2.000, x=-1.500)
    host_guest = add_dummy(host_guest, residue_name="DM3", z=-15.200, y=4.200, x=-1.500)

    host_guest.save(
        os.path.join(temporary_directory, "cb6-but-dum.pdb"), overwrite=True, #renumber=False
    )

    write_dummy_frcmod(path=temporary_directory)
    write_dummy_mol2(path=temporary_directory, filename="dm1.mol2", residue_name="DM1")
    write_dummy_mol2(path=temporary_directory, filename="dm2.mol2", residue_name="DM2")
    write_dummy_mol2(path=temporary_directory, filename="dm3.mol2", residue_name="DM3")

    sys = System()
    cb6_frcmod = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/cb6-but/cb6.frcmod")
    )
    cb6_mol2 = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/cb6-but/cb6.mol2")
    )
    but_frcmod = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/cb6-but/but.frcmod")
    )
    but_mol2 = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/cb6-but/but.mol2")
    )

    sys.template_lines = [
        "source leaprc.gaff",
        f"loadamberparams {cb6_frcmod}",
        f"CB6 = loadmol2 {cb6_mol2}",
        f"loadamberparams {but_frcmod}",
        f"BUT = loadmol2 {but_mol2}",
        "loadamberparams dummy.frcmod",
        "DM1 = loadmol2 dm1.mol2",
        "DM2 = loadmol2 dm2.mol2",
        "DM3 = loadmol2 dm3.mol2",
        "model = loadpdb cb6-but-dum.pdb",
    ]
    sys.output_path = temporary_directory
    sys.output_prefix = "cb6-but-dum"
    sys.pbc_type = None
    sys.neutralize = False
    sys.build()

    structure = pmd.load_file(
        os.path.join(temporary_directory, "cb6-but-dum.prmtop"),
        os.path.join(temporary_directory, "cb6-but-dum.rst7"),
    )
    dummy_atoms = extract_dummy_atoms(structure, serial=True)
    print(type(dummy_atoms["DM1"]["pos"]))

    assert np.allclose(dummy_atoms["DM1"]["pos"], [-1.5, 2.0, -11.0])
    assert np.allclose(dummy_atoms["DM2"]["pos"], [-1.5, 2.0, -13.0])
    assert np.allclose(dummy_atoms["DM3"]["pos"], [-1.5, 4.2, -15.2])
