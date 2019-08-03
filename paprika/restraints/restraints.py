import logging

import numpy as np
import parmed as pmd
import pytraj as pt

from paprika import utils

try:
    import simtk.openmm as mm
    import simtk.unit as unit
except ImportError:
    pass

logger = logging.getLogger(__name__)


class DAT_restraint(object):
    """
    Distance or angle or torsion restraints on atoms in the simulation.
    """

    instances = []

    def __init__(self):

        self.topology = None
        self.mask1 = None
        self.mask2 = None
        self.mask3 = None
        self.mask4 = None
        self.index1 = None
        self.index2 = None
        self.index3 = None
        self.index4 = None

        # In the case of a non-harmonic restraint, the pre-calculated values can be overridden with
        # ones from this dictionary.
        self.custom_restraint_values = {}

        self.auto_apr = (
            False
        )  # If True, sets some pull and release values automatically.
        # If True, the first window of pull is re-used as last window of attach and the last window
        # of pull is re-used as first window of release.
        self.continuous_apr = True
        self.amber_index = False

        self.attach = {
            "target": None,  # The target value for the restraint (mandatory)
            "fc_initial": None,  # The initial force constant (optional)
            "fc_final": None,  # The final force constant (optional)
            "num_windows": None,  # The number of windows (optional)
            "fc_increment": None,  # The force constant increment (optional)
            # The percentage of the force constant increment (optional)
            "fraction_increment": None,
            # The list of force constant percentages (optional)
            "fraction_list": None,
            # The list of force constants (will be created if not given)
            "fc_list": None,
        }

        self.pull = {
            "fc": None,  # The force constant for the restraint (mandatory)
            "target_initial": None,  # The initial target value (optional)
            "target_final": None,  # The final target value (optional)
            "num_windows": None,  # The number of windows (optional)
            "target_increment": None,  # The target value increment (optional)
            # The percentage of the target value increment (optional)
            "fraction_increment": None,
            # The list of target value percentages (optional)
            "fraction_list": None,
            # The list of target values (will be created if not given)
            "target_list": None,
        }

        self.release = {
            "target": None,  # The target value for the restraint (mandatory)
            "fc_initial": None,  # The initial force constant (optional)
            "fc_final": None,  # The final force constant (optional)
            "num_windows": None,  # The number of windows (optional)
            "fc_increment": None,  # The force constant increment (optional)
            # The percentage of the force constant increment (optional)
            "fraction_increment": None,
            # The list of force constant percentages (optional)
            "fraction_list": None,
            # The list of force constants (will be created if not
            "fc_list": None
            # given)
        }

        DAT_restraint.instances.append(self)

    def __eq__(self, other):
        self_dictionary = self.__dict__
        other_dictionary = other.__dict__
        for dct in [self_dictionary, other_dictionary]:
            # Skip checking topology.
            # It is difficult to check topology for two reasons.
            # One, there are numerical variances in the numeric parameters.
            # We can use `np.allcose()` but it needs to be done manually on the
            # nested ParmEd AmberParm or Structure classes.
            # Two, sometimes the dummy atoms seem to be encoded with index `-1`.
            # This can happen if the structure used to setup the restraint
            # does not have the dummy atoms already included.
            # Previously, I checked if the topology was an AmberParm, and tried
            # to replace that with the string representation of the file name,
            # but for some reason, that attribute was not always available.
            # Thus, I simply do not compare the `topology` attribute.
            dct["topology"] = None
        logger.debug(self_dictionary)
        logger.debug(other_dictionary)
        keys = set(self_dictionary.keys()) & set(other_dictionary.keys())
        for key in keys:
            if key != "phase":
                assert self_dictionary[key] == other_dictionary[key]
            else:
                for phs in ["attach", "pull", "release"]:
                    for value in ["force_constants", "targets"]:
                        if (
                            self_dictionary["phase"][phs][value] is None
                            and other_dictionary["phase"][phs][value] is None
                        ):
                            continue
                        else:
                            assert np.allclose(
                                self_dictionary["phase"][phs][value],
                                other_dictionary["phase"][phs][value],
                            )
        return True

    def _calc_meth(self, phase, rdict, meth):
        """ Return the appropriate list of force_constants and targets depending on the method """

        force_constants = None
        targets = None

        # Attach/Release, Force Constant Method 1
        if phase in ("a", "r") and meth == "1":
            force_constants = np.linspace(
                rdict["fc_initial"], rdict["fc_final"], rdict["num_windows"]
            )

        # Attach/Release, Force Constant Method 1a
        elif phase in ("a", "r") and meth == "1a":
            force_constants = np.linspace(0.0, rdict["fc_final"], rdict["num_windows"])

        # Attach/Release, Force Constant Method 2
        elif phase in ("a", "r") and meth == "2":
            force_constants = np.arange(
                rdict["fc_initial"],
                rdict["fc_final"] + rdict["fc_increment"],
                rdict["fc_increment"],
            )

        # Attach/Release, Force Constant Method 2a
        elif phase in ("a", "r") and meth == "2a":
            force_constants = np.arange(
                0.0, rdict["fc_final"] + rdict["fc_increment"], rdict["fc_increment"]
            )

        # Attach/Release, Force Constant Method 3
        elif phase in ("a", "r") and meth == "3":
            force_constants = np.asarray(
                [fraction * rdict["fc_final"] for fraction in rdict["fraction_list"]]
            )

        # Attach/Release, Force Constant Method 4
        elif phase in ("a", "r") and meth == "4":
            fractions = np.arange(
                0, 1.0 + rdict["fraction_increment"], rdict["fraction_increment"]
            )
            force_constants = np.asarray(
                [fraction * rdict["fc_final"] for fraction in fractions]
            )

        # Attach/Release, Force Constant Method 5
        elif phase in ("a", "r") and meth == "5":
            force_constants = np.asarray(rdict["fc_list"])

        # Attach/Release, Target Method
        if phase in ("a", "r"):
            targets = np.asarray([rdict["target"]] * len(force_constants))

        # Pull, Target Method 1
        if phase == "p" and meth == "1":
            targets = np.linspace(
                rdict["target_initial"], rdict["target_final"], rdict["num_windows"]
            )

        # Pull, Target Method 1a
        elif phase == "p" and meth == "1a":
            targets = np.linspace(0.0, rdict["target_final"], rdict["num_windows"])

        # Pull, Target Method 2
        elif phase == "p" and meth == "2":
            targets = np.arange(
                rdict["target_initial"],
                rdict["target_final"] + rdict["target_increment"],
                rdict["target_increment"],
            )

        # Pull, Target Method 2a
        elif phase == "p" and meth == "2a":
            targets = np.arange(
                0.0,
                rdict["target_final"] + rdict["target_increment"],
                rdict["target_increment"],
            )

        # Pull, Target Method 3
        elif phase == "p" and meth == "3":
            targets = np.asarray(
                [
                    fraction * rdict["target_final"]
                    for fraction in rdict["fraction_list"]
                ]
            )

        # Pull, Target Method 4
        elif phase == "p" and meth == "4":
            fractions = np.arange(
                0, 1.0 + rdict["fraction_increment"], rdict["fraction_increment"]
            )
            targets = np.asarray(
                [fraction * rdict["target_final"] for fraction in fractions]
            )

        # Pull, Target Method 5
        elif phase == "p" and meth == "5":
            targets = np.asarray(rdict["target_list"])

        # Pull, Force Constant Method
        if phase == "p":
            force_constants = np.asarray([rdict["fc"]] * len(targets))

        if force_constants is None and targets is None:
            logger.error("Unsupported Phase/Method: {} / {}".format(phase, meth))
            raise Exception("Unexpected phase/method combination passed to _calc_meth")

        return force_constants, targets

    def initialize(self):
        """
        Depending on which dict values are provided for each phase, a different method will
        be used to determine the list of force_constants and targets (below).

        For Attach/Release, a `target` value is required and the method is determined if the
        following dict values are not `None`:
            Method 1:   num_windows, fc_initial, fc_final
            Method 1a:  num_windows, fc_final
            Method 2:   fc_increment, fc_initial, fc_final
            Method 2a:  fc_increment, fc_final
            Method 3:   fraction_list, fc_final
            Method 4:   fraction_increment, fc_final
            Method 5:   fc_list

        For Pull, a `fc` value is required and the method is determined if the
        following dict values are not `None`:
            Method 1:   num_windows, target_initial, target_final
            Method 1a:  num_windows, target_final
            Method 2:   target_increment, target_initial, target_final
            Method 2a:  target_increment, target_final
            Method 3:   fraction_list, target_final
            Method 4:   fraction_increment, target_final
            Method 5:   target_list
        """

        # Setup. These are the lists that will be most used by other modules
        self.phase = {
            "attach": {"force_constants": None, "targets": None},
            "pull": {"force_constants": None, "targets": None},
            "release": {"force_constants": None, "targets": None},
        }
        # ------------------------------------ ATTACH ------------------------------------ #
        logger.debug("Calculating attach targets and force constants...")

        # Temporary variables to improve readability
        force_constants = None
        targets = None

        if (
            self.attach["num_windows"] is not None
            and self.attach["fc_final"] is not None
        ):
            if self.attach["fc_initial"] is not None:
                ### METHOD 1 ###
                logger.debug("Attach, Method #1")
                force_constants, targets = self._calc_meth("a", self.attach, "1")
            else:
                ### METHOD 1a ###
                logger.debug("Attach, Method #1a")
                force_constants, targets = self._calc_meth("a", self.attach, "1a")

        elif (
            self.attach["fc_increment"] is not None
            and self.attach["fc_final"] is not None
        ):
            if self.attach["fc_initial"] is not None:
                ### METHOD 2 ###
                logger.debug("Attach, Method #2")
                force_constants, targets = self._calc_meth("a", self.attach, "2")
            else:
                ### METHOD 2a ###
                logger.debug("Attach, Method #2a")
                force_constants, targets = self._calc_meth("a", self.attach, "2a")

        elif (
            self.attach["fraction_list"] is not None
            and self.attach["fc_final"] is not None
        ):
            ### METHOD 3 ###
            logger.debug("Attach, Method #3")
            force_constants, targets = self._calc_meth("a", self.attach, "3")

        elif (
            self.attach["fraction_increment"] is not None
            and self.attach["fc_final"] is not None
        ):
            ### METHOD 4 ###
            logger.debug("Attach, Method #4")
            force_constants, targets = self._calc_meth("a", self.attach, "4")

        elif self.attach["fc_list"] is not None:
            ### METHOD 5 ###
            logger.debug("Attach, Method #5")
            force_constants, targets = self._calc_meth("a", self.attach, "5")

        elif all(v is None for k, v in self.attach.items()):
            logger.debug("No restraint info set for the attach phase! Skipping...")

        else:
            logger.error(
                "Attach restraint input did not match one of the supported methods..."
            )
            for k, v in self.attach.items():
                logger.debug("{} = {}".format(k, v))
            raise Exception(
                "Attach restraint input did not match one of the supported methods..."
            )

        if force_constants is not None and targets is not None:
            self.phase["attach"]["force_constants"] = force_constants
            self.phase["attach"]["targets"] = targets

        # ------------------------------------ PULL ------------------------------------ #
        logger.debug("Calculating pull targets and force constants...")

        force_constants = None
        targets = None

        if self.auto_apr and self.pull["target_final"] is not None:
            self.pull["fc"] = self.phase["attach"]["force_constants"][-1]
            self.pull["target_initial"] = self.phase["attach"]["targets"][-1]

        if (
            self.pull["num_windows"] is not None
            and self.pull["target_final"] is not None
        ):
            if self.pull["target_initial"] is not None:
                ### METHOD 1 ###
                logger.debug("Pull, Method #1")
                force_constants, targets = self._calc_meth("p", self.pull, "1")
            else:
                ### METHOD 1a ###
                logger.debug("Pull, Method #1a")
                force_constants, targets = self._calc_meth("p", self.pull, "1a")

        elif (
            self.pull["target_increment"] is not None
            and self.pull["target_final"] is not None
        ):
            if self.pull["target_initial"] is not None:
                ### METHOD 2 ###
                logger.debug("Pull, Method #2")
                force_constants, targets = self._calc_meth("p", self.pull, "2")
            else:
                ### METHOD 2a ###
                logger.debug("Pull, Method #2a")
                force_constants, targets = self._calc_meth("p", self.pull, "2a")

        elif (
            self.pull["fraction_list"] is not None
            and self.pull["target_final"] is not None
        ):
            ### METHOD 3 ###
            logger.debug("Pull, Method #3")
            force_constants, targets = self._calc_meth("p", self.pull, "3")

        elif (
            self.pull["fraction_increment"] is not None
            and self.pull["target_final"] is not None
        ):
            ### METHOD 4 ###
            logger.debug("Pull, Method #4")
            force_constants, targets = self._calc_meth("p", self.pull, "4")

        elif self.pull["target_list"] is not None:
            ### METHOD 5 ###
            logger.debug("Pull, Method #5")
            force_constants, targets = self._calc_meth("p", self.pull, "5")

        elif all(v is None for k, v in self.pull.items()):
            logger.debug("No restraint info set for the pull phase! Skipping...")

        else:
            logger.error(
                "Pull restraint input did not match one of the supported methods..."
            )
            for k, v in self.pull.items():
                logger.debug("{} = {}".format(k, v))
            raise Exception(
                "Pull restraint input did not match one of the supported methods..."
            )

        if force_constants is not None and targets is not None:
            self.phase["pull"]["force_constants"] = force_constants
            self.phase["pull"]["targets"] = targets

        # ------------------------------------ RELEASE ------------------------------------ #
        logger.debug("Calculating release targets and force constants...")

        force_constants = None
        targets = None

        # I don't want auto_apr to make release restraints, unless I'm sure the user wants them.
        # I'm gonna assume that specifying self.attach['fc_final'] indicates you want it,
        # although this weakens the whole purpose of auto_apr
        if self.auto_apr and self.release["fc_final"] is not None:
            self.release["target"] = self.phase["pull"]["targets"][-1]
            for key in [
                "fc_final",
                "fc_initial",
                "num_windows",
                "fc_increment",
                "fraction_increment",
                "fraction_list",
                "fc_list",
            ]:
                if self.attach[key] is not None and self.release[key] is None:
                    self.release[key] = self.attach[key]

        if (
            self.release["num_windows"] is not None
            and self.release["fc_final"] is not None
        ):
            if self.release["fc_initial"] is not None:
                ### METHOD 1 ###
                logger.debug("Release, Method #1")
                force_constants, targets = self._calc_meth("r", self.release, "1")
            else:
                ### METHOD 1a ###
                logger.debug("Release, Method #1a")
                force_constants, targets = self._calc_meth("r", self.release, "1a")

        elif (
            self.release["fc_increment"] is not None
            and self.release["fc_final"] is not None
        ):
            if self.release["fc_initial"] is not None:
                ### METHOD 2 ###
                logger.debug("Release, Method #2")
                force_constants, targets = self._calc_meth("r", self.release, "2")
            else:
                ### METHOD 2a ###
                logger.debug("Release, Method #2a")
                force_constants, targets = self._calc_meth("r", self.release, "2a")

        elif (
            self.release["fraction_list"] is not None
            and self.release["fc_final"] is not None
        ):
            ### METHOD 3 ###
            logger.debug("Release, Method #3")
            force_constants, targets = self._calc_meth("r", self.release, "3")

        elif (
            self.release["fraction_increment"] is not None
            and self.release["fc_final"] is not None
        ):
            ### METHOD 4 ###
            logger.debug("Release, Method #4")
            force_constants, targets = self._calc_meth("r", self.release, "4")

        elif self.release["fc_list"] is not None:
            ### METHOD 5 ###
            logger.debug("Release, Method #5")
            force_constants, targets = self._calc_meth("r", self.release, "5")

        elif all(v is None for k, v in self.release.items()):
            logger.debug("No restraint info set for the release phase! Skipping...")

        else:
            logger.error(
                "Release restraint input did not match one of the supported methods..."
            )
            for k, v in self.release.items():
                logger.debug("{} = {}".format(k, v))
            raise Exception(
                "Release restraint input did not match one of the supported methods..."
            )

        if force_constants is not None and targets is not None:
            self.phase["release"]["force_constants"] = force_constants
            self.phase["release"]["targets"] = targets

        # ----------------------------------- WINDOWS ------------------------------------ #

        for phase in ["attach", "pull", "release"]:
            if self.phase[phase]["targets"] is not None:
                window_count = len(self.phase[phase]["targets"])
                # DAT_restraint.window_counts[phase].append(window_count)
                logger.debug("Number of {} windows = {}".format(phase, window_count))
            else:
                # DAT_restraint.window_counts[phase].append(None)
                logger.debug(
                    "This restraint will be skipped in the {} phase".format(phase)
                )

        # ---------------------------------- ATOM MASKS ---------------------------------- #
        logger.debug("Assigning atom indices...")
        self.index1 = utils.index_from_mask(self.topology, self.mask1, self.amber_index)
        self.index2 = utils.index_from_mask(self.topology, self.mask2, self.amber_index)
        if self.mask3:
            self.index3 = utils.index_from_mask(
                self.topology, self.mask3, self.amber_index
            )
        else:
            self.index3 = None
        if self.mask4:
            self.index4 = utils.index_from_mask(
                self.topology, self.mask4, self.amber_index
            )
        else:
            self.index4 = None
        # If any `index` has more than one atom, mark it as a group restraint.
        # print('Masks:',self.mask1, self.mask2, self.mask3, self.mask4)
        # print('index:',self.index1, self.index2, self.index3, self.index4)
        if self.mask1 and len(self.index1) > 1:
            self.group1 = True
        else:
            self.group1 = False
        if self.mask2 and len(self.index2) > 1:
            self.group2 = True
        else:
            self.group2 = False
        if self.mask3 and len(self.index3) > 1:
            self.group3 = True
        else:
            self.group3 = False
        if self.mask4 and len(self.index4) > 1:
            self.group4 = True
        else:
            self.group4 = False
        # print('index:',self.group1, self.group2, self.group3, self.group4)


def static_DAT_restraint(
    restraint_mask_list,
    num_window_list,
    ref_structure,
    force_constant,
    continuous_apr=True,
    amber_index=False,
):
    """ Create a static restraint """
    ref_traj = pt.iterload(ref_structure, traj=True)

    # Check num_window_list
    if len(num_window_list) != 3:
        raise Exception(
            "The num_window_list needs to contain three integers corresponding to the number of windows in the "
            "attach, pull, and release phase, respectively "
        )

    # Setup restraint
    rest = DAT_restraint()
    rest.continuous_apr = continuous_apr
    rest.amber_index = amber_index
    rest.topology = pmd.load_file(ref_structure, structure=True)
    rest.mask1 = restraint_mask_list[0]
    rest.mask2 = restraint_mask_list[1]
    if len(restraint_mask_list) >= 3:
        rest.mask3 = restraint_mask_list[2]
    if len(restraint_mask_list) == 4:
        rest.mask4 = restraint_mask_list[3]

    # Target value
    mask_string = " ".join(restraint_mask_list)
    if len(restraint_mask_list) == 2:
        # Distance restraint
        target = pt.distance(ref_traj, mask_string, image=True)[0]
    elif len(restraint_mask_list) == 3:
        # Angle restraint
        target = pt.angle(ref_traj, mask_string)[0]
    elif len(restraint_mask_list) == 4:
        # Dihedral restraint
        target = pt.dihedral(ref_traj, mask_string)[0]
    else:
        raise Exception(
            "The number of masks ("
            + str(len(restraint_mask_list))
            + ") in restraint_mask_list is not 2, 3, or 4 and thus is not one of the supported types: distance, angle, dihedral"
        )

    # Attach phase
    if num_window_list[0] is not None and num_window_list[0] != 0:
        rest.attach["target"] = target
        rest.attach["fc_initial"] = force_constant
        rest.attach["fc_final"] = force_constant
        rest.attach["num_windows"] = num_window_list[0]

    # Pull phase
    if num_window_list[1] is not None and num_window_list[1] != 0:
        rest.pull["fc"] = force_constant
        rest.pull["target_initial"] = target
        rest.pull["target_final"] = target
        rest.pull["num_windows"] = num_window_list[1]

    # Release phase
    if num_window_list[2] is not None and num_window_list[2] != 0:
        rest.release["target"] = target
        rest.release["fc_initial"] = force_constant
        rest.release["fc_final"] = force_constant
        rest.release["num_windows"] = num_window_list[2]

    rest.initialize()

    return rest


def check_restraints(restraint_list, create_window_list=False):
    """
    Do basic tests to ensure a list of DAT_restraints are consistent.
    We're gonna create the window list here too, because it needs the same code.
    """

    if all(restraint.continuous_apr is True for restraint in restraint_list):
        logger.debug('All restraints are "continuous_apr" style.')
        all_continuous_apr = True
    elif all(restraint.continuous_apr is False for restraint in restraint_list):
        logger.debug('All restraints are not "continuous_apr" style.')
        all_continuous_apr = False
    else:
        logger.error("All restraints must have the same setting for .continuous_apr")
        # Should we do the following?
        raise Exception("All restraints must have the same setting for .continuous_apr")

    window_list = []
    phases = ["attach", "pull", "release"]
    for phase in phases:
        win_counts = []
        for restraint in restraint_list:
            if restraint.phase[phase]["targets"] is not None:
                win_counts.append(len(restraint.phase[phase]["targets"]))
            else:
                win_counts.append(0)
        max_count = np.max(win_counts)

        if max_count > 999:
            logger.info("Window name zero padding only applied up to 999.")

        # For each restraint, make sure the number of windows is either 0 (the restraint
        # is not active) or equal to the maximum number of windows for any
        # restraint.
        if all(count == 0 or count == max_count for count in win_counts):
            if max_count > 0:
                # `continuous_apr` during attach means that the final attach window
                # should be skipped and replaced with `p000`. `continuous_apr` during
                # release means that `r000` should be skipped and replaced with the
                # final pull window.

                if phase == "attach" and all_continuous_apr:
                    window_list += [
                        phase[0] + str("{:03.0f}".format(val))
                        for val in np.arange(0, max_count - 1, 1)
                    ]
                elif phase == "attach" and not all_continuous_apr:
                    window_list += [
                        phase[0] + str("{:03.0f}".format(val))
                        for val in np.arange(0, max_count, 1)
                    ]
                elif phase == "pull":
                    window_list += [
                        phase[0] + str("{:03.0f}".format(val))
                        for val in np.arange(0, max_count, 1)
                    ]
                elif phase == "release" and all_continuous_apr:
                    window_list += [
                        phase[0] + str("{:03.0f}".format(val))
                        for val in np.arange(1, max_count, 1)
                    ]
                elif phase == "release" and not all_continuous_apr:
                    window_list += [
                        phase[0] + str("{:03.0f}".format(val))
                        for val in np.arange(0, max_count, 1)
                    ]
        else:
            logger.error(
                "Restraints have unequal number of windows during the {} phase.".format(
                    phase
                )
            )
            logger.debug("Window counts for each restraint are as follows:")
            logger.debug(win_counts)
            raise Exception(
                "Restraints have unequal number of windows during the {} "
                "phase.".format(phase)
            )

    logger.info("Restraints appear to be consistent")

    if create_window_list:
        return window_list


def create_window_list(restraint_list):
    """
    Create list of APR windows. Runs everything through check_restraints because
    we need to do that.
    """

    return check_restraints(restraint_list, create_window_list=True)