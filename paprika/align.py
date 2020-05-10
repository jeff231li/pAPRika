import logging

import numpy as np
import parmed as pmd

logger = logging.getLogger(__name__)


def zalign(structure, mask1, mask2, save=False, filename=None):
    """Align the mask1 -- mask2 vector to the z axis.

    Parameters
    ----------
    structure : parmed.Structure
        Molecular structure containing coordinates
    mask1 : str
        Selection of first set of atoms
    mask2 : str
        Selection of second set of atoms
    save : bool, optional
        Whether to save the coordinates (the default is False, which does nothing)
    filename : str, optional
        The filename for the saved coordinates (the default is None, which does nothing)

    Returns
    -------
    parmed.Structure
        A molecular structure with the coordinates aligned as specified.
    """

    mask1_coordinates = structure[mask1].coordinates
    mask1_masses = [atom.mass for atom in structure[mask1].atoms]
    mask1_com = pmd.geometry.center_of_mass(
        np.asarray(mask1_coordinates), np.asarray(mask1_masses)
    )

    mask2_coordinates = structure[mask2].coordinates
    mask2_masses = [atom.mass for atom in structure[mask2].atoms]
    mask2_com = pmd.geometry.center_of_mass(
        np.asarray(mask2_coordinates), np.asarray(mask2_masses)
    )

    logger.info(
        "Moving {} ({} atoms) to the origin...".format(mask1, len(mask1_coordinates))
    )
    logger.info(
        "Aligning {} ({} atoms) with the z axis...".format(
            mask2, len(mask2_coordinates)
        )
    )

    axis = np.array([0.0, 0.0, 1.0])

    identity = np.identity(3)
    # https://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another
    # 1. Define the vector from mask1 to mask2.
    mask2_com = mask2_com + -1.0 * mask1_com

    # 2. Find axis and angle between the mask vector and the axis using cross
    # and dot products.
    try:
        x = np.cross(mask2_com, axis) / np.linalg.norm(np.cross(mask2_com, axis))
    except RuntimeWarning:
        # The structure is already aligned and the denominator is invalid
        pass

    theta = np.arccos(
        np.dot(mask2_com, axis) / (np.linalg.norm(mask2_com) * np.linalg.norm(axis))
    )
    # 3. Find the rotation matrix
    A = np.array(
        [[0, -1.0 * x[2], x[1]], [x[2], 0, -1.0 * x[0]], [-1.0 * x[1], x[0], 0]]
    )

    rotation_matrix = (
        identity
        + np.dot(np.sin(theta), A)
        + np.dot((1.0 - np.cos(theta)), np.dot(A, A))
    )

    # This is certainly not the fastest approach, but it is explicit.
    aligned_coords = np.empty_like(structure.coordinates)
    for atom in range(len(structure.atoms)):
        aligned_coords[atom] = structure.coordinates[atom] + -1.0 * mask1_com
        aligned_coords[atom] = np.dot(rotation_matrix, aligned_coords[atom])
    structure.coordinates = aligned_coords

    if save:
        if not filename:
            logger.warning(
                "Unable to save aligned coordinates (no filename provided)..."
            )
        else:
            logger.info("Saved aligned coordinates to {}".format(filename))
            # This seems to write out HETATM in place of ATOM
            # We should offer the option of writing a mol2 file, directly.
            structure.write_pdb(filename)

    return structure


def get_theta(structure, mask1, mask2, axis):
    """Get the angle (theta) between the vector formed by two masks and an axis.
    
    Parameters
    ----------
    structure : parmed.Structure
        Molecular structure containing coordinates
    mask1 : str
        Selection of first set of atoms
    mask2 : str
        Selection of second set of atoms
    axis : str
        Axis: x, y, or z
    
    Returns
    -------
    float
        The angle between the masks and the axis.
    """

    if "x" in axis.lower():
        axis = np.array([1.0, 0.0, 0.0])
    elif "y" in axis.lower():
        axis = np.array([0.0, 1.0, 0.0])
    elif "z" in axis.lower():
        axis = np.array([0.0, 0.0, 1.0])

    mask1_coordinates = structure[mask1].coordinates
    mask1_masses = [atom.mass for atom in structure[mask1].atoms]
    mask1_com = pmd.geometry.center_of_mass(
        np.asarray(mask1_coordinates), np.asarray(mask1_masses)
    )

    mask2_coordinates = structure[mask2].coordinates
    mask2_masses = [atom.mass for atom in structure[mask2].atoms]
    mask2_com = pmd.geometry.center_of_mass(
        np.asarray(mask2_coordinates), np.asarray(mask2_masses)
    )

    vector = mask2_com + -1.0 * mask1_com
    theta = np.arccos(
        np.dot(vector, axis) / (np.linalg.norm(vector) * np.linalg.norm(axis))
    )

    return theta


def check_coordinates(structure, mask):
    """Return the coordinates of an atom selection.
    
    Parameters
    ----------
    structure : parmed.Structure
        Molecular structure containing coordinates
    mask : str
        Atom selection
    
    Returns
    -------
    np.array
        Coordinates of the selection center of mass
    """

    mask_coordinates = structure[mask].coordinates
    mask_masses = [atom.mass for atom in structure[mask].atoms]
    mask_com = pmd.geometry.center_of_mass(
        np.asarray(mask_coordinates), np.asarray(mask_masses)
    )
    return mask_com


def offset_structure(structure, offset):
    """Return a structure whose coordinates have been offset.
    
    Parameters
    ----------
    structure : parmed.Structure
        Molecular structure containing coordinates
    offset : float
        The offset that will be added to *every* atom in the structure
    
    Returns
    -------
    :py:class:`parmed.Structure`
        Coordinates of the structure offset by the given amount.
    """

    offset_coords = np.empty_like(structure.coordinates)
    for atom in range(len(structure.atoms)):
        offset_coords[atom] = structure.coordinates[atom] + offset
    structure.coordinates = offset_coords
    logger.info("Added offset of {} to atomic coordinates...".format(offset))
    return structure


def align_to_axis(structure, mask=None, v_axis=None):
    """
    Aligns the 1st principal axis of a system to a given axis. This method
    mimics the method given in the link:
    https://www.ks.uiuc.edu/Research/vmd/script_library/scripts/orient/
    but does not align the 2nd or 3rd principal axes (i.e. one-step alignment only).

    Parameters
    ----------
    structure : str or parmed.Structure
        Structure we want to align.
    mask : str
        A mask that will select a particular molecule to use as the choice
        for calculating the moment of inertia.
    v_axis: list
        The axis vector to align the system to.

    Returns
    -------
    structure : parmed.Structure
        aligned structure

    """
    # Axis vector for alignment
    if v_axis is None:
        v_axis = [0, 0, 1]

    # Load structure if not already a parmed object
    if isinstance(structure, str):
        structure = pmd.load_file(f'{structure}.prmtop',
                                  f'{structure}.rst7',
                                  structure=True)

    # Reference structure for calculating the moment of inertia
    structure_ref = structure
    if mask:
        structure_ref = structure[mask]

    # Shift system to origin
    structure, masses = move_com_to_origin(structure, mask, return_mass=True)

    # Construct Inertia tensor
    Ixx = Ixy = Ixz = Iyy = Iyz = Izz = 0
    for xyz, mass in zip(structure_ref.coordinates, masses):
        Ixx += mass * (xyz[1] * xyz[1] + xyz[2] * xyz[2])
        Ixy -= mass * (xyz[0] * xyz[1])
        Ixz -= mass * (xyz[0] * xyz[2])
        Iyy += mass * (xyz[0] * xyz[0] + xyz[2] * xyz[2])
        Iyz -= mass * (xyz[1] * xyz[2])
        Izz += mass * (xyz[0] * xyz[0] + xyz[1] * xyz[1])
    inertia = np.array([[Ixx, Ixy, Ixz],
                        [Ixy, Iyy, Iyz],
                        [Ixz, Iyz, Izz]])

    # Eigen-decomposition
    evals, evecs = np.linalg.eig(inertia)
    sort_indices = [0, 1, 2]
    a1 = evecs[:, sort_indices[0]]  # 1st principal axis
    a2 = evecs[:, sort_indices[1]]  # 2nd principal axis
    a3 = evecs[:, sort_indices[2]]  # 3rd principal axis

    # Linear algebra stuff
    v_axis = np.array(v_axis)
    x = np.cross(a1, v_axis) / np.linalg.norm(np.cross(a1, v_axis))
    theta = np.arccos(np.dot(a1, v_axis) / (np.linalg.norm(a1) * np.linalg.norm(v_axis)))
    A = np.array(
        [[0, -1.0 * x[2], x[1]],
         [x[2], 0, -1.0 * x[0]],
         [-1.0 * x[1], x[0], 0]]
    )
    rotation_matrix = (
            np.identity(3)
            + np.dot(np.sin(theta), A)
            + np.dot((1.0 - np.cos(theta)), np.dot(A, A))
    )

    # Align the 1st principal axis to specified axis
    aligned_coords = np.empty_like(structure.coordinates)
    for atom in range(len(structure.atoms)):
        aligned_coords[atom] = structure.coordinates[atom]
        aligned_coords[atom] = np.dot(rotation_matrix, aligned_coords[atom])
    structure.coordinates = aligned_coords

    return structure


def move_com_to_origin(structure, mask=None, return_mass=False):
    """
    Translate the center of mass of a system to the origin.

    Parameters
    ----------
    structure : str or parmed.Structure
        Structure we want to translate.
    mask : str
        A mask that will select a particular molecule to use as the choice
        for calculating the center of mass.
    return_mass : bool
        Return the masses of the structure as well?

    Returns
    -------
    structure : parmed.Structure
        Translated structure.

    """
    structure_ref = structure
    if mask:
        structure_ref = structure[mask]

    # Get masses of atoms
    masses = [atom.mass for atom in structure_ref.atoms]

    # Shift COM to origin
    com = pmd.geometry.center_of_mass(
        structure_ref.coordinates, np.asarray(masses)
    )
    structure.coordinates -= com

    if return_mass:
        return structure, masses

    return structure


def move_anchor_to_origin(structure, center_atom, xyz_mask=None, save=False, filename=None):
    """
    Translate a system based on a single anchoring atom to the origin

    Parameters
    ----------
    structure : str or parmed.Structure
        Structure we want to translate.
    xyz_mask : str
        A mask that will filter out the dimension of the coordinate.
    save : bool
        Save the translated structure?
    filename : str
        Filename to save the structure after translating

    Returns
    -------
    structure : parmed.Structure
        Translated structure.

    """
    center_coordinates = structure[center_atom].coordinates
    center_masses = [atom.mass for atom in structure[center_atom].atoms]
    center_com = pmd.geometry.center_of_mass(
        np.asarray(center_coordinates), np.asarray(center_masses)
    )

    logger.info(
        "Centering system based on moving atom {} to the origin...".format(center_atom)
    )

    if xyz_mask is None:
        xyz_mask = [1, 1, 1]
    xyz_mask = np.array(xyz_mask)

    aligned_coords = np.empty_like(structure.coordinates)
    for atom in range(len(structure.atoms)):
        aligned_coords[atom] = structure.coordinates[atom] + -1.0 * center_com * xyz_mask
    structure.coordinates = aligned_coords

    if save:
        if not filename:
            logger.warning(
                "Unable to save aligned coordinates (no filename provided)..."
            )
        else:
            logger.info("Saved aligned coordinates to {}".format(filename))
            # This seems to write out HETATM in place of ATOM
            # We should offer the option of writing a mol2 file, directly.
            structure.write_pdb(filename)

    return structure
