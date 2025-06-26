import numpy as np


def compute_volume(bounds):
    """
    Compute the volume of an axis-aligned hyper-rectangle
    """
    # Compute the length of the interval in each dimension.
    side_lengths = bounds[:, 1] - bounds[:, 0]
    
    # Volume is the product of all side lengths.
    volume = np.prod(side_lengths)
    return volume


def get_valid_target_bounds(domain_bounds, target_bounds):
    """
    Compute the intersection of target_bounds and domain_bounds.
    """
    # Compute the valid lower and upper bounds per dimension
    valid_lower = np.maximum(domain_bounds[:, 0], target_bounds[:, 0])
    valid_upper = np.minimum(domain_bounds[:, 1], target_bounds[:, 1])
    # Check for a valid intersection in each dimension
    if np.any(valid_lower > valid_upper):
        return None
    # Stack the lower and upper bounds to form a valid bounds array.
    valid_target_bounds = np.stack([valid_lower, valid_upper], axis=1)
    return valid_target_bounds