
def compute_bin_year(year: int, time_frame_start: int, bin_size: int):
    """
    Given an input year, the start of a time frame, and bin size,
    computes which bin start date this year belongs to.
    """
    return ((year - time_frame_start) // bin_size) * bin_size + time_frame_start
