
def compute_bin_year(year, time_frame_start, time_frame_end, bin_size):
    """
    Given an input year, the start and end of a time frame, and bin size,
    computes which bin start date this year belongs to.
    """
    return ((year - time_frame_start) // bin_size) * bin_size + time_frame_start
