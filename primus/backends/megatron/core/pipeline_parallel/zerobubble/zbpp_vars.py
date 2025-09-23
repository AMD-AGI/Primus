_SEQ_SPLIT_IDX = None


def get_seq_split_idx():
    """Get the sequence split index"""
    global _SEQ_SPLIT_IDX
    return _SEQ_SPLIT_IDX


def set_seq_split_idx(idx):
    """Set the sequence split index"""
    global _SEQ_SPLIT_IDX
    _SEQ_SPLIT_IDX = idx


# need to call when inprocess_restart
def destroy_zbpp_vars():
    global _SEQ_SPLIT_IDX
    _SEQ_SPLIT_IDX = None
