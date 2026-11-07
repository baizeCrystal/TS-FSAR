"""Parameter-name boundaries shared by DTO solver and gradient diagnostics."""


def is_lsn_parameter(name):
    """Real LSN uses model.side_*; model.side.* supports tiny legacy fixtures.

    This identifies candidates only. The solver additionally requires a
    trainable parameter connected to both objectives, including numerical zeros.
    Heads (up_linear/up_ln/Linear), teacher Adapter and TSM stay outside LSN.
    """
    return name.startswith(('model.side_', 'model.side.'))
