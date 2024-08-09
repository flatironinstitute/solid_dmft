# miscellanous functions for the DMFT calculation used in other modules

def get_n_orbitals(sum_k):
    """
    determines the number of orbitals within the
    solver block structure.

    Parameters
    ----------
    sum_k : dft_tools sumk object

    Returns
    -------
    n_orb : dict of int
        number of orbitals for up / down as dict for SOC calculation
        without up / down block up holds the number of orbitals
    """
    n_orbitals = [{'up': 0, 'down': 0} for i in range(sum_k.n_inequiv_shells)]
    for icrsh in range(sum_k.n_inequiv_shells):
        for block, n_orb in sum_k.gf_struct_solver[icrsh].items():
            if 'down' in block:
                n_orbitals[icrsh]['down'] += sum_k.gf_struct_solver[icrsh][block]
            else:
                n_orbitals[icrsh]['up'] += sum_k.gf_struct_solver[icrsh][block]

    return n_orbitals