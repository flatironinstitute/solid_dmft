# -*- coding: utf-8 -*-
################################################################################
#
# solid_dmft - A versatile python wrapper to perform DFT+DMFT calculations
#              utilizing the TRIQS software library
#
# Copyright (C) 2018-2020, ETH Zurich
# Copyright (C) 2021-2024, The Simons Foundation
#      authors: A. Hampel,
#
# solid_dmft is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# solid_dmft is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# solid_dmft (in the file COPYING.txt in this directory). If not, see
# <http://www.gnu.org/licenses/>.
#
################################################################################
# pyright: reportUnusedExpression=false
"""
Module for gw embedding tools
"""

import numpy as np

import triqs.utility.mpi as mpi

from triqs.gf import Gf, BlockGf, MeshDLRImFreq, make_gf_dlr_imfreq, make_gf_dlr

def calc_Pi_imp_NN(chi_imp_dlr, U_loc_dlr):
    """
    Calculate the impurity polarization function from the impurity susceptibility
    and the local interaction in the density density approximation

    Parameters
    ----------
    chi_imp_dlr : Gf on DLR mesh
        impurity susceptibility 2*norb x 2*norb

    U_loc_dlr : Gf on DLR mesh
        local interaction 2*norb x 2*norb

    Returns
    -------
    Pi_dlr_iw : Gf on DLR mesh
        impurity polarization function density-density approximation 2*norb x 2*norb

    """

    # make sure we are on the DLR ImFreq mesh
    if not isinstance(chi_imp_dlr.mesh, MeshDLRImFreq):
        chi_imp_dlr = make_gf_dlr_imfreq(chi_imp_dlr)

    if not isinstance(U_loc_dlr.mesh, MeshDLRImFreq):
        U_loc_dlr = make_gf_dlr_imfreq(U_loc_dlr)

    Pi_dlr_iw = Gf(mesh=chi_imp_dlr.mesh, target_shape=chi_imp_dlr.target_shape)

    norb = chi_imp_dlr.target_shape[0]
    ones = np.eye(norb, dtype=complex)

    for iwn in chi_imp_dlr.mesh:
        denom = U_loc_dlr[iwn] @ chi_imp_dlr[iwn] - ones
        Pi_dlr_iw[iwn] = chi_imp_dlr[iwn] @ np.linalg.inv(denom)

    return Pi_dlr_iw
