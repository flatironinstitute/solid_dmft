################################################################################
#
# solid_dmft - A versatile python wrapper to perform DFT+DMFT calculations
#              utilizing the TRIQS software library
#
# Copyright (C) 2018-2020, ETH Zurich
# Copyright (C) 2021, The Simons Foundation
#      authors: A. Hampel, M. Merkel, and S. Beck
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

import triqs.utility.mpi as mpi
import importlib.util


# check which modules are available and import only those ones

interfaces_dict = {}


check_solver = importlib.util.find_spec("triqs_cthyb") is not None
if check_solver:
    from solid_dmft.dmft_tools.solvers.cthyb_interface import CTHYBInterface
    interfaces_dict['cthyb'] = CTHYBInterface

check_solver = importlib.util.find_spec("triqs_ctint") is not None
if check_solver:
    from solid_dmft.dmft_tools.solvers.ctint_interface import CTINTInterface
    interfaces_dict['ctint'] = CTINTInterface

check_solver = importlib.util.find_spec("triqs_ctseg") is not None
if check_solver:
    from solid_dmft.dmft_tools.solvers.ctseg_interface import CTSEGInterface
    interfaces_dict['ctseg'] = CTSEGInterface

check_solver = importlib.util.find_spec("triqs_hartree_fock") is not None
if check_solver:
    from solid_dmft.dmft_tools.solvers.hartree_interface import HartreeInterface
    interfaces_dict['hartree'] = HartreeInterface

check_solver = importlib.util.find_spec("triqs_hubbardI") is not None
if check_solver:
    from solid_dmft.dmft_tools.solvers.hubbardI_interface import HubbardIInterface
    interfaces_dict['hubbardI'] = HubbardIInterface

check_solver = importlib.util.find_spec("forktps") is not None
if check_solver:
    from solid_dmft.dmft_tools.solvers.ftps_interface import FTPSInterface
    interfaces_dict['ftps'] = FTPSInterface

# generic dispatch function for the solver classes
def create_solver(general_params, solver_params, sum_k, icrsh, h_int, iteration_offset,
                    deg_orbs_ftps, gw_params=None, advanced_params=None):
        '''
        Dispatch the solver to the correct subclass

        Returns
        -------
        solver: subclass of AbstractDMFTSolver
                instance of the correct solver subclass
        '''
        if solver_params['type'] in interfaces_dict.keys():
            mpi.report(f"Using {solver_params['type']} solver")
            solver_interface = interfaces_dict[solver_params['type']] 
            return solver_interface(general_params, solver_params, sum_k, icrsh, h_int,
                                iteration_offset, deg_orbs_ftps, gw_params, advanced_params)
        else:
            raise ValueError(f"Unknown solver type {solver_params['type']}")

