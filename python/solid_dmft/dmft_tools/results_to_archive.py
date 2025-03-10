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

import os
import triqs.utility.mpi as mpi


def _compile_information(sum_k, general_params, solver_params, solvers, map_imp_solver, solver_type_per_imp,
                         previous_mu, density_mat_pre, density_mat, deltaN, dens):
    """ Collects all results in a dictonary. """

    write_to_h5 = {'chemical_potential_post': sum_k.chemical_potential,
                   'chemical_potential_pre': previous_mu,
                   'DC_pot': sum_k.dc_imp,
                   'DC_energ': sum_k.dc_energ,
                   'dens_mat_pre': density_mat_pre,
                   'dens_mat_post': density_mat,
                  }

    if deltaN is not None:
        write_to_h5['deltaN'] = deltaN
    if dens is not None:
        write_to_h5['deltaN_trace'] = dens

    if any(str(entry) in ('crpa_dynamic') for entry in general_params['dc_type']):
        write_to_h5['DC_pot_dyn'] = sum_k.dc_imp_dyn

    for icrsh in range(sum_k.n_inequiv_shells):
        isolvsec = map_imp_solver[icrsh]
        if solver_type_per_imp[icrsh] in ['cthyb', 'hubbardI']:
            write_to_h5['Delta_time_{}'.format(icrsh)] = solvers[icrsh].Delta_time

            # Write the full density matrix to last_iter only - it is large
            if solver_params[isolvsec]['measure_density_matrix']:
                write_to_h5['full_dens_mat_{}'.format(icrsh)] = solvers[icrsh].density_matrix
                write_to_h5['h_loc_diag_{}'.format(icrsh)] = solvers[icrsh].h_loc_diagonalization
                if solver_type_per_imp[icrsh] in ('cthyb','hubbardI'):
                    write_to_h5['Sigma_moments_{}'.format(icrsh)] = solvers[icrsh].Sigma_moments
                    write_to_h5['G_moments_{}'.format(icrsh)] = solvers[icrsh].G_moments
                    write_to_h5['Sigma_Hartree_{}'.format(icrsh)] = solvers[icrsh].Sigma_Hartree

        elif solver_type_per_imp[icrsh] == 'ftps':
            write_to_h5['Delta_freq_{}'.format(icrsh)] = solvers[icrsh].Delta_freq

        write_to_h5['Gimp_time_{}'.format(icrsh)] = solvers[icrsh].G_time
        write_to_h5['G0_freq_{}'.format(icrsh)] = solvers[icrsh].G0_freq
        write_to_h5['Gimp_freq_{}'.format(icrsh)] = solvers[icrsh].G_freq
        write_to_h5['Sigma_freq_{}'.format(icrsh)] = solvers[icrsh].Sigma_freq

        if solver_type_per_imp[icrsh] == 'cthyb':
            if solver_params[isolvsec]['measure_pert_order']:
                write_to_h5['pert_order_imp_{}'.format(icrsh)] = solvers[icrsh].perturbation_order
                write_to_h5['pert_order_total_imp_{}'.format(icrsh)] = solvers[icrsh].perturbation_order_total

            if solver_params[isolvsec]['measure_chi'] is not None:
                write_to_h5['O_{}_time_{}'.format(solver_params[isolvsec]['measure_chi'], icrsh)] = solvers[icrsh].O_time

            # if legendre was set, that we have both now!
            if (solver_params[isolvsec]['measure_G_l']
                or not solver_params[isolvsec]['perform_tail_fit'] and solver_params[isolvsec]['legendre_fit']):
                write_to_h5['G_time_orig_{}'.format(icrsh)] = solvers[icrsh].G_time_orig
                write_to_h5['Gimp_l_{}'.format(icrsh)] = solvers[icrsh].G_l

            if solver_params[isolvsec]['crm_dyson_solver']:
                write_to_h5['G_time_dlr_{}'.format(icrsh)] = solvers[icrsh].G_time_dlr
                write_to_h5['Sigma_dlr_{}'.format(icrsh)] = solvers[icrsh].Sigma_dlr

        if solver_type_per_imp[icrsh] == 'ctint' and solver_params[isolvsec]['measure_histogram']:
            write_to_h5['pert_order_imp_{}'.format(icrsh)] = solvers[icrsh].perturbation_order

        if solver_type_per_imp[icrsh] == 'hubbardI':
            write_to_h5['G0_Refreq_{}'.format(icrsh)] = solvers[icrsh].G0_Refreq
            write_to_h5['Gimp_Refreq_{}'.format(icrsh)] = solvers[icrsh].G_Refreq
            write_to_h5['Sigma_Refreq_{}'.format(icrsh)] = solvers[icrsh].Sigma_Refreq

            if solver_params[isolvsec]['measure_G_l']:
                write_to_h5['Gimp_l_{}'.format(icrsh)] = solvers[icrsh].G_l

        if solver_type_per_imp[icrsh] == 'hartree':
            write_to_h5['Sigma_Refreq_{}'.format(icrsh)] = solvers[icrsh].Sigma_Refreq

        if solver_type_per_imp[icrsh] == 'ctseg':
            # if legendre was set, that we have both now!
            if (solver_params[isolvsec]['legendre_fit']):
                write_to_h5['G_time_orig_{}'.format(icrsh)] = solvers[icrsh].G_time_orig
                write_to_h5['Gimp_l_{}'.format(icrsh)] = solvers[icrsh].G_l
            if solver_params[isolvsec]['improved_estimator']:
                write_to_h5['F_freq_{}'.format(icrsh)] = solvers[icrsh].F_freq
                write_to_h5['F_time_{}'.format(icrsh)] = solvers[icrsh].F_time
            if solver_params[isolvsec]['measure_pert_order']:
                write_to_h5['pert_order_histo_imp_{}'.format(icrsh)] = solvers[icrsh].perturbation_order_histo
                write_to_h5['avg_order_imp_{}.format(icrsh)'] = solvers[icrsh].avg_pert_order
            if solver_params[isolvsec]['measure_nn_tau']:
                write_to_h5['O_NN_{}'.format(icrsh)] = solvers[icrsh].triqs_solver.results.nn_tau
            if solver_params[isolvsec]['measure_state_hist']:
                write_to_h5['state_hist_{}'.format(icrsh)] = solvers[icrsh].state_histogram
            if solver_params[isolvsec]['crm_dyson_solver']:
                write_to_h5['G_time_dlr_{}'.format(icrsh)] = solvers[icrsh].G_time_dlr
                write_to_h5['Sigma_dlr_{}'.format(icrsh)] = solvers[icrsh].Sigma_dlr
                write_to_h5['Sigma_Hartree_{}'.format(icrsh)] = solvers[icrsh].Sigma_Hartree
            if general_params['h_int_type'][icrsh] == 'dyn_density_density':
                write_to_h5['D0_time_{}'.format(icrsh)] = solvers[icrsh].triqs_solver.D0_tau
                write_to_h5['Jperp_time_{}'.format(icrsh)] = solvers[icrsh].triqs_solver.Jperp_tau

    return write_to_h5

def write(archive, sum_k, general_params, solver_params, solvers, map_imp_solver, solver_type_per_imp, it, is_sampling,
          previous_mu, density_mat_pre, density_mat, deltaN=None, dens=None):
    """
    Collects and writes results to archive.
    """

    if not mpi.is_master_node():
        return

    write_to_h5 = _compile_information(sum_k, general_params, solver_params, solvers, map_imp_solver, solver_type_per_imp,
                                       previous_mu, density_mat_pre, density_mat, deltaN, dens)

    # Saves the results to last_iter
    archive['DMFT_results']['iteration_count'] = it
    for key, value in write_to_h5.items():
        archive['DMFT_results/last_iter'][key] = value

    # Permanently saves to h5 archive every h5_save_freq iterations
    if ((not is_sampling and it % general_params['h5_save_freq'] == 0)
            or (is_sampling and it % general_params['sampling_h5_save_freq'] == 0)):

        archive['DMFT_results'].create_group('it_{}'.format(it))
        for key, value in write_to_h5.items():
            # Full density matrix only written to last_iter - it is large
            if 'full_dens_mat_' not in key and 'h_loc_diag_' not in key:
                archive['DMFT_results/it_{}'.format(it)][key] = value

        # Saves CSC input
        if general_params['csc']:
            for dft_var in ['dft_update', 'dft_input', 'dft_misc_input']:
                if dft_var in archive:
                    archive['DMFT_results/it_{}'.format(it)].create_group(dft_var)
                    for key, value in archive[dft_var].items():
                        # do only store changing elements
                        if key not in ['symm_kpath', 'kpts_cart']:
                            archive['DMFT_results/it_{}'.format(it)][dft_var][key] = value
            for band_elem in ['_bands.dat', '_bands.dat.gnu', '_bands.projwfc_up', '_band.dat']:
                if os.path.isfile('./{}{}'.format(general_params['seedname'], band_elem)):
                    os.rename('./{}{}'.format(general_params['seedname'], band_elem),
                              './{}{}_it{}'.format(general_params['seedname'], band_elem, it))
            for w90_elem in ['_hr.dat', '.wout']:
                if os.path.isfile('./{}{}'.format(general_params['seedname'], w90_elem)):
                    os.rename('./{}{}'.format(general_params['seedname'], w90_elem),
                              './{}_it{}{}'.format(general_params['seedname'], it, w90_elem))
