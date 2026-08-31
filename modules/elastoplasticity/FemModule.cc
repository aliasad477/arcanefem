// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FemModule.cc                                                (C) 2000-2026 */
/*                                                                           */
/* FEM code for Elastoplasticity problem.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/core/IParallelMng.h>
#include <arcane/accelerator/MDVariableViews.h>

#include "FemModule.h"
#include "ElementMatrix.h"
#include "ElementMatrixHexQuad.h"
#include "ExternalBodyForce.h"
#include "Traction.h"
#include "Dirichlet.h"
#include "InternalBodyForce.h"
#include "InternalBodyForceVonMises.h"

/*---------------------------------------------------------------------------*/
/**
 * @brief Initializes the FemModuleElastoplasticity at the start of the simulation.
 *
 * This method initializes degrees of freedom (DoFs) on nodes.
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
startInit()
{
  info() << "[ArcaneFem-Info] Started module  startInit()";
  Real elapsedTime = platform::getRealTime();

  tmax = options()->tmax(); // max time 𝑡ₘₐₓ
  dt = options()->dt(); // time step δ𝑡

  E = options()->E(); // Youngs modulus
  nu = options()->nu(); // Poission ratio ν
  sig0 = options()->sig0(); // Yield Strength

  m_dof_per_node = defaultMesh()->dimension();
  m_matrix_format = options()->matrixFormat();
  m_assemble_linear_system = options()->assembleLinearSystem();
  m_solve_linear_system = options()->solveLinearSystem();
  m_cross_validation = options()->hasSolutionComparisonFile();
  m_petsc_flags = options()->petscFlags();
  m_hex_quad_mesh = options()->hexQuadMesh();

  m_dofs_on_nodes.initialize(defaultMesh(), m_dof_per_node);

  m_nonlinear_law = options()->nonlinearLaw();
  m_newton_max_iters = options()->newtonMaxIters();
  m_newton_atol = options()->newtonAtol();
  m_newton_rtol = options()->newtonRtol();

  m_use_gpu_functions =   (m_matrix_format == "BSR" || m_matrix_format == "AF-BSR") &&
                          (options()->linearSystem.serviceName() == "HypreLinearSystem" ||
                           options()->linearSystem.serviceName() == "PetscLinearSystem" ||
                           options()->linearSystem.serviceName() == "AlephLinearSystem");

  m_gp_material_tensor_strategy = options()->gpMaterialTensorStrategy();
  m_check_with_bilinear_operator = options()->checkBilinearOperatorForResidual();

  // The native von Mises update stores one algorithmic tangent per integration
  // point. Tria3 has one integration point, so a cell variable is sufficient.
  if (m_nonlinear_law)
    m_gp_material_tensor_strategy = "global";

  if (m_gp_material_tensor_strategy == "global") {
    if (mesh()->dimension() == 2) {
      m_C_tang_2d_cell.reshape({ 3, 3 });
    } else {
      m_C_tang_3d_cell.reshape({ 6, 6 });
    }
  }

  if (m_nonlinear_law) {

    if (mesh()->dimension() != 2 || m_hex_quad_mesh)
      ARCANE_FATAL("Native von Mises plasticity currently supports only 2D Tria3 elements");

    if (mesh()->dimension() == 2) {

      m_nGP = 1;

      m_epsilon_2d_gp.reshape({m_nGP, 3});
      m_sigma_2d_gp.reshape({m_nGP, 3});
      m_sigma_old_2d_gp.reshape({m_nGP, 3});
      m_sigma_trial_2d_gp.reshape({m_nGP, 3});
      m_dev_2d_gp.reshape({m_nGP, 3});
      m_flowN_2d_gp.reshape({m_nGP, 3});

      m_sigma_zz_2d_gp.reshape({m_nGP});
      m_sigma_zz_old_2d_gp.reshape({m_nGP});
      m_p_old_2d_gp.reshape({m_nGP});
      m_dp_2d_gp.reshape({m_nGP});

    }
  }

  t = dt;
  tmax = tmax - dt;
  m_global_deltat.assign(dt);

  _readCaseTables();

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"initialize", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Performs the main computation for the FemModuleElastoplasticity.
 *
 * This method:
 *   1. Stops the time loop after 1 iteration since the equation is steady state.
 *   2. Resets, configures, and initializes the linear system.
 *   3. Sets Petsc flags if user has provided them.
 *   4. Executes the stationary solve.
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
compute()
{
  info() << "[ArcaneFem-Info] Started module  compute()";
  Real elapsedTime = platform::getRealTime();

  // Stop code after computations
  // if (m_global_iteration() > 0)
  //   subDomain()->timeLoopMng()->stopComputeLoop(true);
  // Stop the computation loop if the maximum time is reached
  if (t >= tmax)
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  info() << "[ArcaneFem-Info] Time iteration at t : " << t << " (s) ";
  bool keep_struct = true;
  if (m_linear_system.isInitialized() && keep_struct) {
    m_linear_system.clearValues();
  } else {
    m_linear_system.reset();
    m_linear_system.setLinearSystemFactory(options()->linearSystem());
    m_linear_system.initialize(subDomain(), acceleratorMng()->defaultRunner(), m_dofs_on_nodes.dofFamily(), "Solver");
  }

  if (m_petsc_flags != NULL){
    CommandLineArguments args = ArcaneFemFunctions::GeneralFunctions::getPetscFlagsFromCommandline(m_petsc_flags);
    m_linear_system.setSolverCommandLineArguments(args);
  }

  if (m_matrix_format == "BSR" || m_matrix_format == "AF-BSR")
    _initBsr();

  Int64 nb_node = mesh()->ownNodes().size();
  Int64 total_nb_node = mesh()->parallelMng()->reduce(Parallel::ReduceSum, nb_node);

  Int64 nb_face = mesh()->outerFaces().size();
  Int64 total_nb_boundary_elt = mesh()->parallelMng()->reduce(Parallel::ReduceSum, nb_face);

  Int64 nb_cell = mesh()->ownCells().size();
  Int64 total_nb_elt = mesh()->parallelMng()->reduce(Parallel::ReduceSum, nb_cell);

  info() << "[ArcaneFem-Info] mesh dimension " << defaultMesh()->dimension();
  info() << "[ArcaneFem-Info] mesh boundary elements " << total_nb_boundary_elt;
  info() << "[ArcaneFem-Info] mesh cells " << total_nb_elt;
  info() << "[ArcaneFem-Info] mesh nodes " << total_nb_node;

  _doStationarySolve();
  _updateTime();

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"compute", elapsedTime);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_updateTime()
{
  t += dt;
}


/*---------------------------------------------------------------------------*/
/**
 * @brief Initializes BSR matrix.
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_initBsr()
{
  info() << "[ArcaneFem-Info] Started module  _initBsr()";
  Real elapsedTime = platform::getRealTime();

  bool use_csr_in_linearsystem =
  options()->linearSystem.serviceName() == "HypreLinearSystem" ||
  options()->linearSystem.serviceName() == "AlienLinearSystem" ||
  options()->linearSystem.serviceName() == "PetscLinearSystem";

  if (m_matrix_format == "BSR")
    m_bsr_format.initialize(defaultMesh(), m_dof_per_node, use_csr_in_linearsystem, 0);
  else
    m_bsr_format.initialize(defaultMesh(), m_dof_per_node, use_csr_in_linearsystem, 1);

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"initialize-bsr-matrix", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Performs a stationary solve for the FEM system.
 *
 * This method does the following
 *   1. Solves the FEM system either with:
 *      _solveNewton()
 *      or
 *      _solveLinear()
 *      based on the nature of the constitutive law
 *
 *   2. _validateResults()           Regression test
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_doStationarySolve()
{
  if (m_nonlinear_law) {
    _solveNewton();
  }
  else {
    _solveLinear();
  }

  if(m_cross_validation){
    if (t > 0. && t==tmax)
    _validateResults();
  }

}

/*---------------------------------------------------------------------------*/
/**
 * @brief Performs a linear solve for the FEM system using Newton method.
 *
 * This method follows a sequence of steps to solve FEM system:
 *
 *   1. _getMaterialParameters()     Updates nonlinear material parameters
 *   2. _assembleBilinearOperator()  Assembles the FEM  matrix 𝐀
 *   3. _assembleLinearOperator()    Assembles the FEM RHS vector 𝐛
 *   4. _solve()                     Solves for solution vector 𝐮 = 𝐀⁻¹𝐛
 *   5. _updateVariables()           Updates FEM variables 𝐮 = 𝐱
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_solveLinear()
{
  info() << "[ArcaneFem-Info] Started module  _solveLinear()";
  _getMaterialParameters();

  if(m_assemble_linear_system){
    _assembleBilinearOperator();
    _assembleLinearOperator();
  }
  if(m_solve_linear_system){
    _solve();
    _updateVariables();
  }
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Performs a nonlinear solve for the FEM system using Newton method.
 *
 * This method follows a sequence of steps to solve FEM system:
 *
 *   1. _getMaterialParameters()     Updates nonlinear material parameters
 *   2. _assembleBilinearOperator()  Assembles the FEM  matrix 𝐀ʹ
 *   3. _assembleLinearOperator()    Assembles the FEM RHS vector 𝐛
 *   4. _solve()                     Solves for solution vector 𝐝𝐮 = 𝐀ʹ⁻¹𝐛
 *   5. _updateNewtonIncrements()    Updates FEM variables 𝐝𝐮 = 𝐱
 *   5. _checkNewtonConvergence()    Check convergence on norm of 𝐝𝐮 /𝐮
 *   6. _validateResults()           Regression test
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_solveNewton()
{
  if (!m_assemble_nonlinear_system)
    return;

  info() << "[ArcaneFem-Info] Started module  _solveNewton()";

  _getMaterialParameters();

  // --- initialize_increment ---- //
  m_DU.fill({0., 0., 0.});
  m_dU.fill({0., 0., 0.});
  m_newton_iter = 0;

  // --- restore_converged_state ---- //
  ENUMERATE_ (Cell, icell, allCells())
  {
    Cell cell = *icell;

    for (Int8 iGP = 0; iGP < m_nGP; ++iGP ) {
      m_sigma_2d_gp(cell, iGP, 0) = m_sigma_old_2d_gp(cell, iGP, 0);
      m_sigma_2d_gp(cell, iGP, 1) = m_sigma_old_2d_gp(cell, iGP, 1);
      m_sigma_2d_gp(cell, iGP, 2) = m_sigma_old_2d_gp(cell, iGP, 2);
    }

    for (Int8 ix = 0; ix < 3; ++ix)
      for (Int8 iy = 0; iy < 3; ++iy)
        m_C_tang_2d_cell(cell, ix, iy) = m_C_2d(ix, iy); // set tangent C equal to elastic C
  }

  // --- assemble_linear_system ---- //
  if (m_assemble_linear_system) {
    _assembleBilinearOperator();
    _assembleLinearOperator();
  }

  // --- calculate_residual ---- //
  VariableDoFReal& residual_values(m_linear_system.rhsVariable());
  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());
  // _applyDirichlet0(residual_values, node_dof);
  m_residual_norm0 = _norm_l2(residual_values, node_dof);
  info() << "[ArcaneFem-Info] Initial residual norm = " << m_residual_norm0;


  // --- start_newton_loop ---- //
  while (m_newton_iter < m_newton_max_iters && !m_newton_solver_converged) {

    m_newton_iter++;

    // --- solve_linear_system ---- //
    if(m_solve_nonlinear_system){
      _solve();
      _updateNewtonIncrements();
    }

    // --- update_increment ---- //
    _incrementVariables();

    ENUMERATE_ (Cell, icell, allCells())
    {
      Cell cell = *icell;

      for (Int8 iGP = 0; iGP < m_nGP; ++iGP ) {

        // --- compute_trial_state ---- //
        // computeTrialStateVM();
        // epsilon(DU)
        Real3x3 grad_DU = ArcaneFemFunctions::FeOperation2D::FeOperation2D::computeGradientTria3(cell, m_node_coord, m_DU);
        Real eps_xx = grad_DU(0, 0);
        Real eps_yy = grad_DU(1, 1);
        Real eps_xy = M_SQRT1_2 * (grad_DU(0, 1) + grad_DU(1, 0));

        //info() << "[ArcaneFem-Info] eps_xx = " << eps_xx << " eps_yy = " << eps_yy << " eps_xy = " << eps_xy;

        m_epsilon_2d_gp(cell, iGP, 0) = eps_xx;
        m_epsilon_2d_gp(cell, iGP, 1) = eps_yy;
        m_epsilon_2d_gp(cell, iGP, 2) = eps_xy;

        //info() << "[ArcaneFem-Info] eps_xx = " << eps_xx << " eps_yy = " << eps_yy << " eps_xy = " << eps_xy;

        Real sigma_trial_xx = m_sigma_old_2d_gp(cell, iGP, 0) + m_C_2d(0, 0) * eps_xx + m_C_2d(0, 1) * eps_yy + m_C_2d(0, 2) * eps_xy;
        Real sigma_trial_yy = m_sigma_old_2d_gp(cell, iGP, 1) + m_C_2d(1, 0) * eps_xx + m_C_2d(1, 1) * eps_yy + m_C_2d(1, 2) * eps_xy;
        Real sigma_trial_xy = m_sigma_old_2d_gp(cell, iGP, 2) + m_C_2d(2, 0) * eps_xx + m_C_2d(2, 1) * eps_yy + m_C_2d(2, 2) * eps_xy;

        Real sigma_trial_zz = m_sigma_zz_old_2d_gp(cell, iGP) + lambda * (eps_xx + eps_yy);

        m_sigma_trial_2d_gp(cell, iGP, 0) = sigma_trial_xx;
        m_sigma_trial_2d_gp(cell, iGP, 1) = sigma_trial_yy;
        m_sigma_trial_2d_gp(cell, iGP, 2) = sigma_trial_xy;

        // Plane strain retains sigma_zz in the three-dimensional deviator.
        Real sigma_trial_mean = (sigma_trial_xx + sigma_trial_yy + sigma_trial_zz) / 3.0;

        Real dev_xx = m_sigma_trial_2d_gp(cell, iGP, 0) - sigma_trial_mean;
        Real dev_yy = m_sigma_trial_2d_gp(cell, iGP, 1) - sigma_trial_mean;
        Real dev_xy = m_sigma_trial_2d_gp(cell, iGP, 2);

        Real dev_zz = sigma_trial_zz - sigma_trial_mean;

        m_dev_2d_gp(cell, iGP, 0) = dev_xx;
        m_dev_2d_gp(cell, iGP, 1) = dev_yy;
        m_dev_2d_gp(cell, iGP, 2) = dev_xy;

        //info() << "[ArcaneFem-Info] dev_xx = " << dev_xx << " dev_yy = " << dev_yy << " dev_xy = " << dev_xy;

        Real sigma_eq_trial = math::sqrt(1.5 * (dev_xx * dev_xx + dev_yy * dev_yy + dev_zz * dev_zz + dev_xy * dev_xy) );

        //info() << "[ArcaneFem-Info] sigma_eq_trial = " << sigma_eq_trial;

        // --- evaluate_yield_function ---- //
        // _computeYieldFunctionVM();
        Real yield_function = sigma_eq_trial - sig0 - H * m_p_old_2d_gp(cell, iGP);
        Real yield_positive = (yield_function + math::abs(yield_function)) / 2.;
        m_dp_2d_gp(cell, iGP) = yield_positive/ (3. * mu + H);
        Real plastic_switch = yield_positive / (math::abs(yield_function) + 1e-14 * sig0);

        // --- radial_return_update ---- //
        // _computeRadialReturnVM();
        Real flowN_xx = plastic_switch * dev_xx / (sigma_eq_trial + 1e-14 * sig0);
        Real flowN_yy = plastic_switch * dev_yy / (sigma_eq_trial + 1e-14 * sig0);
        Real flowN_xy = plastic_switch * dev_xy / (sigma_eq_trial + 1e-14 * sig0);
        Real flowN_zz = plastic_switch * dev_zz / (sigma_eq_trial + 1e-14 * sig0);

        Real beta = 3. * mu * m_dp_2d_gp(cell, iGP) / (sigma_eq_trial + 1e-14 * sig0);

        m_flowN_2d_gp(cell, iGP, 0) = flowN_xx;
        m_flowN_2d_gp(cell, iGP, 1) = flowN_yy;
        m_flowN_2d_gp(cell, iGP, 2) = flowN_xy;

        //info() << "[ArcaneFem-Info] flowN_xx = " << flowN_xx << " flowN_yy = " << flowN_yy << " flowN_xy = " << flowN_xy;
        //info() << "[ArcaneFem-Info] flowN_zz = " << flowN_zz;
        //info() << "[ArcaneFem-Info] beta = " << beta;

        // --- update_consistent_tangent ---- //
        // _updateStressTensorVM();
        Real sigma_xx = m_sigma_trial_2d_gp(cell, iGP, 0) - dev_xx * beta;
        Real sigma_yy = m_sigma_trial_2d_gp(cell, iGP, 1) - dev_yy * beta;
        Real sigma_xy = m_sigma_trial_2d_gp(cell, iGP, 2) - dev_xy * beta;

        Real sigma_zz = sigma_trial_zz - dev_zz * beta;

        m_sigma_2d_gp(cell, iGP, 0) = sigma_xx;
        m_sigma_2d_gp(cell, iGP, 1) = sigma_yy;
        m_sigma_2d_gp(cell, iGP, 2) = sigma_xy;

        m_sigma_zz_2d_gp(cell, iGP) = sigma_zz;

        //info() << "[ArcaneFem-Info] sigma_xx = " << sigma_xx << " sigma_yy = " << sigma_yy << " sigma_xy = " << sigma_xy;
        //info() << "[ArcaneFem-Info] sigma_zz = " << sigma_zz;

        // _updateTangentMaterialTensorVM(); // NOTE if not we store everything and assemble locally with at element matrix assembly i.e., local gp technique.
        Real tangentA = 3.* mu * (3. * mu / (3. * mu + H) - beta);
        //info() << "[ArcaneFem-Info] tangentA = " << tangentA;

        // Consistent algorithmic tangent for the radial-return update:
        m_C_tang_2d_cell(cell, 0, 0) = m_C_2d(0, 0) - tangentA * flowN_xx * flowN_xx - 4. * mu * beta / 3.;
        m_C_tang_2d_cell(cell, 0, 1) = m_C_2d(0, 1) - tangentA * flowN_xx * flowN_yy + 2. * mu * beta / 3.;
        m_C_tang_2d_cell(cell, 0, 2) = m_C_2d(0, 2) - tangentA * flowN_xx * flowN_xy;

        //info() << "[ArcaneFem-Info] C_tang row 1 done ";

        m_C_tang_2d_cell(cell, 1, 0) = m_C_2d(1, 0) - tangentA * flowN_xx * flowN_yy + 2. * mu * beta / 3.;
        m_C_tang_2d_cell(cell, 1, 1) = m_C_2d(1, 1) - tangentA * flowN_yy * flowN_yy - 4. * mu * beta / 3.;
        m_C_tang_2d_cell(cell, 1, 2) = m_C_2d(1, 2) - tangentA * flowN_yy * flowN_xy;

        //info() << "[ArcaneFem-Info] C_tang row 2 done ";

        m_C_tang_2d_cell(cell, 2, 0) = m_C_2d(2, 0) - tangentA * flowN_xx * flowN_xy;
        m_C_tang_2d_cell(cell, 2, 1) = m_C_2d(2, 1) - tangentA * flowN_yy * flowN_xy;
        m_C_tang_2d_cell(cell, 2, 2) = m_C_2d(2, 2) - tangentA * flowN_xy * flowN_xy - 2. * mu * beta;

        //info() << "[ArcaneFem-Info] C_tang row 3 done ";
      }
    }

    // --- assemble_linear_system ---- //
    if(m_assemble_nonlinear_system) {
      if (m_linear_system.isInitialized()) {
        m_linear_system.clearValues();

        if (m_matrix_format == "BSR" || m_matrix_format == "AF-BSR")
          m_bsr_format.resetMatrixValues();

        _assembleBilinearOperator(); // assembles Jacobian
        _assembleLinearOperator(); // assembles Residuals(m_DU) + BCs
      }
    }

    // --- calculate_residual ---- //
    _checkNewtonConvergence();

  }

  if (m_newton_solver_converged) {
    info() << "[ArcaneFem-Info] Newton solver converged after " << m_newton_iter << " iterations.";

    if (t == dt) {
      Real Ri = 1.0;
      Real Re = 1.3;
      Qlim = 2./math::sqrt(3.) * math::log( Re/Ri) * sig0;
    }
    Real tl = math::sqrt(1.1 / tmax * (t));

    info() << "[ArcaneFem-Info] At Time Step " << t - 1 << ":\tPressure applied: " << Qlim * tl << "\tNewton iters: " << m_newton_iter << "\tresidual norm: " << m_residual_norm;

    m_newton_solver_converged = false;
    m_newton_iter = 0;
  }

  if (m_newton_iter == m_newton_max_iters && !m_newton_solver_converged) {
    info() << "[ArcaneFem-Info] Newton iterations did not converge after maximum (" << m_newton_max_iters << ") iterations";
    ARCANE_FATAL("Newton iterations diverged after max iters");
  }

  // --- commit_displacements ---- //
  m_U.synchronize();
  m_DU.synchronize();
  ENUMERATE_ (Node, inode, ownNodes()) {
    m_U[inode] += m_DU[inode];
  }
  m_U.synchronize();

  // --- commit_internal_variables ---- //
  ENUMERATE_ (Cell, icell, allCells())
  {
    Cell cell = *icell;

    for (Int8 iGP = 0; iGP < m_nGP; ++iGP ) {
      m_sigma_old_2d_gp(cell, iGP, 0) = m_sigma_2d_gp(cell, iGP, 0);
      m_sigma_old_2d_gp(cell, iGP, 1) = m_sigma_2d_gp(cell, iGP, 1);
      m_sigma_old_2d_gp(cell, iGP, 2) = m_sigma_2d_gp(cell, iGP, 2);

      m_sigma_zz_old_2d_gp(cell, iGP) = m_sigma_zz_2d_gp(cell, iGP);
      m_p_old_2d_gp(cell, iGP) += m_dp_2d_gp(cell, iGP);
    }
  }


}

/*---------------------------------------------------------------------------*/
/**
 * @brief Retrieves and sets the material parameters for the simulation.
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_getMaterialParameters()
{
  info() << "[ArcaneFem-Info] Started module  _getMaterialParameters()";
  Real elapsedTime = platform::getRealTime();

  mu = (E / (2 * (1 + nu))); // lame parameter μ
  lambda = E * nu / ((1 + nu) * (1 - 2 * nu)); // lame parameter λ

  // Elastic material tensors
  /*
 {{lambda + 2. * mu, lambda,           0.},
  {lambda,           lambda + 2. * mu, 0.},
  {0.,               0.,               2 * mu}}
*/ // 2D elastic material tensor

  /*
   {{lambda + 2.*mu, lambda,         lambda,         0.,  0.,  0.},
    {lambda,         lambda + 2.*mu, lambda,         0.,  0.,  0.},
    {lambda,         lambda,         lambda + 2.*mu, 0.,  0.,  0.},
    {0.,             0.,             0.,             2mu,  0.,  0.},
    {0.,             0.,             0.,             0.,  2mu,  0.},
    {0.,             0.,             0.,             0.,  0.,  2mu}}
  */ // 3D elastic material tensor

  if (mesh()->dimension() == 2) {
    m_C_2d.fill(0.);
    m_C_2d(0, 0) = lambda + 2. * mu;
    m_C_2d(1, 1) = lambda + 2. * mu;
    m_C_2d(2, 2) = 2. * mu;
    m_C_2d(0, 1) = lambda;
    m_C_2d(1, 0) = lambda;
  } else {
    m_C_3d.fill(0.);
    m_C_3d(0, 0) = lambda + 2. * mu;
    m_C_3d(1, 1) = lambda + 2. * mu;
    m_C_3d(2, 2) = lambda + 2. * mu;
    m_C_3d(3, 3) = 2 * mu;
    m_C_3d(4, 4) = 2 * mu;
    m_C_3d(5, 5) = 2 * mu;
    m_C_3d(0, 1) = lambda;
    m_C_3d(1, 0) = lambda;
    m_C_3d(0, 2) = lambda;
    m_C_3d(2, 0) = lambda;
    m_C_3d(1, 2) = lambda;
    m_C_3d(2, 1) = lambda;
  }

  if (m_nonlinear_law) { // set von Mises params
    Et = E / 100.;
    H = E * Et / (E - Et);

    // Constitutive history is initialized once. Calling this routine at the
    // next load step must not erase the converged stress and plastic strain.
    if (!m_material_initialized && mesh()->dimension() == 2) {
      ENUMERATE_ (Cell, icell, allCells())
      {
        for (Int8 iGP = 0; iGP < m_nGP; ++iGP ) {
          m_sigma_2d_gp(icell, iGP, 0) = 0.;
          m_sigma_2d_gp(icell, iGP, 1) = 0.;
          m_sigma_2d_gp(icell, iGP, 2) = 0.;
          m_sigma_zz_2d_gp(icell, iGP) = 0.;

          m_sigma_old_2d_gp(icell, iGP, 0) = 0.;
          m_sigma_old_2d_gp(icell, iGP, 1) = 0.;
          m_sigma_old_2d_gp(icell, iGP, 2) = 0.;
          m_sigma_zz_old_2d_gp(icell, iGP) = 0.;

          m_sigma_trial_2d_gp(icell, iGP, 0) = 0.;
          m_sigma_trial_2d_gp(icell, iGP, 1) = 0.;
          m_sigma_trial_2d_gp(icell, iGP, 2) = 0.;

          m_p_old_2d_gp(icell, iGP) = 0.;
          m_dp_2d_gp(icell, iGP) = 0.;
        }
      }
    } else if (mesh()->dimension() != 2) {
      ARCANE_FATAL("Not implemented yet");
    }
  }



  if (m_gp_material_tensor_strategy == "local") {
     if (mesh()->dimension() == 2) {
       m_C_tang_2d = m_C_2d;
     } else {
       m_C_tang_3d = m_C_3d;
     }
  } else {
    if (mesh()->dimension() == 2) {
      ENUMERATE_ (Cell, icell, allCells()) {
        for (Int8 ix = 0; ix < 3; ++ix) {
          for (Int8 iy = 0; iy < 3; ++iy) {
            m_C_tang_2d_cell(icell, ix, iy) = m_C_2d(ix, iy);
          }
        }
      }
    } else {
      ENUMERATE_ (Cell, icell, allCells()) {
        for (Int8 ix = 0; ix < 6; ++ix) {
          for (Int8 iy = 0; iy < 6; ++iy) {
            m_C_tang_3d_cell(icell,ix, iy) = m_C_3d(ix, iy);
          }
        }
      }
    }
  }
  m_material_initialized = true;
  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"get-material-params", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/*---------------------------------------------------------------------------*/
/**
 * @brief Assemble the FEM linear operator.
 *
 * This method follows a sequence of steps to assemble RHS of FEM linear system:
 *
 *   1. assembles the bodyforce contribution (source term) ∫∫∫ (𝐟.𝐯) on Ω
 *   2. assembles the traction contribution (Neumann term) ∫∫ (𝐭.𝐯)  on ∂Ω
 *   3. apply Dirichlet contributions to LHS and RHS
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_assembleLinearOperator()
{
  info() << "[ArcaneFem-Info] Started module  _assembleLinearOperator()";
  Real elapsedTime = platform::getRealTime();

  VariableDoFReal& rhs_values(m_linear_system.rhsVariable()); // Temporary variable to keep values for the RHS
  rhs_values.fill(0.0);

  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

  _applyExternalBodyForce(rhs_values, node_dof);
  _applyTraction(rhs_values, node_dof);

  if (m_nonlinear_law) {
    _applyInternalBodyForceVonMises(rhs_values, node_dof);
    _applyDirichletNewton(rhs_values, node_dof);
   } else {
    _applyDirichlet(rhs_values, node_dof);
  }

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"rhs-vector-assembly", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Calls the right function for LHS assembly given as mesh type.
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_assembleBilinearOperator()
{
  info() << "[ArcaneFem-Info] Started module  _assembleBilinearOperator()";
  Real elapsedTime = platform::getRealTime();

  if (m_matrix_format == "BSR") {
    UnstructuredMeshConnectivityView m_connectivity_view(mesh());
    auto cn_cv = m_connectivity_view.cellNode();
    auto command = makeCommand(acceleratorMng()->defaultQueue());
    auto in_node_coord = Accelerator::viewIn(command, m_node_coord);
    auto in_C_tang_2d = Accelerator::viewIn(command, m_C_tang_2d_cell);
    auto in_C_tang_3d = Accelerator::viewIn(command, m_C_tang_2d_cell); // Maybe should be m_C_tang_3d_cell instead of m_C_tang_2d_cell

    Real lambda_cell = lambda;
    Real mu_cell = mu;
    RealVector<2> hooke_params = {lambda_cell, mu_cell};

    auto C_tang_3d = m_C_tang_3d;

    m_bsr_format.computeSparsity();
    if (mesh()->dimension() == 2) {
      if (m_gp_material_tensor_strategy == "local") {
        m_bsr_format.assembleBilinearAtomic([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid) { return computeHookeElementMatrixTria3Gpu(cell_lid, cn_cv, in_node_coord, hooke_params); });
      } else {
        m_bsr_format.assembleBilinearAtomic([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid) { return computeElementMatrixTria3Gpu(cell_lid, cn_cv, in_node_coord, in_C_tang_2d); });
      }
    }
    else {
      m_bsr_format.assembleBilinearAtomic([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid) { return computeElementMatrixTetra4Gpu(cell_lid, cn_cv, in_node_coord, C_tang_3d); });
    }
    m_bsr_format.toLinearSystem(m_linear_system);
  } else if (m_matrix_format == "AF-BSR") {
    UnstructuredMeshConnectivityView m_connectivity_view(mesh());
    auto cn_cv = m_connectivity_view.cellNode();
    auto command = makeCommand(acceleratorMng()->defaultQueue());
    auto in_node_coord = Accelerator::viewIn(command, m_node_coord);
    auto in_C_tang_2d = Accelerator::viewIn(command, m_C_tang_2d_cell);
    auto in_C_tang_3d = Accelerator::viewIn(command, m_C_tang_2d_cell); // Maybe should be m_C_tang_3d_cell instead of m_C_tang_2d_cell

    Real lambda_cell = lambda;
    Real mu_cell = mu;
    RealVector<2> hooke_params = {lambda_cell, mu_cell};
    auto C_tang_3d = m_C_tang_3d;

    m_bsr_format.computeSparsity();
    if (mesh()->dimension() == 2) {
      if (m_gp_material_tensor_strategy == "local") {
        m_bsr_format.assembleBilinearAtomicFree([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid, Int32 node_lid) { return computeHookeElementVectorTria3Gpu(cell_lid, cn_cv, in_node_coord, hooke_params, node_lid); });
      } else {
        m_bsr_format.assembleBilinearAtomicFree([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid, Int32 node_lid) { return computeElementVectorTria3Gpu(cell_lid, cn_cv, in_node_coord, in_C_tang_2d, node_lid); });
      }
    } else {
      m_bsr_format.assembleBilinearAtomicFree([=] ARCCORE_HOST_DEVICE(CellLocalId cell_lid, Int32 node_lid) { return computeElementVectorTetra4Gpu(cell_lid, cn_cv, in_node_coord, C_tang_3d, node_lid); });
    }
    m_bsr_format.toLinearSystem(m_linear_system);
  } else if (m_matrix_format == "DOK") {
    if (mesh()->dimension() == 2) {
      if (m_hex_quad_mesh) {
        _assembleBilinearOperatorCpu<8>([this](const Cell& cell) { return _computeElementMatrixQuad4(cell); });
      }
      else {
        _assembleBilinearOperatorCpu<6>([&](const Cell& cell) { return _computeElementMatrixTria3(cell); });
      }
    }
    if (mesh()->dimension() == 3) {
      if (m_hex_quad_mesh) {
        _assembleBilinearOperatorCpu<24>([this](const Cell& cell) { return _computeElementMatrixHexa8(cell); });
      }
      else {
        _assembleBilinearOperatorCpu<12>([this](const Cell& cell) { return _computeElementMatrixTetra4(cell); });
      }
    }
  } else {
    ARCANE_FATAL("Unsupported matrix type, only DOK| BSR|AF-BSR is supported.");
  }

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"lhs-matrix-assembly", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Assembles the FEM bilinear operator on CPU.
 *
 * This method assembles the FEM stiffness matrix by iterating over each cell,
 * computing the element stiffness matrix using the provided function, and
 * populating the global stiffness matrix accordingly.
 *
 * @tparam N Total DOF size (nodes_per_element × dimensions).
 * @param compute_element_matrix function computing cell's element stiffness matrix.
 */
/*---------------------------------------------------------------------------*/

template <int N>
void FemModuleElastoplasticity::
_assembleBilinearOperatorCpu(const std::function<RealMatrix<N, N>(const Cell&)>& compute_element_matrix)
{
  const Int32 dim = mesh()->dimension();
  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    auto K_e = compute_element_matrix(cell);

    Int32 n1_index = 0;
    for (Node node1 : cell.nodes()) {
      if (node1.isOwn()) {
        Int32 n2_index = 0;
        for (Node node2 : cell.nodes()) {
          for (Int32 i = 0; i < dim; ++i) {
            DoFLocalId dof1 = node_dof.dofId(node1, i);
            for (Int32 j = 0; j < dim; ++j) {
              DoFLocalId dof2 = node_dof.dofId(node2, j);
              Real value = K_e(dim * n1_index + i, dim * n2_index + j);
              m_linear_system.matrixAddValue(dof1, dof2, value);
            }
          }
          ++n2_index;
        }
      }
      ++n1_index;
    }
  }
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Solves the linear system.
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_solve()
{
  info() << "[ArcaneFem-Info] Started module  _solve()";
  Real elapsedTime = platform::getRealTime();

  m_linear_system.applyLinearSystemTransformationAndSolve();

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"solve-linear-system", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_validateResults()
{
  info() << "[ArcaneFem-Info] Started module  _validateResults()";
  Real elapsedTime = platform::getRealTime();

  ENUMERATE_ (Node, inode, allNodes()) {
    Node node = *inode;
    info() << "U["<< node.uniqueId() << "] = " << m_DU[node].x << " " << m_DU[node].y << " " << m_DU[node].z;
  }

  String filename = options()->solutionComparisonFile();
  const double epsilon = options()->resultEpsilon();
  const double min_value_to_test = 1.0e-10;

  Arcane::FemUtils::checkNodeResultFile(traceMng(), filename, m_DU, epsilon, min_value_to_test);

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"result-validation", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/*
  * @brief Reads case tables for traction boundary conditions.
  *
  * This method reads the case tables specified in the options and stores
  * them in a list for later use.
  */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_readCaseTables()
{
  IParallelMng* pm = subDomain()->parallelMng();
  BC::IArcaneFemBC* bc = options()->boundaryConditions();

  // loop over all traction boundries
  for (BC::ITractionBoundaryCondition* bs : bc->tractionBoundaryConditions()) {
    CaseTable* case_table = nullptr;
    auto traction_table_file_name = bs->getTractionInputFile();
    bool getTractionFromTable = !traction_table_file_name.empty();
    if (getTractionFromTable)
      case_table = readFileAsCaseTable(pm, traction_table_file_name, 3);
    m_traction_case_table_list.add(CaseTableInfo{ traction_table_file_name, case_table });
  }
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Update the FEM variables.
 *
 * This method performs the following actions:
 *   1. Fetches values of solution from solved linear system to FEM variables,
 *      i.e., it copies RHS DOF to u.
 *   2. Performs synchronize of FEM variables across subdomains.
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_updateVariables()
{
  info() << "[ArcaneFem-Info] Started module  _updateVariables()";
  Real elapsedTime = platform::getRealTime();

  {
    VariableDoFReal& dof_u(m_linear_system.solutionVariable());
    auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());
    if (mesh()->dimension() == 3)
      ENUMERATE_ (Node, inode, ownNodes()) {
        Node node = *inode;
        Real u1_val = dof_u[node_dof.dofId(node, 0)];
        Real u2_val = dof_u[node_dof.dofId(node, 1)];
        Real u3_val = dof_u[node_dof.dofId(node, 2)];
        m_DU[node] = Real3(u1_val, u2_val, u3_val);
      }
    else
      ENUMERATE_ (Node, inode, ownNodes()) {
        Node node = *inode;
        Real u1_val = dof_u[node_dof.dofId(node, 0)];
        Real u2_val = dof_u[node_dof.dofId(node, 1)];
        m_DU[node] = Real3(u1_val, u2_val, 0.);
      }
  }

  m_DU.synchronize();

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"update-variables", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Update the FEM Newton increment.
 *
 * This method performs the following actions:
 *   1. Fetches values of solution from solved linear system to FEM variables,
 *      i.e., it copies RHS DOF to du.
 *   2. Performs synchronize of FEM variables across subdomains.
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_updateNewtonIncrements()
{
  info() << "[ArcaneFem-Info] Started module  _updateNewtonIncrements()";
  Real elapsedTime = platform::getRealTime();

  {
    VariableDoFReal& dof_du(m_linear_system.solutionVariable());
    auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());
    if (mesh()->dimension() == 3)
      ENUMERATE_ (Node, inode, ownNodes()) {
        Node node = *inode;
        Real du1_val = dof_du[node_dof.dofId(node, 0)];
        Real du2_val = dof_du[node_dof.dofId(node, 1)];
        Real du3_val = dof_du[node_dof.dofId(node, 2)];
        m_dU[node] = Real3(du1_val, du2_val, du3_val);
      }
    else
      ENUMERATE_ (Node, inode, ownNodes()) {
        Node node = *inode;
        Real du1_val = dof_du[node_dof.dofId(node, 0)];
        Real du2_val = dof_du[node_dof.dofId(node, 1)];
        m_dU[node] = Real3(du1_val, du2_val, 0.);
      }
  }
  m_dU.synchronize();

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(),"update-Newton-increments", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Reinitialize the solution vector of the linear solve with the FEM variables.
 *
 * This method performs the following actions:
 *   1. Performs synchronization of FEM increment variables across subdomains.
 *   2. Fetches the FEM increment variables to the solution vector of the
 *      linear solver for the next nonlinear solver iteration.
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_updateGuessFromIncrement()
{
  info() << "[ArcaneFem-Info] Started module _updateGuessFromIncrement()";
  Real elapsedTime = platform::getRealTime();

  m_dU.synchronize();

  {
    VariableDoFReal& dof_du(m_linear_system.solutionVariable());
    auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());
    if (mesh()->dimension() == 3)
      ENUMERATE_ (Node, inode, ownNodes()) {
      Node node = *inode;
      dof_du[node_dof.dofId(node, 0)] = m_dU[node][0];
      dof_du[node_dof.dofId(node, 1)] = m_dU[node][1];
      dof_du[node_dof.dofId(node, 2)] = m_dU[node][2];
    }
    else
      ENUMERATE_ (Node, inode, ownNodes()) {
      Node node = *inode;
      dof_du[node_dof.dofId(node, 0)] = m_dU[node][0];
      dof_du[node_dof.dofId(node, 1)] = m_dU[node][1];
    }
  }

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(), "_update-guess-from-increment", elapsedTime);
}
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/**
 * @brief Increments the FEM variables.
 *
 * This method updates the FEM solutions with the increment of
 * the current Newton iteration
 *
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::
_incrementVariables()
{
  info() << "[ArcaneFem-Info] Started module _incrementVariables()";
  Real elapsedTime = platform::getRealTime();

  m_dU.synchronize();
  m_DU.synchronize();
  {
      ENUMERATE_ (Node, inode, ownNodes()) {
      m_DU[inode] += m_dU[inode];
    }
  }
  m_DU.synchronize();

  elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(), "increment-fem-variables", elapsedTime);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Check for the convergence of nonlinear solver.
 *
 * This method performs the following actions:
 *   1. Evaluates the convergence norm with Newton increment FEM variables
 *   2. Checks for convergence
 *
 */
/*---------------------------------------------------------------------------*/
void FemModuleElastoplasticity::
_checkNewtonConvergence()
{
  info() << "[ArcaneFem-Info] Started module _checkNewtonConvergence()";
  Real elapsedTime = platform::getRealTime();

  m_dU.synchronize();
  m_DU.synchronize();

  Real l2_norm_du = _norm_l2(m_dU);
  Real l2_norm_u = _norm_l2(m_DU);

  m_increment_norm = l2_norm_u != 0.0 ? l2_norm_du / l2_norm_u : 1.0;
  Real convergence_error_increment = l2_norm_du / (m_newton_rtol * l2_norm_u  + m_newton_atol);

  VariableDoFReal& residual_values(m_linear_system.rhsVariable());
  auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());
  _applyDirichlet0(residual_values, node_dof);

  Real l2_norm_rhs = _norm_l2(residual_values, node_dof);

  m_residual_norm = l2_norm_rhs!=0 ? l2_norm_rhs / m_residual_norm0 : 1.0;
  Real convergence_error_residual = l2_norm_rhs / (m_residual_norm0 + 1e-30);

  // The OR criterion follows petsc SNES
  if (convergence_error_residual <= m_newton_rtol) {
    m_newton_solver_converged = true;
    m_newton_converged_reason = "RESIDUAL_CONVERGED";
    info() << "[ArcaneFem-Info] At Newton iteration " << m_newton_iter
           << ": ||X_k+1 - X_k||/||X_k+1|| = " << m_increment_norm
           << " ||(F_ext - F_int(X_k))||/||(F_ext - F_int(X_0))|| = " << m_residual_norm
           << " => " << "CONVERGED with " << m_newton_converged_reason;
  } else if (convergence_error_increment <= 1.0) {
    m_newton_solver_converged = true;
    m_newton_converged_reason = "INCREMENT_CONVERGED";
    info() << "[ArcaneFem-Info] At Newton iteration " << m_newton_iter
           << ": ||X_k+1 - X_k||/||X_k+1|| = " << m_increment_norm
           << " ||(F_ext - F_int(X_k))||/||(F_ext - F_int(X_0))|| = " << m_residual_norm
           << " => " << "CONVERGED with " << m_newton_converged_reason;
  } else {
    m_newton_solver_converged = false;
    info() << "[ArcaneFem-Info] At Newton iteration " << m_newton_iter
           << ": ||X_k+1 - X_k||/||X_k+1|| = " << m_increment_norm
           << " ||(F_ext - F_int(X_k))||/||(F_ext - F_int(X_0))|| = " << m_residual_norm
           << " => " << "NOT CONVERGED";
  }

    elapsedTime = platform::getRealTime() - elapsedTime;
  ArcaneFemFunctions::GeneralFunctions::printArcaneFemTime(traceMng(), "check-newton-convergence", elapsedTime);
}

inline Real FemModuleElastoplasticity::
_norm_l2(VariableNodeReal3& u) {
  Real l2_norm_u = 0.0;
  {
    ENUMERATE_ (Node, inode, ownNodes()) {
      const Real norm_u = math::pow(u[inode][0], 2.0) + math::pow(u[inode][1], 2.0) + math::pow(u[inode][2], 2.0);
      l2_norm_u += norm_u;
    }
  }
  IParallelMng* pm = defaultMesh()->parallelMng();
  l2_norm_u = pm->reduce(Parallel::ReduceSum, l2_norm_u);

  return math::sqrt(l2_norm_u);
}

inline Real FemModuleElastoplasticity::
_norm_l2(VariableDoFReal& u, const IndexedNodeDoFConnectivityView& node_dof) {
  Real l2_norm_u = 0.0;
  {
    ENUMERATE_ (Node, inode, ownNodes()) {
      Real norm_residual = 0.0;
      if (mesh()->dimension() == 2) {
        norm_residual =  math::pow(u[node_dof.dofId(inode, 0)], 2.0)
                  + math::pow(u[node_dof.dofId(inode, 1)], 2.0);
      } else {
        norm_residual =  math::pow(u[node_dof.dofId(inode, 0)], 2.0)
                  + math::pow(u[node_dof.dofId(inode, 1)], 2.0)
                  + math::pow(u[node_dof.dofId(inode, 2)], 2.0);
      }
      l2_norm_u += norm_residual;
    }
  }
  IParallelMng* pm = defaultMesh()->parallelMng();
  l2_norm_u = pm->reduce(Parallel::ReduceSum, l2_norm_u);
  return math::sqrt(l2_norm_u);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_FEM(FemModuleElastoplasticity);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
