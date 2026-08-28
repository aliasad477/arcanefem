// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* InternalBodyForceRHSVonMises.h                              (C) 2000-2026 */
/*                                                                           */
/* Contains functions to compute and assemble source term contribution to RHS*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/**
 * @brief Applies nonlinear internal body force term to RHS vector of
 * the linear system.
 * 
 * @param rhs_values The variable representing the RHS vector to be updated.
 * @param node_dof The connectivity view mapping nodes to their corresponding
 */
/*---------------------------------------------------------------------------*/

inline void FemModuleElastoplasticity::
_applyInternalBodyForceVonMises(VariableDoFReal& rhs_values, const IndexedNodeDoFConnectivityView& node_dof)
{
  auto use_gpu = options()->linearSystem.serviceName() == "HypreLinearSystem" ||
    options()->linearSystem.serviceName() == "PetscLinearSystem";

  if (use_gpu && m_use_gpu_functions) {
    auto queue = subDomain()->acceleratorMng()->defaultQueue();
    auto mesh_ptr = mesh();
    if (mesh()->dimension() == 2) {
      if (m_hex_quad_mesh) {
        ARCANE_FATAL("Not IMPLEMENTED");
      }
      else {
        ARCANE_FATAL("Not IMPLEMENTED");
      }
    } else {
      if (m_hex_quad_mesh) {
        ARCANE_FATAL("Not IMPLEMENTED");
      } else {
        ARCANE_FATAL("Not IMPLEMENTED");
      }
    }
  } else {
    if (mesh()->dimension() == 2) {
      if (m_hex_quad_mesh) {
        ARCANE_FATAL("Not IMPLEMENTED");
      } else {
        _applyInternalBodyForceVonMisesTria3Cpu(rhs_values, node_dof);
      }
    } else {
      if (m_hex_quad_mesh) {
        ARCANE_FATAL("Not IMPLEMENTED");
      } else {
        ARCANE_FATAL("Not IMPLEMENTED");
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Computes the element matrix for a triangular element (ℙ1 FE).
 *
 * Theory:
 *
 *   a(𝑈,𝐯) = ∫∫ σ(𝑈):ε(𝐯)dΩ        with  𝑈 = (𝑈𝑥,𝑈𝑦) and 𝐯 = (𝑣𝑥,𝑣𝑦)
 *   σ(𝑈) is known stress tensor    with  σᵢⱼ = Cᵢⱼₖₗεₖₗ
 *   ε(𝐯) is strain tensor          with  εᵢⱼ = 0.5 (∂𝑣ᵢ/∂xⱼ + ∂𝑣ⱼ/∂xᵢ)
 *
 *   the linear integral expands to
 *
 *      a(𝑈,𝐯) = ∫∫ [σ_𝑥𝑥 ε_𝑥𝑥 + σ_𝑦𝑦 ε_𝑦𝑦 + 2 σ_𝑥𝑦 ε_𝑥𝑦]dΩ
 *
 *   this further expands to
 *
 *      a(𝐮,𝐯) =   ∫∫ C11 ∂𝑈𝑥/∂𝑥 ∂𝑣𝑥/∂𝑥 + C12 ∂𝑈𝑦/∂𝑦 ∂𝑣𝑥/∂𝑥 + C13 (∂𝑈𝑦/∂𝑥 + ∂𝑈𝑥/∂𝑦) ∂𝑣𝑥/∂𝑥
 *               + ∫∫ C12 ∂𝑈𝑥/∂𝑥 ∂𝑣𝑦/∂𝑦 + C22 ∂𝑈𝑦/∂𝑦 ∂𝑣𝑦/∂𝑦 + C23 (∂𝑈𝑦/∂𝑥 + ∂𝑈𝑥/∂𝑦) ∂𝑣𝑥/∂𝑥
 *               + ∫∫ C13 ∂𝑈𝑥/∂𝑥 (∂𝑣𝑥/∂𝑦 + ∂𝑣𝑦/∂𝑥) + C23 ∂𝑈𝑦/∂𝑦 (∂𝑣𝑥/∂𝑦 + ∂𝑣𝑦/∂𝑥) + C33 (∂𝑈𝑦/∂𝑥 + ∂𝑈𝑥/∂𝑦)(∂𝑣𝑥/∂𝑦 + ∂𝑣𝑦/∂𝑥)
 *
 *
 */
/*---------------------------------------------------------------------------*/
inline void FemModuleElastoplasticity::
_applyInternalBodyForceVonMisesTria3Cpu(VariableDoFReal& rhs_values, const IndexedNodeDoFConnectivityView& node_dof)
{
  info() << "[ArcaneFem-Info] Started module  _applyInternalBodyForceTria3Cpu()";

  Real sq2 = math::sqrt(2.);
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    Real area = ArcaneFemFunctions::MeshOperation::computeAreaTria3(cell, m_node_coord);
    Real3 dxu = ArcaneFemFunctions::FeOperation2D::computeGradientXTria3(cell, m_node_coord);
    Real3 dyu = ArcaneFemFunctions::FeOperation2D::computeGradientYTria3(cell, m_node_coord);

    RealVector<6> eps_xx = { dxu[0], 0., dxu[1], 0., dxu[2], 0. };
    RealVector<6> eps_yy = { 0., dyu[0], 0., dyu[1], 0., dyu[2] };
    RealVector<6> eps_xy = { dyu[0], dxu[0], dyu[1], dxu[1], dyu[2], dxu[2] };
    eps_xy = M_SQRT1_2 * eps_xy;

    Int8 iGP = 0; // for tria P1 elements nGP=1
    Real sigma_xx = m_sigma_2d_gp(cell , iGP, 0);
    Real sigma_yy = m_sigma_2d_gp(cell , iGP, 1);
    Real sigma_xy = m_sigma_2d_gp(cell , iGP, 2);

    RealVector<6> rhs = - area * (sigma_xx * eps_xx + sigma_yy * eps_yy + sigma_xy * eps_xy);

    rhs_values[node_dof.dofId(cell.nodeId(0), 0)] += rhs(0);
    rhs_values[node_dof.dofId(cell.nodeId(0), 1)] += rhs(1);
    rhs_values[node_dof.dofId(cell.nodeId(1), 0)] += rhs(2);
    rhs_values[node_dof.dofId(cell.nodeId(1), 1)] += rhs(3);
    rhs_values[node_dof.dofId(cell.nodeId(2), 0)] += rhs(4);
    rhs_values[node_dof.dofId(cell.nodeId(2), 1)] += rhs(5);
  }
}