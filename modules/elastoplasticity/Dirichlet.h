// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Dirichlet.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Contains functions to compute and assemble dirichlet contribution to RHS  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/**
 * @brief Applies dirichlet to LHS matrix and RHS vector of the nonlinear system.
 *
 * This function applies Dirichlet boundary conditions to both the LHS matrix
 * and RHS vector of the nonlinear system.
 *
 * @param rhs_values The variable representing the RHS vector to be updated.
 * @param node_dof The connectivity view mapping nodes to their corresponding
 */
/*---------------------------------------------------------------------------*/
inline void FemModuleElastoplasticity::
_applyDirichletNewton(VariableDoFReal& rhs_values, const IndexedNodeDoFConnectivityView& node_dof)
{
  // check if Hypre|Petsc solver is used and delegate to GPU for dirichlet assembly
  auto use_gpu = options()->linearSystem.serviceName() == "HypreLinearSystem" ||
    options()->linearSystem.serviceName() == "PetscLinearSystem";
  if (use_gpu && m_use_gpu_functions) {
    _assembleDirichletsNewtonGpu();
    return;
  }

  info() << "[ArcaneFem-Info] Started module _assembleDirichletsNewtonCpu()";

  BC::IArcaneFemBC* bc = options()->boundaryConditions();
  if (bc) {
    for (BC::IDirichletBoundaryCondition* bs : bc->dirichletBoundaryConditions()) {
      FaceGroup face_group = bs->getSurface();
      NodeGroup node_group = face_group.nodeGroup();
      const StringConstArrayView u_dirichlet_string = bs->getValue();
      for (Int32 dof_index = 0; dof_index < u_dirichlet_string.size(); ++dof_index) {
        if (u_dirichlet_string[dof_index] != "NULL") {

          Real value = std::stod(u_dirichlet_string[dof_index].localstr());
          if (bs->getEnforceDirichletMethod() == "Penalty") {
            Real penalty = bs->getPenalty();
            ENUMERATE_ (Node, inode, node_group) {
              Node node = *inode;
              if (node.isOwn()) {
                m_linear_system.matrixSetValue(node_dof.dofId(node, dof_index), node_dof.dofId(node, dof_index), penalty);
                Real u_g = penalty * (value - m_DUn[node][dof_index]);
                rhs_values[node_dof.dofId(node, dof_index)] = u_g;
              }
            }
          }
          else if (bs->getEnforceDirichletMethod() == "RowElimination") {
            Real value0 = 0.0;
            ArcaneFemFunctions::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowElimination(dof_index, value0, node_dof, m_linear_system, rhs_values, node_group);
          }
          else if (bs->getEnforceDirichletMethod() == "RowColumnElimination") {
            Real value0 = 0.0;
            ArcaneFemFunctions::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowColumnElimination(dof_index, value0, node_dof, m_linear_system, rhs_values, node_group);
          }
          else {
            ARCANE_FATAL("Unknown Dirichlet method");
          }
        }
      }
    }

    for (BC::IDirichletPointCondition* bs : bc->dirichletPointConditions()) {
      NodeGroup node_group = bs->getNode();
      const StringConstArrayView u_dirichlet_string = bs->getValue();
      for (Int32 dof_index = 0; dof_index < u_dirichlet_string.size(); ++dof_index) {
        if (u_dirichlet_string[dof_index] != "NULL") {
          Real value = 0.0;
          if (m_newton_iter == 0) {
            value = std::stod(u_dirichlet_string[dof_index].localstr());
          }
          if (bs->getEnforceDirichletMethod() == "Penalty") {
            Real penalty = bs->getPenalty();
            ArcaneFemFunctions::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaPenalty(dof_index, value, penalty, node_dof, m_linear_system, rhs_values, node_group);
          }
          else if (bs->getEnforceDirichletMethod() == "RowElimination") {
            ArcaneFemFunctions::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowElimination(dof_index, value, node_dof, m_linear_system, rhs_values, node_group);
          }
          else if (bs->getEnforceDirichletMethod() == "RowColumnElimination") {
            ArcaneFemFunctions::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowColumnElimination(dof_index, value, node_dof, m_linear_system, rhs_values, node_group);
          }
          else {
            ARCANE_FATAL("Unknown Dirichlet method");
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Applies dirichlet to LHS matrix and RHS vector of the nonlinear system on Gpu.
 *
 * This function applies Dirichlet boundary conditions to both the LHS matrix
 * and RHS vector of the nonlinear system using GPU acceleration.
 *
 * @param rhs_values The variable representing the RHS vector to be updated.
 * @param node_dof The connectivity view mapping nodes to their corresponding
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::_assembleDirichletsNewtonGpu()
{
  info() << "[ArcaneFem-Info] Started module  _assembleDirichletsNewtonGpu()";

  auto queue = subDomain()->acceleratorMng()->defaultQueue();
  auto mesh_ptr = mesh();

  BC::IArcaneFemBC* bc = options()->boundaryConditions();

  if (bc) {
    for (BC::IDirichletBoundaryCondition* bs : bc->dirichletBoundaryConditions()) {
      ARCANE_CHECK_PTR(bs);

      FaceGroup face_group = bs->getSurface();
      NodeGroup node_group = face_group.nodeGroup();

      const StringConstArrayView u_dirichlet_string = bs->getValue();

      for (Int32 dof_index = 0; dof_index < u_dirichlet_string.size(); ++dof_index) {
        if (u_dirichlet_string[dof_index] != "NULL") {
          Real value = std::stod(u_dirichlet_string[dof_index].localstr());
          if (bs->getEnforceDirichletMethod() == "Penalty") {
            Real penalty = bs->getPenalty();
            ARCANE_CHECK_PTR(queue);
            ARCANE_CHECK_PTR(mesh_ptr);

            NodeInfoListView nodes_infos(mesh_ptr->nodeFamily());
            auto node_dof(m_dofs_on_nodes.nodeDoFConnectivityView());

            auto command = makeCommand(queue);
            auto in_out_forced_info = viewInOut(command, m_linear_system.getForcedInfo());
            auto in_out_forced_value = viewInOut(command, m_linear_system.getForcedValue());
            auto in_out_rhs_variable = viewInOut(command, m_linear_system.rhsVariable());
            auto in_u = viewIn(command, m_DUn);

            command << RUNCOMMAND_ENUMERATE(NodeLocalId, node_lid, node_group)
            {
              if (nodes_infos.isOwn(node_lid)) {
                DoFLocalId dof_id = node_dof.dofId(node_lid, dof_index);
                in_out_forced_info[dof_id] = true;
                in_out_forced_value[dof_id] = penalty;
                in_out_rhs_variable[dof_id] = penalty * (value - in_u[node_lid][dof_index]);
              }
            };
          }
          else if (bs->getEnforceDirichletMethod() == "RowElimination") {
            Real value0 = 0.0;
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowOrRowColumnElimination(ELIMINATE_ROW, dof_index, value0, queue, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else if (bs->getEnforceDirichletMethod() == "RowColumnElimination") {
            Real value0 = 0.0;
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowOrRowColumnElimination(ELIMINATE_ROW_COLUMN, dof_index, value0, queue, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else {
            ARCANE_FATAL("Unknown method to enforce Dirichlet BC: '{0}'", bs->getEnforceDirichletMethod());
          }
        }
      }
    }

    for (BC::IDirichletPointCondition* bs : bc->dirichletPointConditions()) {
      ARCANE_CHECK_PTR(bs);
      NodeGroup node_group = bs->getNode();

      const StringConstArrayView u_dirichlet_str = bs->getValue();

      for (Int32 dof_index = 0; dof_index < u_dirichlet_str.size(); ++dof_index) {
        if (u_dirichlet_str[dof_index] != "NULL") {
          Real value = 0.0;
          if (m_newton_iter == 0) {
            value = std::stod(u_dirichlet_str[dof_index].localstr());
          }

          if (bs->getEnforceDirichletMethod() == "Penalty") {
            Real penalty = bs->getPenalty();
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaPenalty(dof_index, value, penalty, queue, mesh_ptr, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else if (bs->getEnforceDirichletMethod() == "RowElimination") {
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowOrRowColumnElimination(ELIMINATE_ROW, dof_index, value, queue, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else if (bs->getEnforceDirichletMethod() == "RowColumnElimination") {
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowOrRowColumnElimination(ELIMINATE_ROW_COLUMN, dof_index, value, queue, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else {
            ARCANE_FATAL("Unknown method to enforce Dirichlet BC: '{0}'", bs->getEnforceDirichletMethod());
          }
        }
      }
    }
   }
}



/*---------------------------------------------------------------------------*/
/**
 * @brief Applies dirichlet to LHS matrix and RHS vector of the nonlinear system.
 *
 * This function applies Dirichlet boundary conditions to both the LHS matrix
 * and RHS vector of the nonlinear system.
 *
 * @param rhs_values The variable representing the RHS vector to be updated.
 * @param node_dof The connectivity view mapping nodes to their corresponding
 */
/*---------------------------------------------------------------------------*/
inline void FemModuleElastoplasticity::
_applyDirichlet0(VariableDoFReal& rhs_values, const IndexedNodeDoFConnectivityView& node_dof)
{
  // check if Hypre|Petsc solver is used and delegate to GPU for dirichlet assembly
  // auto use_gpu = options()->linearSystem.serviceName() == "HypreLinearSystem" ||
  //   options()->linearSystem.serviceName() == "PetscLinearSystem";
  // if (use_gpu && m_use_gpu_functions) {
  //   _assembleDirichlets0Gpu();
  //   return;
  //   return;
  // }

  //info() << "[ArcaneFem-Info] Started module _assembleDirichletsNewtonCpu()";
  // Explicitly remove reaction components so the norm contains free DoFs only.
  BC::IArcaneFemBC* bc = options()->boundaryConditions();
  if (bc) {
    for (BC::IDirichletBoundaryCondition* bs : bc->dirichletBoundaryConditions()) {
      FaceGroup face_group = bs->getSurface();
      NodeGroup node_group = face_group.nodeGroup();
      const StringConstArrayView u_dirichlet_string = bs->getValue();
      for (Int32 dof_index = 0; dof_index < u_dirichlet_string.size(); ++dof_index) {
        if (u_dirichlet_string[dof_index] != "NULL") {
          ENUMERATE_ (Node, inode, node_group) {
            if (inode->isOwn())
              rhs_values[node_dof.dofId(*inode, dof_index)] = 0.0;
          }
        }
      }
    }

    for (BC::IDirichletPointCondition* bs : bc->dirichletPointConditions()) {
      NodeGroup node_group = bs->getNode();
      const StringConstArrayView u_dirichlet_string = bs->getValue();
      for (Int32 dof_index = 0; dof_index < u_dirichlet_string.size(); ++dof_index) {
        if (u_dirichlet_string[dof_index] != "NULL") {
          ENUMERATE_ (Node, inode, node_group) {
            if (inode->isOwn())
              rhs_values[node_dof.dofId(*inode, dof_index)] = 0.0;
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Applies dirichlet to LHS matrix and RHS vector of the nonlinear system on Gpu.
 *
 * This function applies Dirichlet boundary conditions to both the LHS matrix
 * and RHS vector of the nonlinear system using GPU acceleration.
 *
 * @param rhs_values The variable representing the RHS vector to be updated.
 * @param node_dof The connectivity view mapping nodes to their corresponding
 */
/*---------------------------------------------------------------------------*/

void FemModuleElastoplasticity::_assembleDirichlets0Gpu()
{
  info() << "[ArcaneFem-Info] Started module  _assembleDirichletsNewtonGpu()";

  auto queue = subDomain()->acceleratorMng()->defaultQueue();
  auto mesh_ptr = mesh();

  BC::IArcaneFemBC* bc = options()->boundaryConditions();

  if (bc) {
    for (BC::IDirichletBoundaryCondition* bs : bc->dirichletBoundaryConditions()) {
      ARCANE_CHECK_PTR(bs);

      FaceGroup face_group = bs->getSurface();
      NodeGroup node_group = face_group.nodeGroup();

      const StringConstArrayView u_dirichlet_string = bs->getValue();

      for (Int32 dof_index = 0; dof_index < u_dirichlet_string.size(); ++dof_index) {
        if (u_dirichlet_string[dof_index] != "NULL") {
          if (bs->getEnforceDirichletMethod() == "Penalty") {
            Real penalty = bs->getPenalty();
            Real value0 = 0.0;
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaPenalty(dof_index, value0, penalty, queue, mesh_ptr, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else if (bs->getEnforceDirichletMethod() == "RowElimination") {
            Real value0 = 0.0;
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowOrRowColumnElimination(ELIMINATE_ROW, dof_index, value0, queue, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else if (bs->getEnforceDirichletMethod() == "RowColumnElimination") {
            Real value0 = 0.0;
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowOrRowColumnElimination(ELIMINATE_ROW_COLUMN, dof_index, value0, queue, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else {
            ARCANE_FATAL("Unknown method to enforce Dirichlet BC: '{0}'", bs->getEnforceDirichletMethod());
          }
        }
      }
    }

    for (BC::IDirichletPointCondition* bs : bc->dirichletPointConditions()) {
      ARCANE_CHECK_PTR(bs);
      NodeGroup node_group = bs->getNode();

      const StringConstArrayView u_dirichlet_str = bs->getValue();

      for (Int32 dof_index = 0; dof_index < u_dirichlet_str.size(); ++dof_index) {
        if (u_dirichlet_str[dof_index] != "NULL") {
          Real value = 0.0;
          if (m_newton_iter == 0) {
            value = std::stod(u_dirichlet_str[dof_index].localstr());
          }

          if (bs->getEnforceDirichletMethod() == "Penalty") {
            Real penalty = bs->getPenalty();
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaPenalty(dof_index, value, penalty, queue, mesh_ptr, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else if (bs->getEnforceDirichletMethod() == "RowElimination") {
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowOrRowColumnElimination(ELIMINATE_ROW, dof_index, value, queue, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else if (bs->getEnforceDirichletMethod() == "RowColumnElimination") {
            Gpu::BoundaryConditionsHelpers::applyDirichletToNodeGroupViaRowOrRowColumnElimination(ELIMINATE_ROW_COLUMN, dof_index, value, queue, m_linear_system, m_dofs_on_nodes, node_group);
          }
          else {
            ARCANE_FATAL("Unknown method to enforce Dirichlet BC: '{0}'", bs->getEnforceDirichletMethod());
          }
        }
      }
    }
   }
}