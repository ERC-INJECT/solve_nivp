import numpy as np
import scipy.sparse as sp
import scipy.optimize._numdiff as numdiff
import pytest

import solve_nivp.nonlinear_solvers as ns
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection


def test_public_jacobian_sparsity_uses_coloring(monkeypatch):
    called = {}

    def fake_approx_derivative(func, y, method='2-point', sparsity=None, rel_step=None):
        called['method'] = method
        called['shape'] = None if sparsity is None else sparsity.shape
        return sp.eye(len(y), format='csr')

    monkeypatch.setattr(numdiff, 'approx_derivative', fake_approx_derivative)

    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        jacobian_sparsity=sp.eye(4, format='csr'),
        sparse=True,
    )

    J = solver._numerical_jacobian(lambda y: y, np.ones(4), sparse=True)

    assert sp.issparse(J)
    assert called['method'] == '2-point'
    assert called['shape'] == (4, 4)


def test_removed_autodiff_options_raise_type_error():
    with pytest.raises(TypeError):
        ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            use_autodiff=True,
        )

    with pytest.raises(TypeError):
        ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            autodiff_mode='cs',
        )


class _FakeVec:
    def __init__(self):
        self.array = None
        self.size = None
        self.comm = None
        self.destroyed = False
        self.create_with_array_calls = 0
        self.set_array_calls = 0
        self.set_calls = 0

    def create(self, comm=None):
        self.comm = comm
        return self

    def createWithArray(self, arr, comm=None):
        self.create_with_array_calls += 1
        self.array = np.asarray(arr)
        self.comm = comm
        return self

    def setType(self, vec_type):
        self.vec_type = vec_type

    def setSizes(self, n):
        self.size = int(n)

    def setUp(self):
        if self.array is None and self.size is not None:
            self.array = np.zeros(self.size)

    def setArray(self, arr):
        self.set_array_calls += 1
        self.array = np.asarray(arr)

    def set(self, value):
        self.set_calls += 1
        if self.array is not None:
            self.array[...] = value

    def getArray(self):
        return self.array

    def destroy(self):
        self.destroyed = True
        return None


class _FakeIS:
    def createGeneral(self, indices, comm=None):
        return np.array(indices, copy=True)


class _FakePC:
    class CompositeType:
        SCHUR = 'schur'

    def __init__(self):
        self.setup_calls = 0
        self.field_splits = []

    def setType(self, pc_type):
        self.pc_type = pc_type

    def setHYPREType(self, hypre_type):
        self.hypre_type = hypre_type

    def setFactorSolverType(self, solver_type):
        self.factor_solver_type = solver_type

    def setReusePreconditioner(self, flag):
        self.reuse_preconditioner = bool(flag)

    def setFieldSplitIS(self, *pairs):
        self.field_splits.extend(pairs)

    def setFieldSplitType(self, fs_type):
        self.field_split_type = fs_type

    def setUp(self):
        self.setup_calls += 1


class _FakeKSP:
    def __init__(self):
        self.pc = _FakePC()
        self.iters = 2

    def create(self, comm=None):
        return self

    def setOperators(self, mat):
        self.mat = mat

    def setType(self, ksp_type):
        self.ksp_type = ksp_type

    def getPC(self):
        return self.pc

    def setTolerances(self, rtol=None, atol=None, max_it=None):
        self.rtol = rtol
        self.atol = atol
        self.max_it = max_it

    def setGMRESRestart(self, restart):
        self.restart = restart

    def setOptionsPrefix(self, prefix):
        self.prefix = prefix

    def setFromOptions(self):
        return None

    def solve(self, b, x):
        x.array = np.array(b.array, copy=True)

    def getConvergedReason(self):
        return 1

    def getIterationNumber(self):
        return self.iters

    def destroy(self):
        return None


class _FakeMat:
    def __init__(self):
        self.shape = None
        self.set_values_calls = 0
        self.last_values = None
        self.comm = None

    def create(self, comm=None):
        self.comm = comm
        return self

    def setType(self, mat_type):
        self.mat_type = mat_type

    def setSizes(self, shape):
        self.shape = tuple(shape)

    def setPreallocationCSR(self, csr):
        self.prealloc = csr

    def setUp(self):
        return None

    def setValuesCSR(self, indptr, indices, data):
        self.set_values_calls += 1
        self.last_values = np.array(data, copy=True)

    def createAIJ(self, size=None, csr=None, comm=None):
        self.shape = tuple(size)
        self.csr = csr
        self.comm = comm
        return self

    def assemble(self):
        return None

    def createVecRight(self):
        vec = _FakeVec()
        vec.setSizes(self.shape[1])
        vec.setUp()
        return vec

    def destroy(self):
        return None


class _FakeOptions(dict):
    pass


class _FakeComm:
    def __init__(self, size):
        self._size = int(size)

    def getSize(self):
        return self._size


class _FakePETSc:
    IntType = np.int32
    ScalarType = np.float64
    COMM_SELF = _FakeComm(1)
    COMM_WORLD = _FakeComm(1)
    PC = _FakePC

    @staticmethod
    def Mat():
        return _FakeMat()

    @staticmethod
    def KSP():
        return _FakeKSP()

    @staticmethod
    def Vec():
        return _FakeVec()

    @staticmethod
    def IS():
        return _FakeIS()

    @staticmethod
    def Options():
        return _FakeOptions()


def test_petsc_iterative_updates_operator_each_solve(monkeypatch):
    monkeypatch.setattr(ns, 'PETSC_AVAILABLE', True)
    monkeypatch.setattr(ns, 'PETSc', _FakePETSc)

    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        linear_solver='petsc',
        petsc_options={'ksp_type': 'gmres', 'pc_type': 'jacobi'},
        petsc_reuse_steps=5,
    )

    J1 = sp.eye(3, format='csr')
    J2 = sp.diags([2.0, 3.0, 4.0], format='csr')
    b = np.array([1.0, 2.0, 3.0])

    x1, ok1 = solver._solve_with_petsc(J1, b)
    assert ok1
    np.testing.assert_allclose(x1, b)
    assert solver._petsc_mat.set_values_calls == 0

    x2, ok2 = solver._solve_with_petsc(J2, b)
    assert ok2
    np.testing.assert_allclose(x2, b)
    assert solver._petsc_mat.set_values_calls == 1
    np.testing.assert_allclose(solver._petsc_mat.last_values, J2.data)


def test_petsc_direct_updates_same_pattern_without_recreating_ksp(monkeypatch):
    monkeypatch.setattr(ns, 'PETSC_AVAILABLE', True)
    monkeypatch.setattr(ns, 'PETSc', _FakePETSc)

    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        linear_solver='petsc',
        petsc_options={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        },
        petsc_reuse_steps=50,
    )

    J1 = sp.diags([1.0, 1.0, 1.0], format='csr')
    J2 = sp.diags([2.0, 3.0, 4.0], format='csr')
    b = np.array([1.0, 2.0, 3.0])

    x1, ok1 = solver._solve_with_petsc(J1, b)
    assert ok1
    np.testing.assert_allclose(x1, b)
    mat0 = solver._petsc_mat
    ksp0 = solver._petsc_ksp

    solver._petsc_needs_matrix_update = True
    x2, ok2 = solver._solve_with_petsc(J2, b)

    assert ok2
    np.testing.assert_allclose(x2, b)
    assert solver._petsc_mat is mat0
    assert solver._petsc_ksp is ksp0
    assert solver._petsc_mat.set_values_calls == 1
    np.testing.assert_allclose(solver._petsc_mat.last_values, J2.data)
    assert solver._petsc_ksp.getPC().setup_calls == 1
    assert solver._petsc_ksp.getPC().reuse_preconditioner is True


def test_petsc_cpu_vectors_are_reused_between_solves(monkeypatch):
    monkeypatch.setattr(ns, 'PETSC_AVAILABLE', True)
    monkeypatch.setattr(ns, 'PETSc', _FakePETSc)

    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        linear_solver='petsc',
        petsc_options={'ksp_type': 'gmres', 'pc_type': 'jacobi'},
        petsc_reuse_steps=5,
    )

    J = sp.eye(3, format='csr')
    b1 = np.array([1.0, 2.0, 3.0])
    b2 = np.array([4.0, 5.0, 6.0])

    x1, ok1 = solver._solve_with_petsc(J, b1)
    assert ok1
    np.testing.assert_allclose(x1, b1)
    b_vec0 = solver._petsc_b_vec
    x_vec0 = solver._petsc_x_vec
    assert b_vec0 is not None
    assert x_vec0 is not None
    assert not b_vec0.destroyed
    assert not x_vec0.destroyed

    x2, ok2 = solver._solve_with_petsc(J, b2)

    assert ok2
    np.testing.assert_allclose(x2, b2)
    assert solver._petsc_b_vec is b_vec0
    assert solver._petsc_x_vec is x_vec0
    assert not b_vec0.destroyed
    assert not x_vec0.destroyed
    assert b_vec0.set_array_calls == 0
    assert x_vec0.set_calls == 2


def test_petsc_cpu_vector_cache_resets_when_shape_changes(monkeypatch):
    monkeypatch.setattr(ns, 'PETSC_AVAILABLE', True)
    monkeypatch.setattr(ns, 'PETSc', _FakePETSc)

    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        linear_solver='petsc',
        petsc_options={'ksp_type': 'gmres', 'pc_type': 'jacobi'},
        petsc_reuse_steps=5,
    )

    _, ok1 = solver._solve_with_petsc(sp.eye(3, format='csr'), np.ones(3))
    assert ok1
    b_vec0 = solver._petsc_b_vec
    x_vec0 = solver._petsc_x_vec

    x2, ok2 = solver._solve_with_petsc(sp.eye(4, format='csr'), np.arange(4.0))

    assert ok2
    np.testing.assert_allclose(x2, np.arange(4.0))
    assert solver._petsc_b_vec is not b_vec0
    assert solver._petsc_x_vec is not x_vec0
    assert b_vec0.destroyed
    assert x_vec0.destroyed


def test_petsc_cpu_vector_cache_is_cleared_by_invalidate_all_caches(monkeypatch):
    monkeypatch.setattr(ns, 'PETSC_AVAILABLE', True)
    monkeypatch.setattr(ns, 'PETSc', _FakePETSc)

    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        linear_solver='petsc',
        petsc_options={'ksp_type': 'gmres', 'pc_type': 'jacobi'},
    )

    _, ok = solver._solve_with_petsc(sp.eye(3, format='csr'), np.ones(3))
    assert ok
    b_vec0 = solver._petsc_b_vec
    x_vec0 = solver._petsc_x_vec

    solver.invalidate_all_caches()

    assert solver._petsc_b_vec is None
    assert solver._petsc_x_vec is None
    assert solver._petsc_b_array is None
    assert solver._petsc_vec_shape is None
    assert b_vec0.destroyed
    assert x_vec0.destroyed


def test_fieldsplit_accepts_array_component_slices(monkeypatch):
    monkeypatch.setattr(ns, 'PETSC_AVAILABLE', True)
    monkeypatch.setattr(ns, 'PETSc', _FakePETSc)

    slices = [np.array([0, 2]), np.array([1, 3])]
    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        linear_solver='petsc',
        component_slices=slices,
        petsc_options={
            'ksp_type': 'gmres',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
        },
    )

    _, ok = solver._solve_with_petsc(sp.eye(4, format='csr'), np.ones(4))
    assert ok

    field_splits = solver._petsc_ksp.getPC().field_splits
    assert len(field_splits) == 2
    np.testing.assert_array_equal(field_splits[0][1], slices[0])
    np.testing.assert_array_equal(field_splits[1][1], slices[1])


def test_petsc_gpu_request_falls_back_to_cpu_when_backend_unavailable(monkeypatch):
    monkeypatch.setattr(ns, 'PETSC_AVAILABLE', True)
    monkeypatch.setattr(ns, 'PETSc', _FakePETSc)

    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        linear_solver='petsc',
        petsc_options={
            'ksp_type': 'gmres',
            'pc_type': 'jacobi',
            'mat_type': 'aijcusparse',
            'vec_type': 'cuda',
        },
    )

    monkeypatch.setattr(
        solver,
        '_petsc_type_supported',
        lambda kind, type_name: False if type_name in ('aijcusparse', 'cuda') else True,
    )

    with pytest.warns(RuntimeWarning, match="Falling back to CPU PETSc objects"):
        _, ok = solver._solve_with_petsc(sp.eye(3, format='csr'), np.ones(3))

    assert ok
    assert solver._petsc_use_gpu is False
    assert solver._petsc_effective_mat_type is None
    assert solver._petsc_effective_vec_type is None
    assert solver._petsc_mat.comm is _FakePETSc.COMM_SELF


def test_petsc_world_comm_is_used_for_single_rank_runs(monkeypatch):
    monkeypatch.setattr(ns, 'PETSC_AVAILABLE', True)
    monkeypatch.setattr(ns, 'PETSc', _FakePETSc)

    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        linear_solver='petsc',
        petsc_comm='world',
        petsc_options={'ksp_type': 'gmres', 'pc_type': 'jacobi'},
    )

    _, ok = solver._solve_with_petsc(sp.eye(3, format='csr'), np.ones(3))

    assert ok
    assert solver._petsc_mat.comm is _FakePETSc.COMM_WORLD
    assert solver._petsc_comm_obj is _FakePETSc.COMM_WORLD


def test_petsc_multi_rank_comm_raises_until_distributed_assembly_exists(monkeypatch):
    monkeypatch.setattr(ns, 'PETSC_AVAILABLE', True)
    monkeypatch.setattr(ns, 'PETSc', _FakePETSc)

    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        linear_solver='petsc',
        petsc_comm=_FakeComm(2),
        petsc_options={'ksp_type': 'gmres', 'pc_type': 'jacobi'},
    )

    with pytest.raises(NotImplementedError, match="Distributed PETSc communicators"):
        solver._solve_with_petsc(sp.eye(3, format='csr'), np.ones(3))


def test_extract_sparse_numeric_diagonal_accepts_stored_zero_offdiagonals():
    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
    )
    D = sp.csr_matrix(
        (
            np.array([1.0, 0.0, 0.0, 2.0, 3.0, 0.0]),
            np.array([0, 1, 0, 1, 2, 3]),
            np.array([0, 2, 4, 5, 6]),
        ),
        shape=(4, 4),
    )

    is_diag, diag = solver._extract_sparse_numeric_diagonal(D, 4)

    assert is_diag
    np.testing.assert_allclose(diag, [1.0, 2.0, 3.0, 0.0])


def test_exact_diag_newton_assembly_matches_generic_sparse_formula():
    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
    )
    J = sp.csr_matrix(
        [
            [4.0, -1.0, 0.0, 0.0],
            [2.0, 3.0, 5.0, 0.0],
            [0.0, 6.0, 7.0, 8.0],
            [0.0, 0.0, 9.0, 10.0],
        ]
    )
    d1 = np.array([1.0, 0.25, 0.0, 0.75])
    lam1 = 0.4

    fast1 = solver._assemble_diag_newton_csr(J, d1, lam1)
    ref1 = J.multiply((d1 * lam1)[:, None]) + sp.diags(1.0 - d1, format='csr')

    assert fast1 is not None
    np.testing.assert_allclose(fast1.toarray(), ref1.toarray())

    d2 = np.array([0.5, 1.0, 0.2, 0.0])
    lam2 = np.array([2.0, 1.5, 0.5, 3.0])

    fast2 = solver._assemble_diag_newton_csr(J, d2, lam2)
    ref2 = J.multiply((d2 * lam2)[:, None]) + sp.diags(1.0 - d2, format='csr')

    assert fast2 is fast1
    np.testing.assert_allclose(fast2.toarray(), ref2.toarray())


def test_exact_diag_newton_assembly_falls_back_without_structural_diagonal():
    solver = ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
    )
    J = sp.csr_matrix(
        [
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [4.0, 0.0, 5.0],
        ]
    )

    fast = solver._assemble_diag_newton_csr(J, np.ones(3), 1.0)

    assert fast is None
