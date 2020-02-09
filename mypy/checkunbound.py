"""Checks for UnboundLocalError, i.e. a local being used before it is defined.
"""

# FIXME: TODO: Assignment expressions (walruses)

from typing import (
    Dict, Set, List, Union, Optional,
)
from mypy.messages import MessageBuilder
from mypy.nodes import (
    Node, Block, FuncDef, Import, ImportFrom, AssertStmt, AssignmentStmt,
    OperatorAssignmentStmt, GlobalDecl, IfStmt, WhileStmt, ForStmt,
    WithStmt, TryStmt, DelStmt, Lvalue, NameExpr, TupleExpr, IntExpr,
    StarExpr, MemberExpr, IndexExpr, SuperExpr, RevealExpr, CallExpr,
    ReturnStmt, ContinueStmt, RaiseStmt, BreakStmt, FuncItem, Statement,
    ListComprehension, GeneratorExpr,
)
from mypy.options import Options
from mypy.reachability import is_sys_attr
from mypy.traverser import TraverserVisitor


def _is_false(e: Union[NameExpr, IntExpr]) -> bool:
    if isinstance(e, NameExpr) and e.fullname == 'builtins.False':
        return True
    elif isinstance(e, IntExpr) and e.value == 0:
        return True
    else:
        return False


def _is_true(e: Union[NameExpr, IntExpr]) -> bool:
    if isinstance(e, NameExpr) and e.fullname == 'builtins.True':
        return True
    elif isinstance(e, IntExpr) and e.value == 1:
        return True
    else:
        return False


class UnboundLocalErrorChecker(TraverserVisitor):
    def __init__(self, msg: MessageBuilder, options: Options):
        self._msg = msg
        self._options = options

    def visit_func_def(self, func_def: FuncDef):
        checker = UsageChecker(func_def, func_def.arg_names, self._msg, self._options)
        func_def.accept(checker)


class GlobalsCollector(TraverserVisitor):
    globals = None  # type: Set[str]

    def __init__(self) -> None:
        self.globals = set()

    def visit_global_decl(self, o: GlobalDecl) -> None:
        self.globals.update(o.names)


class LocalsCollector(TraverserVisitor):
    options = None  # type: Options
    locals = None  # type: Set[str]
    started = None  # type: bool

    def __init__(self, options: Options) -> None:
        self.options = options
        self.locals = {}
        self.started = False

    def visit_list_comprehension(self, o: ListComprehension) -> None:
        # Python 2 assigns list (but not set/dict) comprehension variables
        # to local variables.
        g = o.generator
        if self.options.python_version >= (3,):
            g.accept(self)
        else:
            # Py2: Inlined g.accept(self) with local variable recording:
            for index, sequence, conditions in zip(g.indices, g.sequences,
                                                   g.condlists):
                sequence.accept(self)
                self.locals.update(_get_lvalues(index))
                for cond in conditions:
                    cond.accept(self)
            g.left_expr.accept(self)

    def visit_assignment_stmt(self, o: AssignmentStmt) -> None:
        for lvalue in o.lvalues:
            self.locals.update(_get_lvalues(lvalue))

    def visit_operator_assignment_stmt(self, o: OperatorAssignmentStmt) -> None:
        self.locals.update(_get_lvalues(o.lvalue))

    def visit_for_stmt(self, o: ForStmt) -> None:
        self.locals.update(_get_lvalues(o.index))
        super().visit_for_stmt(o)

    def visit_func_def(self, o: FuncDef) -> None:
        # Don't recurse into nested FuncDefs.
        if not self.started:
            self.started = True
            super().visit_func_def(o)

    def visit_with_stmt(self, o: WithStmt) -> None:
        for i in range(len(o.expr)):
            if o.target[i] is not None:
                self.locals.update(_get_lvalues(o.target[i]))
        super().visit_with_stmt(o)

    def visit_try_stmt(self, o: TryStmt) -> None:
        for i in range(len(o.types)):
            v = o.vars[i]
            if v is not None:
                self.locals.update(_get_lvalues(v))
        super().visit_try_stmt(o)

    def visit_import(self, o: Import) -> None:
        for target, alias in o.ids:
            self.locals.add(alias or target)
        super().visit_import(o)

    def visit_import_from(self, o: ImportFrom) -> None:
        for target, alias in o.names:
            self.locals.add(alias or target)
        super().visit_import_from(o)

    def visit_del_stmt(self, o: DelStmt) -> None:
        self.locals.update(_get_lvalues(o.expr))
        super().visit_del_stmt(o)


def _get_lvalues(lvalue: Lvalue) -> Set[str]:
    def walk(x):
        if isinstance(x, NameExpr):
            print('FOUND LOCAL', x.name, x.node, repr(x.node))
            result[x.name] = x
        elif isinstance(x, TupleExpr):
            for child in x.items:
                walk(child)
        elif isinstance(x, StarExpr):
            if x.valid:
                walk(x.expr)
        else:
            assert isinstance(x, (MemberExpr, IndexExpr, SuperExpr)), type(x)

    result = {}
    walk(lvalue)
    return result


def _get_globals(o: Node) -> Set[str]:
    collector = GlobalsCollector()
    o.accept(collector)
    return collector.globals


def _get_locals(o: Node, options: Options) -> Dict[str, NameExpr]:
    collector = LocalsCollector(options)
    o.accept(collector)
    return collector.locals


def _is_end_unreachable(body: Union[Node, List[Statement]]) -> bool:
    if body is None:
        return False
    walker = ReachabilityChecker()
    if isinstance(body, list):
        for stmt in body:
            stmt.accept(walker)
    else:
        body.accept(walker)
    return walker.is_end_unreachable


class ReachabilityChecker(TraverserVisitor):
    is_end_unreachable = None  # type: bool

    def __init__(self) -> None:
        self.is_end_unreachable = False

    # Descend branches
    def _check_block(self, o: Block) -> None:
        checker = ReachabilityChecker()
        o.accept(checker)
        return checker.is_end_unreachable

    def _visit_bodies_and_else(self, bodies: List[Block],
                               else_body: Optional[Block],
                               else_is_unreachable: bool = False) -> None:
        if self.is_end_unreachable:
            return
        if not else_is_unreachable and else_body:
            bodies.append(else_body)
        self.is_end_unreachable = all(self._check_block(b) for b in bodies)

    def visit_if_stmt(self, o: IfStmt) -> None:
        for e in o.expr:
            e.accept(self)
        self._visit_bodies_and_else(o.body, o.else_body)

    def visit_while_stmt(self, o: WhileStmt) -> None:
        o.expr.accept(self)
        self._visit_bodies_and_else([o.body], o.else_body, _is_true(o.expr))

    def visit_for_stmt(self, o: ForStmt) -> None:
        o.expr.accept(self)
        o.index.accept(self)
        self._visit_bodies_and_else([o.body], o.else_body)

    def visit_try_stmt(self, o: TryStmt) -> None:
        if self.is_end_unreachable:
            return
        has_unreachable_end = [self._check_block(o.body)]
        for i in range(len(o.types)):
            checker = ReachabilityChecker()
            tp = o.types[i]
            if tp is not None:
                tp.accept(checker)
            v = o.vars[i]
            if v is not None:
                v.accept(checker)
            o.handlers[i].accept(checker)
            has_unreachable_end.append(checker.is_end_unreachable)
        if o.else_body is not None:
            has_unreachable_end.append(self._check_block(o.else_body))
        if o.finally_body is not None:
            has_unreachable_end.append(self._check_block(o.finally_body))
        return all(has_unreachable_end)

    # Mark unreachable if we hit return etc.
    def visit_return_stmt(self, o: ReturnStmt) -> None:
        super().visit_return_stmt(o)
        self.is_end_unreachable = True

    def visit_continue_stmt(self, o: ContinueStmt) -> None:
        super().visit_continue_stmt(o)
        self.is_end_unreachable = True

    def visit_raise_stmt(self, o: RaiseStmt) -> None:
        super().visit_raise_stmt(o)
        self.is_end_unreachable = True

    def visit_call_expr(self, o: CallExpr) -> None:
        # Treat "sys.exit" calls like "raise SystemExit".
        if is_sys_attr(o.callee, 'exit'):
            self.is_end_unreachable = True
        else:
            super().visit_call_expr(o)

    def visit_break_stmt(self, o: BreakStmt) -> None:
        super().visit_break_stmt(o)
        self.is_end_unreachable = True

    def visit_assert_stmt(self, o: AssertStmt) -> None:
        if _is_false(o.expr):
            self.is_end_unreachable = True
        super().visit_assert_stmt(o)

    def visit_func(self, o: FuncItem) -> None:
        # Don't recurse into nested functions.
        pass


class _UsageChecker(TraverserVisitor):
    parent = None  # type: UsageChecker
    locals = None  # type: FrozenSet[str]
    assignments = None  # type: Dict[str, bool]

    def __init__(self, parent: Union['UsageChecker', '_UsageChecker']) -> None:
        self.parent = parent
        self.locals = parent.locals
        self.assignments = parent.assignments.copy()

    @property
    def msg(self) -> MessageBuilder:
        return self.parent.msg

    @property
    def options(self) -> Options:
        return self.parent.options

    def _get_block_assignments(self, o: Optional[Block],
                               loop_vars: List[str] = []) \
            -> Optional[Dict[str, bool]]:
        """Get the assignments dict for `o` (if given and its end is
           reachable), treating `loop_vars` as defined.  If `o` is None,
           return a copy of the current assignments."""
        if o is None:
            return self.assignments.copy()
        if _is_end_unreachable(o):
            return None
        checker = _UsageChecker(self)
        for name in loop_vars:
            checker._record_assignment(name)
        o.accept(checker)
        return checker.assignments

    def _record_assignment(self, name: str) -> None:
        if name in self.locals:
            self.assignments[name] = True

    def _record_deletion(self, name: str) -> None:
        if name in self.locals:
            self.assignments[name] = False

    def _merge_blocks(self, blocks: List[Block], loop_vars: List[str]) \
            -> Optional[Dict[str, bool]]:
        assert isinstance(blocks, list)
        assert all(b is None or isinstance(b, Block) for b in blocks)
        block_assignments = []
        for b in blocks:
            assignments = self._get_block_assignments(b, loop_vars)
            if assignments is not None:
                block_assignments.append(assignments)
            loop_vars = []
        return self._merge_assignments(block_assignments)

    def _merge_assignments(self,
                           block_assignments: List[Dict[str, bool]]) \
            -> Optional[Dict[str, bool]]:
        if not block_assignments:
            return None
        # Each variable must be defined in all reachable-ended blocks.
        assignments = {}
        for local in self.locals:
            assignments[local] = all(a[local] for a in block_assignments)
        return assignments

    def _make_generator_checker(self, o: GeneratorExpr) -> '_UsageChecker':
        # Use a separate usage checker in which they aren't locals.
        checker = _UsageChecker(self)
        checker_locals = set(checker.locals)
        for index in o.indices:
            for name in _get_lvalue_names(index):
                # Mark name as non-local within checker
                checker_locals.discard(name)
                checker.assignments.pop(name, None)
        checker.locals = frozenset(checker_locals)
        return checker

    def visit_generator_expr(self, o: GeneratorExpr) -> None:
        checker = self._make_generator_checker(o)
        super(_UsageChecker, checker).visit_generator_expr(o)

    def visit_list_comprehension(self, o: ListComprehension) -> None:
        g = o.generator
        # Python 2 assigns list (but not set/dict) comprehension variables
        # to local variables. Python 3 treats them like globals/outers.
        if self.options.python_version >= (3,):
            checker = self._make_generator_checker(g)
        else:
            checker = self
        # Inlined g.accept(checker) with assignment recording:
        for index, sequence, conditions in zip(g.indices, g.sequences,
                                               g.condlists):
            sequence.accept(checker)
            for name in _get_lvalue_names(index):
                checker._record_assignment(name)
            for cond in conditions:
                cond.accept(checker)
        g.left_expr.accept(checker)

    def _visit_bodies_and_else(self, bodies, else_body,
                               else_is_unreachable: bool = False,
                               loop_vars: List[str] = []) -> None:
        if not else_is_unreachable:
            bodies.append(else_body)
        self._absorb_assignments(self._merge_blocks(bodies, loop_vars))

    def _absorb_assignments(self, assignments: Optional[Dict[str, bool]]) \
            -> None:
        if assignments is None:
            # Unreachable code path, don't warn about locals
            self.assignments = {local: True for local in self.locals}
        else:
            self.assignments = assignments

    def visit_if_stmt(self, o: IfStmt) -> None:
        else_is_unreachable = False
        blocks = []
        for e, body in zip(o.expr, o.body):
            e.accept(self)
            if not _is_false(e):
                blocks.append(body)
            if _is_true(e):
                else_is_unreachable = True
                break  # skip unreachable blocks
        self._visit_bodies_and_else(blocks, o.else_body, else_is_unreachable)

    def visit_while_stmt(self, o: WhileStmt) -> None:
        o.expr.accept(self)
        else_is_unreachable = _is_true(o.expr)
        self._visit_bodies_and_else([o.body], o.else_body, else_is_unreachable)

    def visit_for_stmt(self, o: ForStmt) -> None:
        o.expr.accept(self)
        self._visit_bodies_and_else([o.body], o.else_body, False,
                                    _get_lvalue_names(o.index))

    def _check(self, o: Optional[Node]) -> None:
        if o is not None:
            o.accept(self)

    def visit_try_stmt(self, o: TryStmt) -> None:
        block_assignments = [self._get_block_assignments(o.body)]
        for i in range(len(o.types)):
            checker = _UsageChecker(self)
            checker._check(o.types[i])
            v = o.vars[i]
            if v is not None:
                checker._record_assignment(v.name)
            checker._check(v)
            checker._check(o.handlers[i])
            if v is not None and self.options.python_version >= (3,):
                # Python 3 has an implicit "del <name>" here, Python 2 doesn't.
                checker._record_deletion(v.name)
            if not _is_end_unreachable(o.handlers[i]):
                block_assignments.append(checker.assignments)
        if o.else_body is None:
            if block_assignments[0] is None:
                else_assignments = None
            else:
                else_assignments = block_assignments[0].copy()
        else:
            if block_assignments[0] is None:
                else_assignments = None
            else:
                self.assignments = block_assignments[0]
                else_assignments = self._check(o.else_body)
        block_assignments.append(else_assignments)
        block_assignments = [a for a in block_assignments if a is not None]
        assignments = self._merge_assignments(block_assignments)
        self._absorb_assignments(assignments)
        if o.finally_body is not None:
            assignments = self._check(o.finally_body)
            self._absorb_assignments(assignments)

    def visit_assignment_stmt(self, o: AssignmentStmt) -> None:
        for name in _get_locals(o, self.options):
            self._record_assignment(name)
        super().visit_assignment_stmt(o)

    def visit_func_def(self, o: FuncDef) -> None:
        # Don't recurse into nested FuncDefs, checker.py does that.
        self._record_assignment(o.name())

    def visit_with_stmt(self, o: WithStmt) -> None:
        for i in range(len(o.expr)):
            o.expr[i].accept(self)
            targ = o.target[i]
            if targ is not None:
                for x in _get_lvalue_names(targ):
                    self.assignments[x] = True
                targ.accept(self)
        o.body.accept(self)

    def visit_import(self, o: Import) -> None:
        for target, alias in o.ids:
            self._record_assignment(alias or target)
        super().visit_import(o)

    def visit_import_from(self, o: ImportFrom) -> None:
        for target, alias in o.names:
            self._record_assignment(alias or target)
        super().visit_import_from(o)

    def visit_name_expr(self, o: NameExpr) -> None:
        if o.name in self.locals and not self.assignments[o.name]:
            self.msg.warn_may_raise_unbound_local_error(o.name, o)
        super().visit_name_expr(o)

    def visit_del_stmt(self, o: DelStmt) -> None:
        super().visit_del_stmt(o)
        for name in _get_lvalue_names(o.expr):
            self._record_deletion(name)

    def visit_call_expr(self, o: CallExpr) -> None:
        # Skip RevealExpr calls (reveal_type() or reveal_locals()).
        # They're debug aids that are prone to false positives.
        if not isinstance(o.analyzed, RevealExpr):
            super().visit_call_expr(o)


class UsageChecker(TraverserVisitor):
    msg = None  # type: MessageBuilder
    locals = None  # type: FrozenSet[str]
    assignments = None  # type: Dict[str, bool]
    options = None  # type: Options

    def __init__(self, node: Node, bound_names: List[str], msg: MessageBuilder,
                 options: Options) -> None:
        locals = _get_locals(node, options)
        for name in bound_names:
            locals.add(name)
        for name in _get_globals(node):
            locals.discard(name)
        self.msg = msg
        self.locals = frozenset(locals)
        self.assignments = {}  # local variable name -> is assigned
        for name in self.locals:
            self.assignments[name] = name in bound_names
        self.options = options

    def visit_func_def(self, o: FuncDef) -> None:
        checker = _UsageChecker(self)
        if o.arguments is not None:
            for arg in o.arguments:
                checker.visit_var(arg.variable)
        o.body.accept(checker)
