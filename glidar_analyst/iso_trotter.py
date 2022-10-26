import matplotlib

matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')

from pint import UnitRegistry

units = UnitRegistry()

from glidar_analyst.model.solvers import NewtonSolver


class Target:

    def __init__(self, x, y, var):

        self.target_x = x
        self.target_y = y
        self.variable = var


class TracerParamObject:

    ID_COUNTER = int(0)

    def __init__(self, params: dict, result, targets: dict, variable: str):

        self.id = None

        self.params = params
        self.result = result
        self.targets = targets
        self.variable_changed = variable


class TracerObjectDatabase:

    def __init__(self):

        self.data = {}

    def create(self, tracer_object: TracerParamObject):

        tracer_object.id = TracerParamObject.ID_COUNTER
        TracerParamObject.ID_COUNTER += int(1)

        self.data[tracer_object.id] = tracer_object

        return tracer_object

    def find_by_id(self, id: int) -> TracerParamObject:

        return self.data[id]

    def find_all(self):

        return self.data.values()


class IsoTrotter:

    def __init__(self, model):

        self.model = model
        self.solver = NewtonSolver(self.model)

    def isotrotting(self, params, targets):

        p = self.solver.solve_multi_target(params, targets)

        return p
