class IntrinsicSystem():
    def set_ic(self, q0):
        self.q0 = q0

    def set_name(self):
        pass

    def set_generator(self):

        self.Fdq = None

    def set_solver(self):

        self.eqsolver = getattr(self._library, self.sol_engine)
        

    def solve(self):

        sol = self.eqsolver(self.Fdq, self.q0, **self.sol_settings)
        return sol

        
    def save(self):
        pass
