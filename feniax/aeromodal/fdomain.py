def _compute_modalaero(self):
    approx = self._config.aero.approx
    container = dict()
    if self._config.aero.Qk_struct is not None:
        if len(self._config.aero.Qk_struct[0]) == 1:  # steady
            A0 = self._config.aero.Qk_struct[1]
            container.update(A0=A0)
        else:
            pass

    if self._config.aero.Qk_controls is not None:
        if len(self._config.aero.Qk_controls[0]) == 1:  # steady
            B0 = self._config.aero.Qk_controls[1]
            container.update(B0=B0)

    if self._config.aero.Q0_rigid is not None:
        C0 = self._config.aero.Q0_rigid
        container.update(C0=C0)
