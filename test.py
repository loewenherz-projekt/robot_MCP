def _print_button_events(self):
    # Buttons
    num_buttons = self.gamepad.joystick.get_numbuttons()
    for i in range(num_buttons):
        pressed = self.gamepad.joystick.get_button(i)
        if pressed != self.last_button_state[i]:
            if pressed:
                print(f"Controller Button pressed: {BUTTON_NAMES.get(i, f'Button {i}')}")
            else:
                print(f"Controller Button released: {BUTTON_NAMES.get(i, f'Button {i}')}")
            self.last_button_state[i] = pressed

    # Steuerkreuz / D-Pad (Hat)
    num_hats = self.gamepad.joystick.get_numhats()
    for i in range(num_hats):
        hat = self.gamepad.joystick.get_hat(i)
        if hat != getattr(self, f'last_hat_{i}', (0,0)):
            print(f"D-Pad Hat {i} changed: {hat}")
            setattr(self, f'last_hat_{i}', hat)

    # Trigger und Sticks (Achsen)
    num_axes = self.gamepad.joystick.get_numaxes()
    for i in range(num_axes):
        axis = self.gamepad.joystick.get_axis(i)
        last_axis = getattr(self, f'last_axis_{i}', 0.0)
        if abs(axis - last_axis) > 0.05:  # Nur größere Änderungen anzeigen
            print(f"Axis {i} changed: {axis:.2f}")
            setattr(self, f'last_axis_{i}', axis)
