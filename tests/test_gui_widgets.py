import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from calibration_tool.gui.widgets import to_display_u8


class GuiWidgetTests(unittest.TestCase):
    def test_mono12_fixed_display_uses_physical_0_to_4095_range(self):
        image = np.array([[0, 2048, 4095]], dtype=np.uint16)
        display = to_display_u8(image, sensor_max_value=4095)
        self.assertEqual(display.tolist(), [[0, 127, 255]])

    def test_auto_stretch_is_explicit(self):
        image = np.array([[100, 110, 120, 130]], dtype=np.uint16)
        fixed = to_display_u8(image, sensor_max_value=4095)
        stretched = to_display_u8(image, auto_stretch=True, sensor_max_value=4095)
        self.assertLess(int(fixed.max()), 10)
        self.assertGreater(int(stretched.max()), 240)


if __name__ == "__main__":
    unittest.main()
