from core.hardware_check import HardwareCheck


def test_recommendations_have_device():
    hw = HardwareCheck()
    rec = hw.recommendations
    assert 'device' in rec


def test_disk_check():
    hw = HardwareCheck()
    assert hw.enough_disk(0.1)
