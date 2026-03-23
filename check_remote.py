import ttnn
print("Devices:", ttnn.GetNumAvailableDevices())
print("Pcie devices:", ttnn.GetNumPCIeDevices())

try:
    import ttl
    print("tt-lang: available")
except ImportError:
    print("tt-lang: NOT available")

# Quick single-device open test
device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device 0 grid: {grid.x}x{grid.y}")
print(f"Arch: {device.arch()}")
ttnn.close_device(device)
print("Device open/close: OK")
