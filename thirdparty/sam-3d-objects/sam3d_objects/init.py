# Compatibility shim for the package-level import in sam3d_objects.__init__.
#
# The vendored sam-3d-objects tree imports sam3d_objects.init during package
# import, but this file is absent in the checked-in source. Upstream inference
# examples bypass that path with LIDRA_SKIP_INIT; keeping this no-op module lets
# normal imports work without requiring callers to set that environment variable.
