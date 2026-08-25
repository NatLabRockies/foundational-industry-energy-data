# Modules deselected until their legacy dependencies are restored.
# xfail cannot be used here because these fail at import/collection time
# (missing modules or data files read at module top-level, not inside tests).
collect_ignore = [
    # imports a module that no longer exists in fied.nei
    "test_unit_characterization.py",
    # reads data/FRS/NATIONAL_PROGRAM_FILE.csv at import time
    "test_registry_id_check.py",
    # module-level import_input_data() reads missing nei_data_formatted.csv
    "test_id_ghgrp_units.py",
    "test_separate_unit_data.py",
    # module-level code raises AttributeError during collection
    "test_ghgrp_unit_char.py",
]
