

def _branch_in_process_fuel(scc: pl.LazyFrame) -> pl.LazyFrame:
    """Ind. Proc. - In-process Fuel Use"""
    return (
        scc
        .filter(
            (_LV2 == "In-process Fuel Use")
            & (_SECTOR != "Industrial Processes - Storage and Transfer")
        )
        .with_columns(
            pl.when(_LV4.str.contains("Kiln"))
            .then(pl.struct(
                unit_type_lv1=pl.lit("Kiln"),
                unit_type_lv2=_LV4,
            ))
            .otherwise(pl.struct(
                unit_type_lv1=pl.lit("Other combustion"),
                unit_type_lv2=pl.lit("Other combustion"),
            ))
            .alias("_unit_types")
        )
        .unnest("_unit_types")
        .pipe(map_fuel_types, fuel_type_col="scc_level_three")
    )


def _branch_commercial_cooking(scc: pl.LazyFrame) -> pl.LazyFrame:
    """Determine unit and fuel for Commercial Cooking

    Within sector "Commercial Cooking", there are three variations
    of SCC-level 3:
    - Commercial Cooking - Frying
    - Commercial Cooking - Total
    - Commercial Cooking - Charbroiling

    The 'Total' variation results in unit level-2 = 'Cooking',
    otherwise it takes the variation itself, i.e. a
    'Commercial Cooking - Frying' -> unit level-2 = 'Frying'.

    Notes
    -----
    - In case of a record without 'Commercial Cooking' would result
      in a unit level-2 as `None`.
    """
    return (
        scc
        .filter(_SECTOR == "Commercial Cooking")
        .with_columns(
            pl.when(_LV3.str.contains("Commercial Cooking - Total"))
            .then(pl.struct(
                unit_type_lv1=pl.lit("Other combustion"),
                unit_type_lv2=pl.lit("Cooking"),
            ))
            .when(_LV3.str.contains("Commercial Cooking"))
            .then(pl.struct(
                unit_type_lv1=pl.lit("Other combustion"),
                unit_type_lv2=_LV3.str.split(" - ").list.last(),
            ))

            .otherwise(pl.struct(
                unit_type_lv1=pl.lit(None, dtype=pl.Utf8),
                unit_type_lv2=pl.lit(None, dtype=pl.Utf8),
            ))
            .alias("_unit_type"),
        )
        .unnest("_unit_type")
        .pipe(_with_null_fuel)
    )


def _branch_ammonia(scc: pl.LazyFrame) -> pl.LazyFrame:
    """Ammonia Production

    When ``scc_level_four`` contains ``': '``, the text before the
    separator is the unit description and the text after (up to
    ``' Fired'``) is the fuel-type key.
    """
    base = scc.filter(_LV3 == "Ammonia Production")

    has_colon = _LV4.str.contains(": ")

    with_colon = (
        base
        .filter(has_colon)
        .with_columns(
            pl.lit("Other combustion").alias("unit_type_lv1"),
            _LV4.str.split(": ").list.first().alias("unit_type_lv2"),
            _LV4.str.split(": ").list.last().str.split(" Fired").list.first().alias("_fuel_key"),
        )
        .pipe(map_fuel_types, fuel_type_col="_fuel_key")
        #.unnest("_fuel_types")
        .drop("_fuel_key")
    )

    without_colon = (
        base
        .filter(~has_colon)
        .with_columns(
            pl.lit("Other").alias("unit_type_lv1"),
            _LV4.alias("unit_type_lv2"),
        )
        .pipe(_with_null_fuel)
    )

    return pl.concat([with_colon, without_colon], how="diagonal_relaxed")


def _branch_storage_transport(scc_ind: pl.LazyFrame) -> pl.LazyFrame:
    """Storage & Transport (non-In-process Fuel Use)"""
    return (
        scc_ind
        .filter(
            (_LV2 != "In-process Fuel Use")
            & (_T1DESC == "Storage & Transport")
        )
        .filter(~_LV4.str.contains("Breathing Loss"))
        .with_columns(
            pl.lit("Other").alias("unit_type_lv1"),
            _LV4.alias("unit_type_lv2"),
        )
        .pipe(_with_null_fuel)
    )


def _branch_petroleum_refineries(scc: pl.LazyFrame) -> pl.LazyFrame:
    """Petroleum Refineries"""
    refinery_ut = {
        "Process Heaters": ("Heater", "Process heater"),
        "Flares": ("Other combustion", "Flare"),
        "Fluid Coking Unit": ("Other combustion", "Fluid coking unit"),
        "Petroleum Coke Calcining": ("Other combustion", "Petroleum coke calcining"),
        "Incinerators": ("Other combustion", "Incinerator"),
    }

    base = scc.filter(_SECTOR == "Industrial Processes - Petroleum Refineries")

    is_refinery_type = _LV3.is_in(list(refinery_ut.keys()))

    # -- Build a single when/then chain returning a struct ----
    ut_chain = pl.lit(None, dtype=pl.Struct({
        "unit_type_lv1": pl.Utf8,
        "unit_type_lv2": pl.Utf8,
    }))
    for lv3_key, (u1, u2) in refinery_ut.items():
        ut_chain = (
            pl.when(_LV3 == lv3_key)
            .then(pl.struct(
                unit_type_lv1=pl.lit(u1),
                unit_type_lv2=pl.lit(u2),
            ))
            .otherwise(ut_chain)
        )

    known = (
        base
        .filter(is_refinery_type)
        # -- Unit types ----
        .with_columns(ut_chain.alias("_unit_types"))
        .unnest("_unit_types")
        # -- Fuel types ----
        .with_columns(
            pl.when(_LV4.str.contains(": "))
            .then(_LV4.str.split(": ").list.last())
            .otherwise(_LV4)
            .alias("_fuel_key"),
        )
        .pipe(map_fuel_types, fuel_type_col="_fuel_key")
        # .unnest("_fuel_types")
        .drop("_fuel_key")
    )

    # -- Unknown refinery types ----
    unknown = (
        base
        .filter(~is_refinery_type)
        .with_columns(
            pl.lit("Other").alias("unit_type_lv1"),
            _LV3.alias("unit_type_lv2"),
        )
        .pipe(_with_null_fuel)
    )

    return pl.concat([known, unknown], how="diagonal_relaxed")


def _ferrous_metals(scc: pl.LazyFrame) -> pl.LazyFrame:
    """Ferrous Metals"""
    return (
        scc
        .filter(_SECTOR == "Industrial Processes - Ferrous Metals")
        .with_columns(
            pl.lit("Other").alias("unit_type_lv1"),
            _LV4.alias("unit_type_lv2"),
        )
        .pipe(_with_null_fuel)
    )
