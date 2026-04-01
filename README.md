# Foundational Industry Energy Dataset (FIED)

**FIED** is a dataset that compiles and estimates unit-level information for
industrial energy analysis and modeling.

## Summary

This is an effort by the National Laboratory of the Rockies (NLR) and Argonne National Laboratory (ANL) to create an experimental foundational industry dataset for energy and emissions analysis and modeling. The code draws from various publicly-available data, primarily from the U.S. EPA, to compile a data set on unit-level energy use and characterization for U.S. industrial facilities in 2017 and 2020.

The FIED, and the accompanying technical report, can be downloaded from its [Open Energy Data Initiative submission](https://doi.org/10.25984/2437657).

## Getting Started

### Required Data

All the required data are downloaded and managed automatically, which includes
a local cache so each dataset is downloaded only once and stored locally.
An older version of this procedure required a manual download of each dataset, but that is not required anymore.

### Installation

To prepare your environment to compile the FIED, please follow the instructions in [INSTALL.md](./INSTALL.md).

## Compiling the FIED

Once the FIED Python package is installed, you will have access to a command
that will download all the necessary data and process it to create the FIED dataset.

For instance, from the terminal, you can run:

```bash
fied --vintage=2020
```

That will orchestrate the full data pipeline, including accessing the extenal
public data. We currently support the vintages 2017 and 2020.

## Directory Navigation

The underlying submodules and data are organized as follows:

* [analysis](/analysis/): Methods for analyzing and generating figures of the final dataset.
* [data](/data/): Most folders are created locally for organizing raw data. Contains a [directory list](/data/dir_structure.md).
* [energy](/energy/): Not currently used. For future estimation of facility energy use based on alternative approaches. 
* [frs](/frs): Methods for downloading and formatting EPA Facility Registry Service data.
* [geocoder](/geocoder/): Methods for collecting missing geographical information for facilities.
* [ghgrp](/ghgrp/): Methods for estimating energy use from GHG emissions reported under EPA's Greenhouse Gas Reporting Program. Based on previous projects, such as the [Industry Energy Data Book](https://github.com/NREL/Industry-energy-data-book).
* [nei](/nei/): Methods for downloading and formatting data from EPA's National Emissions Inventory and for using these data to characterize combustion units.
* [qpc](/qpc/): Methods for downloading and formatting operating hours reported under the Census Bureau's Quaterly Survey of Plant Capacity Utlization.
* [scc](/scc/):  Methods to download and apply EPA's Source Classification Codes for characterizing units.
* [tests](/tests/): Testing. Currently very limited.
* [tools](/tools/): Methods that act as various tools used across submodules.

## Overivew of FIED Data Fields

Data fields are compiled and described in `FIED_datafields.yml`. All facilities in the data set are represented by their unique `registryID`, which is their EPA [Facility Registry Service ID](https://www.epa.gov/frs/frs-physical-data-model).

Many of these data fields were included in original EPA data sources. See the [FRS data dictionary](https://www.epa.gov/frs/frs-data-dictionary) for more information.

### Identity

In addition to `registryID`, other identifying fields include

* `eisFacilityID`: EPA ID assigned to facilities reporting to the Emissions Inventory System (EIS).
* `ghgrpID`: EPA ID assigned to facilities reporting under the Greenhouse Gas Reporting Program (GHGRP).
* `name`: Name of facility.
* `locationDescription`: Description of the facility location.
* `naicsCode`: The facility's North American Industrial Classification System (NAICS) code.
* `naicsCodeAdditional`: A facility may have additional NAICS codes assigned (e.g., different reporting systems may have different NAICS assigned).

### Geography

Various levels of geographic identifiers are included, such as

* `geoID`: see [Census description of geographic identifiers (GEOIDs)](https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html)
* `latitude`: facility latitude.
* `longitude`: facility longitude.
* `postalCode`: facility U.S. ZIP code.
* `countyNAME`: name of facility county.
* `countyFIPS`: facility Federal Information Processing Series (FIPS) code.
* `stateName`: name of facility state.
* `legislativeDisctrictNumber`: facility [congressional district number](https://www.census.gov/programs-surveys/geography/guidance/geo-areas/congressional-dist.html).

### Units and Processes

Individual units are characterized (e.g., unit type, capacity, energy, throughput) where possible. Individual units may be associated with multiple processes.

* `designCapacity`: design capacity of unit.
* `eisUnitID`: U.S. EPA Emissions Inventory System (EIS) unit ID.
* `unitName`: unit name.
* `unitType`: reported or inferred unit type.
* `unitTypeStd`: standardized unitType.
* `processDescription`: description of process. Processes may have more than one unit associated with them.
* `eisProcessID`: U.S. EPA Emissions Inventory System (EIS) process ID. Processes may have more than one unit associated with them.

### Energy

Depending on the estimation approach, a unit may have a single estimate of energy use, or a range of energy estimates (i.e., minimum, median, upper quartile). Energy estimates based on the NEI are presented as a range.

* `energyMJ`: energy estimate in MJ
* `energyMJ0`: minimum of energy estimate, in MJ.
* `energyMJq2`: median of energy estimate, in MJ.
* `energyMJq3`: upper quartile of energy estimate, in MJ.
* `fuelType`: combusted fuel type as reported by original data source.
* `fuelTypeStd`: combusted fuel type, standardized.
* `energyEstimateSource`: source of underlying data used to make energy estimate. Some energy values are provided directly by GHGRP data.

### Other

We've attempted to include additional descriptive fields where possible. These tend to be sparsely populated at this time.

* `hucCode8`: Hydrolic Unit Code. Not currently implemented.
* `weeklyOpHours`: Average weekly operating hours by quarter, including 95% confidence interval ranges.
* `sensitiveInd`: Indicates whether or not the associated data is enforcement sensitive.
* `smallBusInd`: Code indicating whether or not a business is requesting relief under EPA’s Small Business Policy, which applies to businesses having less than 100 employees.
* `througputTonne`: Estimated mass throughput.
