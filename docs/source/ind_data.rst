Sources of Industrial Energy Data for the United States
#######################################################

This page provides a brief overview of sources of U.S. industrial energy data, including brief descriptions of their publication timelines and resolution.
Industrial energy datasets developed by NLR are discussed on a :doc:`separate page </nrel_data>`.
By and large, most industrial energy data is available for "industry" (an aggregation of `North American Industrial Classification System [NAICS] codes <https://www.census.gov/naics/>`_ for Agriculture [NAICS 11], Mining [NAICS 21], Construction [NAICS 23], and Manufacturing [NAICS 31-33]) or for manufacturing industries. Unlike manufacturing, the non-manufacturing sectors do not have their own EIA end-use survey and energy data collection by other federal agencies is limited.
The current state of industrial energy data collection and dissemination can be explained by various historical developments.
This historical context is described in more detail in an `NLR technical report <https://www.nlr.gov/docs/fy24osti/90442.pdf>`_ [#mcmillan2024]_.

.. [#mcmillan2024] McMillan, Colin, Carrie Schoeneberger, Sarang Supekar, and David Thierry. “The Foundational Industrial Energy Dataset (FIED): Open-Source Data on Industrial Facilities.” Golden, CO: National Renewable Energy Laboratory (NREL), September 5, 2024. https://doi.org/10.2172/2440617.

.. csv-table:: Summary of Industrial Energy Data Sources
   :header: "Source", "Data Prodct", "Publication Timeline", "Geographic Resolution", "Industry Resolution", "Fuel Type Disaggregation", "End-Use Disaggregation", "Notes" 

   "EIA",            "`Monthly/Annual Energy Review`_", "Monthly/annually", "National", "None", "Yes", "No", "Comprised of supplier responses from various EIA fuel surveys" 
   "EIA",            "`State Energy Data System (SEDS)`_", "Annual", "None", "Yes", "No", "Comprised of supplier responses to EIA fuel surveys." 
   "EIA",            "`Manufacturing Energy Consumption Survey (MECS)`_", "Quadrennial", "Census region", "Up to 6-digit NAICS", "Yes", "Yes", "Survey conducted by the Census Bureau for EIA."
   "EIA",            "`Annual Energy Outlook (AEO) <https://www.eia.gov/forecasts/aeo/>`_", "Annual", "Census division", "Yes", "Yes", "Yes", "AEO provides annual projections out to 2050. Additional results detail can be requested from EIA staff."
   "EIA",            "`Form EIA-923`_", "Monthly and Annual", "Facility", "Up to 6-digit NAICS", "Electricity only", "No", "Provides information for industrial generators (including cogeneration) above 1 MW capacity."
   "Census Bureau",  "`Economic Census`_", "Quinquennial", "Census Places", "Up to 6-digit NAICS", "Electricity and 'fuels'", "No", "Data are withheld at increasingly geographic and industrial resolution. Use of 'fuels' reported in dollars."
   "Census Bureau",  "`Annual Integrated Economic Survey (AIES)`_", "Annual", "State", "Up to 6-digit NAICS", "Electricity and 'fuels'", "No", "AIES has subsumed the Annual Survey of Manufacturers has been. Use of 'fuels' reported in dollars." 
   "DOE",            "`Combined Heat and Power and Microgrid Database`_", "Rolling updates", "Unit", "Up to 6-digit NAICS", "Yes", "No", 
   "DOE",            "`Industrial Assessment Center (IAC) Database`_", "Rolling updates", "Facility", "Up to 6-digit NAICS", "Yes", "No", "Represents participants in IAC program assessments, which are small and medium sized manufacturers."



Energy Information Administration (EIA)
***************************************

The EIA is the primary institution responsible for collecting national industrial energy data. Its data products include the supplier-based surveys of fuel use, as well as the user-based Manufacturing Energy Consumption Survey conducted by the U.S. Census Breau. 
EIA industrial energy data sources include:

* `Monthly/Annual Energy Review`_: National estimates of industry energy use compiled from `fuel supplier survey forms`_.
* `State Energy Data System (SEDS)`_: State estimates of industry energy use compiles from `fuel supplier survey forms`_.
* `Manufacturing Energy Consumption Survey (MECS)`_: Quadrennial survey of manufacturing energy use. 
* `Form EIA-923`_: Reports information related to industrial facilities that generate their own electricity.

U.S. Department of Energy (DOE)
*******************************

Although EIA is responsible for collecting much of the nation's industrial energy data, the DOE does fund complementary data collection efforts. These include:

* `Combined Heat and Power and Microgrid Database`_: Database of CHP and microgrid installations.
* `Industrial Assessment Center (IAC) Database`_: Database of recorded energy use, production output, production hours, and other information for facilities that have participated in IAC assessments. 


U.S. Census Bureau
******************

Most data collected by the Census Bureau relate to the economic characteristics of industry. The Economic Census does report electricity use in kWh, in addition to fuel purchases (in $). The Economic Census also provides individual tables for materials and fuels consumed by industries.  

* `Economic Census`_: Survey conducted every five years on economic characteristics of industrial establishments.
  * Example 2017 tables (link to zip folder) for materials consumed by `mining <https://www2.census.gov/programs-surveys/economic-census/data/2017/sector21/EC1721MATFUEL.zip>`_ and `manufacturing <https://www2.census.gov/programs-surveys/economic-census/data/2017/sector31/EC1731MATFUEL.zip>`_.
* `Annual Integrated Economic Survey (AIES)`_: Annual survey that includes economic characteristics of manufacturing establishments. 
* `Quarterly Survey of Plant Capacity Utilization (QPC) <https://www.census.gov/programs-surveys/qpc.html>`_: Quarterly survey that provides statistics on the rates of industrial capacity utilization and average weekly operating hours.

U.S. Environmental Protection Agency (EPA)
******************************************

Several EPA datasets provide detailed information about facilities and their individual units. Although these datasets are not explicitly collected to estimated energy use, the emissions they report provide a path for alternative estimates of energy use, as documented in various :doc:`NLR datasets </nrel_data>`.



Other Federal agencies
**********************

Several other federal agencies collect industrial energy information, including the U.S. Geological Survey (USGS) and the U.S. Department of Agriculture (USDA). 


U.S. Geological Survey (USGS)
=============================

The USGS may collect additional industrial energy data, but only the Cement Minerals Yearbook is known to include estimates of energy consumption.

* `Minerals Yearbook: Cement <https://www.usgs.gov/centers/national-minerals-information-center/cement-statistics-and-information>`_: Includes estimates of fuels used by cement district.


U.S. Department of Agriculture (USDA)
=====================================

The USDA collects a limited amount of energy consumption data for farming operations. Known sources are:

* `USDA Economic Research Service Farm Income and Wealth Statistics <https://data.ers.usda.gov/reports.aspx?ID=17842#P474eafd3e12544e19338a00227af3001_2_252iT0R0x17>`_: Electricity and petroleum fuel and oil expenses (in $) by state.
* `USDA National Agricultural Statistics Service <https://quickstats.nass.usda.gov>`_: Expenses from Ag Census (conducted every 5 years) and various surveys. 

The USDA also maintains the `Federal Life Cycle Assessment (LCA) Commons <https://www.lcacommons.gov>`_, a data repository to support LCA.


Other Publicly-Available Data
******************************

* `U.S. industrial boiler inventory <https://github.com/carriescho/Electrification-of-Boilers>`_: Based on 2022 work from `Schoeneberger et al. <https://doi.org/10.1016/j.adapen.2022.100089>`_. 
* `Global Steel Plant Tracker (GSPT) <https://globalenergymonitor.org/projects/global-steel-plant-tracker/>`_: Data compiled by `Global Energy Monitor`_ on global steel facilities.  
* `Global Blast Furnace Tracker (GBFT) <https://globalenergymonitor.org/projects/global-blast-furnace-tracker/>`_: Data compiled by `Global Energy Monitor`_ on global blast furnaces.
* `Global Cement and Concrete Tracker <https://globalenergymonitor.org/projects/global-cement-and-concrete-tracker/>`_: Scheduled release in July 2025 by `Global Energy Monitor`_. 


Private (Proprietary) Sources
*****************************

A mix of industry trade groups and other private sources collect data and may offer access for a fee. Notable examples include the `Portland Cement Association <https://www.cement.org/intelligence-resources/market-intelligence/industry-information/>`_ and the `American Iron and Steel Institute <https://www.steel.org/industry-data/>`_.


.. _Monthly/Annual Energy Review: https://www.eia.gov/totalenergy/data/monthly/
.. _State Energy Data System (SEDS): https://www.eia.gov/state/seds/
.. _Manufacturing Energy Consumption Survey (MECS): https://www.eia.gov/consumption/manufacturing/about.php
.. _Form EIA-923: https://www.eia.gov/electricity/data/eia923
.. _Economic Census: https://www.census.gov/programs-surveys/economic-census.html
.. _Annual Integrated Economic Survey (AIES): https://www.census.gov/programs-surveys/aies.html
.. _Combined Heat and Power and Microgrid Database: https://www.doe.icfwebservices.com/chp
.. _Industrial Assessment Center (IAC) Database: https://iac.university
.. _fuel supplier survey forms: https://www.eia.gov/Survey/index.php
.. _Global Energy Monitor: https://globalenergymonitor.org
