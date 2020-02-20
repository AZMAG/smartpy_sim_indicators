"""
Variable (computed column definitions) for
various tables.

"""

from __future__ import print_function, division

import numpy as np
import pandas as pd
import orca

from smartpy_core.wrangling import broadcast, fill_nulls, categorize
from .framework import make_broadcast_injectable


##########
# SETUP
##########


# the assumed base year
orca.add_injectable('base_year', 2018)


# the assumed end year
orca.add_injectable('end_year', 2055)


@orca.injectable()
def year(base_year):
    """
    The current year. This will be the base unless called within
    the context of a run.

    """
    if 'iter_var' in orca.list_injectables():
        year = orca.get_injectable('iter_var')
        if year is not None:
            return year

    return base_year


############
# PARCELS
#############


@orca.column('parcels')
def maz(parcels):
    return parcels['maz_2019']


@orca.column('parcels')
def taz(parcels):
    return parcels['taz_2019']


@orca.column('parcels')
def raz(parcels):
    return parcels['raz_2019']


@orca.column('parcels')
def mpa(parcels):
    return parcels['mpa_2019']


@orca.column('parcels')
def mpa_fullname(parcels):
    return parcels['mpa_fullname_2019']


@orca.column('parcels')
def county(parcels):
    return parcels['county_2019']


@orca.column('parcels')
def ewc_pinal(parcels):
    return parcels['ewc_pinal_2019']


@orca.column('parcels', cache=True, cache_scope='iteration')
def bldg_sqft(parcels, buildings):
    """
    Total built square feet per parcel.

    """
    b = buildings.to_frame(['parcel_id', 'total_sqft'])
    return b.groupby('parcel_id')['total_sqft'].sum().reindex(parcels.index).fillna(0)


@orca.column('parcels', cache=True, cache_scope='iteration')
def bldg_far(parcels, buildings):
    """
    Total build floor-area-ratio for the parcel.
    """
    return fill_nulls(parcels['bldg_sqft'] / parcels['area'])


@orca.column('parcels', cache=True, cache_scope='iteration')
def posths_enrollment(parcels, posths):
    """
    Post high school enrollment.

    """
    p = posths.local
    return p.groupby(
        'parcel_id')['enrollment'].sum().reindex(parcels.index).fillna(0)


@orca.column('parcels', cache=True, cache_scope='iteration')
def k6_enrollment(parcels, k12):
    """
    Kinder through 6th grade enrollment (elementary).

    """
    k = k12.local
    return k.groupby(
        'parcel_id')['K6'].sum().reindex(parcels.index).fillna(0)


@orca.column('parcels', cache=True, cache_scope='iteration')
def g7_8_enrollment(parcels, k12):
    """
    Enrollment for grades 7-8 (middle).

    """
    k = k12.local
    return k.groupby(
        'parcel_id')['G7_8'].sum().reindex(parcels.index).fillna(0)


@orca.column('parcels', cache=True, cache_scope='iteration')
def g9_12_enrollment(parcels, k12):
    """
    Enrollment for grades 9-12 (high school).

    """
    k = k12.local
    return k.groupby(
        'parcel_id')['G9_12'].sum().reindex(parcels.index).fillna(0)


@orca.column('parcels', cache=True, cache_scope='iteration')
def k12_enrollment(parcels):
    """
    Enrollment for grades kinder through 12th.

    """
    return parcels['k6_enrollment'] + parcels['g7_8_enrollment'] + parcels['g9_12_enrollment']


@orca.column('parcels', cache=True, cache_scope='iteration')
def enrollment_all(parcels):
    """
    Enrollment for all grades

    """
    return parcels['k12_enrollment'] + parcels['posths_enrollment']


@orca.column('parcels', cache=True)
def is_MC(parcels):
    """
    Dummy for Maricopa County.

    """
    return (parcels.county == 'MC').astype(int)


@orca.column('parcels')
def is_PC(parcels):
    """
    Dummy for Pinal County.

    """
    return (parcels.county == 'PC').astype(int)


@orca.column('parcels', cache=True)
def is_tribe(parcels):
    """
    Dummy for tribal areas.

    """
    tribal_mpas = ['AK', 'FM', 'GR', 'SA', 'SN', 'TC']
    return parcels['mpa'].isin(tribal_mpas).astype(int)


@orca.column('parcels', cache=True)
def east_valley(parcels):
    """
    Dummy for presence in East Valley.

    """
    in_ev = parcels['mpa'].isin([
        'AJ', 'CA', 'CC', 'CH', 'FH', 'FM', 'GC', 'GI',
        'GU', 'ME', 'PA', 'QC', 'SA', 'SC', 'TE'
    ])
    return (parcels['is_MC'] & in_ev).astype(int)


@orca.column('parcels', cache=True)
def west_valley(parcels):
    """
    Dummy for presence in West Valley.

    """
    in_wv = parcels['mpa'].isin([
        'AV', 'BU', 'EL', 'GB', 'GL', 'GO', 'LP', 'PE', 'SU', 'TO', 'WI', 'YO'
    ])
    return (parcels['is_MC'] & in_wv).astype(int)


@orca.column('parcels')
def freeway_dist(year, parcels):
    """
    Year dependent freeway distance.

    """
    if year <= 2024:
        return parcels['fwys_2019_dist']
    elif year <= 2030:
        return parcels['fwys_2030_dist']
    else:
        return parcels['fwys_2031_dist']


# make all parcel columns available
# Note: this is ugly, but right now I'm hard-coding these so we don't have to
# load the table first
parcel_broadcast_cols = [
    'city',
    'hex_id',
    'section_id',
    'bg_geoid',
    'cbd_dist',
    'airport_dist',
    'x',
    'y',
    'area',
    'taz_2019',
    'mpa_2019',
    'in_mag_2019',
    'county_2019',
    'mpa_fullname_2019',
    'ewc_pinal_2019',
    'raz_2019',
    'maz_2019',
    'maz',
    'taz',
    'raz',
    'mpa',
    'mpa_fullname',
    'county',
    'ewc_pinal',
    'bldg_sqft',
    'bldg_far',
    'posths_enrollment',
    'k6_enrollment',
    'g7_8_enrollment',
    'g9_12_enrollment',
    'k12_enrollment',
    'enrollment_all',
    'is_MC',
    'is_PC',
    'is_tribe',
    'east_valley',
    'west_valley',
    'freeway_dist'
]


for par_col in parcel_broadcast_cols:
    for curr_tab in ['buildings', 'households', 'persons', 'jobs',
                     'seasonal_households', 'gq_persons', 'flu_space',
                     'k12', 'posths']:
        make_broadcast_injectable('parcels', curr_tab, par_col, 'parcel_id')


##############
# BUILDINGS
##############


@orca.column('buildings', 'building_age', cache=True)
def building_age(buildings, year):
    return year - buildings['year_built']


@orca.column('buildings', cache=True, cache_scope='forever')
def sqft_per_res_unit(buildings):
    return fill_nulls(
        buildings['residential_sqft'] / buildings['residential_units'], 0)


@orca.column('buildings', cache=True, cache_scope='iteration')
def res_hh(buildings, households):
    hh_sums = households.local.groupby('building_id').size()
    return hh_sums.reindex(buildings.index).fillna(0)


@orca.column('buildings', cache=True, cache_scope='iteration')
def seas_hh(buildings, seasonal_households):
    seas_sums = seasonal_households.local.groupby('building_id').size()
    return seas_sums.reindex(buildings.index).fillna(0)


@orca.column('buildings', cache=True, cache_scope='iteration')
def total_hh(buildings):
    return buildings['res_hh'] + buildings['seas_hh']


@orca.column('buildings', cache=True, cache_scope='iteration')
def vac_res_units(buildings):
    return buildings['residential_units'] - buildings['total_hh']


@orca.column('buildings', cache=True, cache_scope='iteration')
def site_based_jobs(buildings, jobs):
    sb = jobs.local.query("job_class == 'site based'")
    return sb.groupby('building_id').size().reindex(buildings.index).fillna(0)


@orca.column('buildings', cache=True, cache_scope='iteration')
def job_spaces(buildings):
    est_job_spaces = np.round(fill_nulls(
        buildings['non_residential_sqft'] / buildings['sqft_per_job'], 0))
    return pd.DataFrame([est_job_spaces, buildings['site_based_jobs']]).max()


@orca.column('buildings', cache=True, cache_scope='iteration')
def vac_job_spaces(buildings):
    return buildings['job_spaces'] - buildings['site_based_jobs']


@orca.column('buildings', cache=True, cache_scope='iteration')
def total_sqft(buildings):
    return buildings['residential_sqft'].fillna(0) + buildings['non_residential_sqft'].fillna(0)


# broadcast building variables
bldg_broadcast_cols = ['building_type_name', 'residential_sqft', 'residential_units',
                       'total_fcv', 'year_built', 'parcel_id']

for bldg_col in bldg_broadcast_cols:
        for curr_tab in ['households', 'persons', 'jobs', 'seasonal_households', 'gq_persons']:
            make_broadcast_injectable('buildings', curr_tab, bldg_col, 'building_id')


##############
# HOUSEHOLDS
##############


@orca.column('households', cache=True, cache_scope='iteration')
def income_quintile(households):
    """
    Household income quintile at the MSA level.

    """
    return pd.Series(
        pd.qcut(
            households['income'], 5, [1, 2, 3, 4, 5]),
        index=households.index
    )


@orca.column('households', cache=True, cache_scope='iteration')
def income_category(households):
    """
    Define 3 houshold income categories for HLCM submodels

    """
    inc = fill_nulls(households['income'], 0)
    brks = [np.nan, 29999, 99999, np.nan]
    labels = ['income group ' + str(x) for x in range(1, len(brks))]
    with np.errstate(invalid='ignore'):
        # for some reason, the cut method is now throwing a warning about nulls
        # for now, suppress this as the results look fine
        # todo: look into this further
        c = categorize(inc, brks, labels)
    return c


@orca.column('households', cache=True, cache_scope='iteration')
def income1(households):
    """
    Dummy for for income group 1

    """
    return (households['income_category'] == 'income group 1').astype(int)


@orca.column('households', cache=True, cache_scope='iteration')
def income2(households):
    """
    Dummy for for income group 2

    """
    return (households['income_category'] == 'income group 2').astype(int)


@orca.column('households', cache=True, cache_scope='iteration')
def income3(households):
    """
    Dummy for for income group 3

    """
    return (households['income_category'] == 'income group 3').astype(int)


@orca.column('households', cache=True, cache_scope='iteration')
def workers(persons, households):
    """
    Number of workers in the household.

    """
    return persons.local.groupby(
        'household_id')['is_worker'].sum().reindex(households.index).fillna(0)


@orca.column('households', cache=True, cache_scope='iteration')
def children(persons, households):
    """
    Indicates the presence of children in the household (0, 1).

    """
    min_age = persons.local.groupby(
        'household_id')['age'].min().reindex(households.index).fillna(0)
    return (min_age < 18).astype(int)


# broadcast household variables
hh_broadcast_cols = ['building_id', 'income', 'owns', 'income_quintile', 'year_added']
for hh_col in hh_broadcast_cols:

        make_broadcast_injectable('households', 'persons', hh_col, 'household_id')


###############
# PERSONS
###############


@orca.column('persons')
def is_adult(persons):
    return (persons['age'] > 17).astype(int)


@orca.column('persons')
def is_kid(persons):
    return (persons['age'] <= 17).astype(int)


#############
# JOBS
#############


@orca.injectable(cache=True, cache_scope='forever')
def job_model_sectors():
    """
    Define a series of naics by job sector.

    Generalizes the 2 digit naics into the sectors we will be
    modeling with. Note: this is being applied to all rows,
    but will only be applicable for site-based jobs.

    """
    return pd.Series({
        '11': 'agriculture',
        '21': 'mining',
        '22': 'utilities',
        '23': 'construction',
        '31': 'manufacturing',
        '32': 'manufacturing',
        '33': 'manufacturing',
        '42': 'warehouse and transport',
        '44': 'retail',
        '45': 'retail',
        '48': 'warehouse and transport',
        '49': 'warehouse and transport',
        '51': 'office',
        '52': 'office',
        '53': 'office',
        '54': 'office',
        '55': 'office',
        '56': 'office',
        '61': 'education',
        '62': 'medical',
        '71': 'hotel arts ent',
        '721': 'hotel arts ent',
        '722': 'retail',
        '81': 'office',
        '92': 'public',
        '93': 'public'
    })


@orca.column('jobs')
def model_sector(jobs, job_model_sectors):
    return broadcast(job_model_sectors, j['mag_naics'])


@orca.injectable(cache=True, cache_scope='forever')
def qcew_naics_mapping():
    """
    Maps mag naics to the QCEW naics as is used for the controls, e.g. 31-33

    """
    return pd.Series({
        '11': '11',
        '21': '21',
        '22': '22',
        '23': '23',
        '31': '31-33',
        '32': '31-33',
        '33': '31-33',
        '42': '42',
        '44': '44-45',
        '45': '44-45',
        '48': '48-49',
        '49': '48-49',
        '51': '51',
        '52': '52',
        '53': '53',
        '54': '54',
        '55': '55',
        '56': '56',
        '61': '61',
        '62': '62',
        '71': '71',
        '721': '721',
        '722': '722',
        '81': '81',
        '92': '92',
        '93': '93'
    })


@orca.column('jobs')
def qcew_naics(jobs, qcew_naics_mapping):
    return broadcast(qcew_naics_mapping, j['mag_naics'])
