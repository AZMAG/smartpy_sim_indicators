from __future__ import division, print_function

import gc
import numpy as np
import pandas as pd
import orca
from smartpy_core.wrangling import broadcast


def clear_cache(table_name, columns=None):
    """
    Manually clears cached columns. If no column(s)
    are provided then all columns are cleared.

    Parameters:
    ----------
    table_name: string
        Name of orca data frame wrapper to clear.
    columns: string or list, optional
        Name of orca column wrapper(s) to clean.

    """
    if columns is None:
        orca_tab = orca.get_table(table_name)
        columns = list(set(orca_tab.columns).difference(set(orca_tab.local_columns)))

    if isinstance(columns, list):
        for col in columns:
            orca.orca._COLUMNS[(table_name, col)].clear_cached()
    else:
        orca.orca._COLUMNS[(table_name, columns)].clear_cached()


def clear_table(table_name):
    """
    Clears out an entire table cache. Only call this if you want
    the entire table to be recreated. Use the 'clear_cache' method (above) if you
    want to keep the table but just clear additional/computed columns.

    """
    orca.orca._TABLES[table_name].clear_cached()


def clear_injectable(injectable_name):
    """
    Clears out the cache for an injected function and forces it to be
    re-evaluated.

    """
    orca.orca._INJECTABLES[injectable_name].clear_cached()


def get_year_bin(year, year_bins):
    """
    Returns the bin containing the given year. Intended for small lists.

    Parameters:
    -----------
    year: int
        The current simulation year.
    year_bins: list
        List of years.

    Returns:
    --------
    The year bin that contains the provided year.

    """
    year_bins = sorted(year_bins)

    first_bin = year_bins[0]
    if year is None or year <= first_bin:
        return first_bin

    idx = -1
    for y_bin in year_bins:
        if year < y_bin:
            break
        idx += 1

    return year_bins[idx]


############################################
# FUNCTION/INJECTABLE FACTORIES
############################################


def make_broadcast_injectable(from_table, to_table, col_name, fkey,
                              cache=True, cache_scope='iteration'):
    """
    This creates a broadcast column function/injectable and registers it with orca.

    Parameters:
    -----------
    from_table: str
        The table name to brodacast from (the right).
    to_table: str
        The table name to broadcast to (the left).
    col_name: str
        Name of the column to broadcast.
    fkey: str
        Name of column on the to table that serves as the foreign key.
    cache: bool, optional, default True
        Whether or not the broadcast is cached.
    cache_scope: str, optional, default `iteration`
        Cache scope for the broadcast.

    """
    def broadcast_template():

        return broadcast(
            orca.get_table(from_table)[col_name],
            orca.get_table(to_table)[fkey]
        )

    orca.add_column(to_table, col_name, broadcast_template, cache=cache, cache_scope=cache_scope)


def make_reindex_injectable(from_table, to_table, col_name, cache=True, cache_scope='iteration'):
    """
    This creates a PK-PK reindex injectable.

    """
    def reindex_template():
        from_wrap = orca.get_table(from_table)
        to_wrap = orca.get_table(to_table)
        return from_wrap[col_name].reindex(to_wrap.index).fillna(0)

    orca.add_column(to_table, col_name, reindex_template, cache=cache, cache_scope=cache_scope)


def make_series_broadcast_injectable(from_series, to_table, col_name, fkey, fill_with=None,
                                     cache=True, cache_scope='iteration'):
    """
    Broadcasts an injected series to table.

    """
    def s_broadcast_template():
        b = broadcast(
            orca.get_injectable(from_series),
            orca.get_table(to_table)[fkey]
        )
        if fill_with is not None:
            b.fillna(fill_with, inplace=True)
        return b

    orca.add_column(to_table, col_name, s_broadcast_template, cache=cache, cache_scope=cache_scope)


#################################################
# FOR LOADING H5 tables and registering w/ orca
##################################################


def load_tables(h5, year, tables=None):
    """
    Loads tables for the desired year and registers them with orca.

    Parameters:
    -----------
    h5: str
        full path to the h5 file containing the results.
    year: int or str
        Year to grab tables for. Provide 'base' for the base year.
    tables: list of str, default None
        List of tables to load. If None, all tables in that year
        will be loaded.

    """
    with pd.HDFStore(h5, mode='r') as store:

        # grab all the table names in the current year
        if tables is None:
            tables = [t.split('/')[-1] for t in store.keys() if t.startswith('/{}'.format(year))]

        elif not isinstance(tables, list):
            tables = [tables]

        # read in the table and register it with orca
        for t in tables:
            df = df = store['{}/{}'.format(year, t)]
            orca.add_table(t, df)


def list_store_years(h5, table_name=None):
    """
    List the available years in the h5. This assumes tables
    follow the structure: /<year>/<table_name>

    Parameters:
    -----------
    h5: str
        Full path to the h5 fil.
    table_name: str, optional default None
        Specific table to look for.
        If not provided, returns years for any table.

    Returns:
    --------
    list of str

    """
    with pd.HDFStore(h5, mode='r') as s:
        if table_name is None:
            prefixes = [k.split('/')[1] for k in s.keys()]
        else:
             prefixes = [k.split('/')[1] for k in s.keys() if k.endswith('/{}'.format(table_name))]

    return sorted(list(set(prefixes)))


def list_store_tables(h5, year, full=False):
    """
    List the table names available in a given year

    Parameters:
    -----------
    h5: str
        Full path to the h5 fil.
    year: str or in
        The year to look for.
    full: bool, optional default False
        If True, returns full paths, e.g. /2020/households
        If False, returns the base table name, e.g. households

    Returns:
    --------
    list of str

    """
    with pd.HDFStore(h5, mode='r') as s:
        tables = [t for t in s.keys() if t.startswith('/{}'.format(year))]
        if not full:
            tables = [t.split('/')[-1] for t in tables]

    return tables


#################################################
# FOR GENERATING INDICATORS
##################################################


def get_indicators(h5,
                   years,
                   tables,
                   by,
                   agg_func,
                   **agg_kwargs):
    """
    Generates indicators (summary attributes).

    Parameters:
    -----------
    h5: str
        Full path to the h5 file containing the data to summarize.
    years: list
        List of years to process.
    tables: list
        List of tables to load from the h5.
    by: str or list of str:
        Column(s) to aggregate by
    agg_func: func
        The aggration/indicator function to apply. Should accept
        'by' as the input argument.
    **agg_kwargs: kwarg dict
        Additional arguments to pass to the aggregation function.

    Returns:
    --------
    dict of pandas.DataFrame, keyed by year

    """
    base_year = orca.get_injectable('base_year')
    to_concat= {}

    # get summaries for all years
    for y in years:
        print('on year: {}...'.format(y))
        gc.collect()

        # load tables and register w/ orca
        if y == base_year:
            load_tables(sim_h5, 'base', tabs_to_process)
        else:
            load_tables(sim_h5, y, tabs_to_process)

        # get summary results
        to_concat[y] = agg_func(by, **agg_kwargs)

    return to_concat


def compile_to_cols(to_concat, collapse_col_idx=True, collapse_row_idx=True):
    """
    Take a dictionary of data frames and concat colum-wise
    so there is a column for every column/year combination.

    Parameters:
    -----------
    to_concat: dict of pandas.DataFrames
        The data frames to compile
    collapse_col_idx: bool, optional, default True
        If True, combines multi-columns so that the resulting
        columns names are <level2>_<level1>, e.g. pop_2020
    collaspe_row_idx: bool, optional, default True
        If True, and the dataframe has a multi-index, sends the index
        levels to columns and generates a new index.

    Returns:
    --------
    pandas.DataFrame

    """
    c = pd.concat(to_concat, axis=1)

    # collapse multi columns into a single column
    # note this assumes there's only two levels
    if collapse_col_idx:
        z = zip(
                c.columns.get_level_values(0),
                c.columns.get_level_values(1),
            )

        c.columns = ['{}{}{}'.format(y, '_', x) for x, y in z]

    # collapse multi-index rows
    if collapse_row_idx and c.index.nlevels > 1:
        col_names = list(c.index.names)
        c.reset_index(inplace=True)
        c.index
    return c


def compile_to_rows(to_concat, collapse_row_idx=False):
    """
    Take a dictionary of data frames and concat row-wise
    so there is a row for every year, group combination.

    Parameters:
    -----------
    to_concat: dict of pandas.DataFrames
        The data frames to compile
    collaspe_row_idx: bool, optional, default True
        If True, and the dataframe has a multi-index, sends the index
        levels to columns and generates a new index.

    Returns:
    --------
    pandas.DataFrame

    """
    c = pd.concat(to_concat)
    grp_levels = list(c.index.names[1:])
    c.index.names = ['year'] + grp_levels
    c.reset_index(inplace=True)

    if not collapse_row_idx or len(grp_levels) == 1:
        c.set_index(grp_levels, inplace=True)

    return c

