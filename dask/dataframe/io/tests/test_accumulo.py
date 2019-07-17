from __future__ import print_function, division, absolute_import

import os
from io import BytesIO

import pytest

pd = pytest.importorskip('pandas')
dd = pytest.importorskip('dask.dataframe')

from toolz import partition_all

import pandas.util.testing as tm

import dask
import dask.dataframe as dd
from dask.base import compute_as_if_collection
from dask.dataframe.io.accumulo import (read_accumulo, to_accumulo)
from dask.dataframe.utils import assert_eq, has_known_categories
from dask.utils import filetexts, filetext, tmpfile, tmpdir
from dask.bytes.compression import files as cfiles, seekable_files

import random
import string
import numpy as np

config = {
    'proxy_host': 'zoo1',
    'proxy_port': 42424,
    'instance': "LucdAI",
    'user': "cc_user",
    'password': "D33pinsight",
    'zk_str': "zoo1:5181,zoo2:5181,zoo3:5181",
    'vis': {
        'A': 'vis_1',
        'B': 'vis_1',
        'C': 'vis_2',
        'D': 'vis_2',
        'E': 'vis_3'
    }
}

sm = 64
lg = 1024

cf = list('RST')
cq = list('XYZ')
sm_columns = list('ABCDE')

arrays = [[f"cf_{i}" for i in range(0, sm)],
          [f"cq_{i}" for i in range(0, 4)]]

arrays = [[f"{i}" for i in range(0, sm)],
          [f"{i}" for i in cf],
          [f"{i}" for i in cq]]

index = pd.MultiIndex.from_product(arrays, names=['row_ids', 'families', 'qualifiers'])
small = pd.DataFrame(np.random.randint(0, sm, size=(len(index), 5)), columns=sm_columns, index=index)

# Dask from_pandas will not load pandas DF with multiIndex
small = small.reset_index(level=[1, 2])

'''
small1 = pd.DataFrame({'col1': [range(0, sm)],
                      'col2': [range(0, sm)],
                      'col3': [range(0, sm)],
                      'col4': [range(0, sm)],
                      'col5': [''.join([random.choice(string.ascii_letters + string.digits) for n in range(sm)]) for _ in range(0, sm)]
                      })

large = pd.DataFrame({'col1': range(1, lg),
                      'col2': range(1, lg),
                      'col3': range(1, lg),
                      'col4': range(1, lg),
                      'col5': [''.join([random.choice(string.ascii_letters + string.digits) for n in range(lg)]) for _ in range(1, lg)]
                      })
'''

################################
#  to_accumulo with test data  #
################################


to_acc_and_table = pytest.mark.parametrize('tid,cfg,df',
                                           [('test_table_small', config, dd.from_pandas(small, 4))])


@to_acc_and_table
def test_to_accumulo(tid, cfg, df):
    config['table'] = tid

    meta = {
        'rows': {
            'cf': 'families',
            'cq': 'qualifiers'
        }
    }

    rows = df.to_accumulo(config=cfg, meta=meta)

    assert rows == len(df.index)

###################################
#  read_accumulo using test data  #
###################################

'''
acc_and_table = pytest.mark.parametrize('tid,cfg',
                                        [('test_table_small', config)])


@acc_and_table
def test_pandas_read_accumulo(tid, cfg):
    df = read_accumulo(tid, cfg)

    print("\n")
    print(f"{df.head(20)}")
    print(f"Index: {df.index}")

    # 16 partitions is the default
    assert df.npartitions == 16
    assert type(df) == dask.dataframe.core.DataFrame
    assert list(df.columns) == sm_columns
'''

'''
@acc_and_table
def test_pandas_read_accumulo_without_compute(tid, cfg):
    df = read_accumulo(tid, cfg, npartitions=2, compute=False)

    assert type(df) == list


@acc_and_table
def test_pandas_read_accumulo_withcolumn_families(tid, cfg):
    columns = ['col1']
    df = read_accumulo(tid, cfg, columns=columns)

    assert type(df) == list
'''

'''
@acc_and_table
def test_pandas_read_rows_dtype_coercion(reader, files):
    b = files['test_table1']
    df = pandas_read_rows(reader, b, b'', {}, {'amount': 'float'})
    assert df.amount.dtype == 'float'


@acc_and_table
def test_rows_to_pandas_simple(reader, files):
    blocks = [[files[k]] for k in sorted(files)]
    kwargs = {}
    head = pandas_read_rows(reader, files['test_table1'], b'', {})

    df = rows_to_pandas(reader, blocks, head, kwargs, collection=True)
    assert isinstance(df, dd.DataFrame)
    assert list(df.columns) == ['name', 'amount', 'id']

    values = rows_to_pandas(reader, blocks, head, kwargs, collection=False)
    assert isinstance(values, list)
    assert len(values) == 3
    assert all(hasattr(item, 'dask') for item in values)

    assert_eq(df.amount.sum(),
              100 + 200 + 300 + 400 + 500 + 600)


@acc_and_table
def test_rows_to_pandas_kwargs(reader, files):
    blocks = [files[k] for k in sorted(files)]
    blocks = [[b] for b in blocks]
    kwargs = {'usecols': ['name', 'id']}
    head = pandas_read_rows(reader, files['test_table1'], b'', kwargs)

    df = rows_to_pandas(reader, blocks, head, kwargs, collection=True)
    assert list(df.columns) == ['name', 'id']
    result = df.compute()
    assert (result.columns == df.columns).all()


@acc_and_table
def test_rows_to_pandas_blocked(reader, files):
    blocks = []
    for k in sorted(files):
        b = files[k]
        lines = b.split(b'\n')
        blocks.append([b'\n'.join(bs) for bs in partition_all(2, lines)])

    df = rows_to_pandas(reader, blocks, expected.head(), {})
    assert_eq(df.compute().reset_index(drop=True),
              expected.reset_index(drop=True), check_dtype=False)

    expected2 = expected[['name', 'id']]
    df = rows_to_pandas(reader, blocks, expected2.head(),
                        {'usecols': ['name', 'id']})
    assert_eq(df.compute().reset_index(drop=True),
              expected2.reset_index(drop=True), check_dtype=False)


@pytest.mark.parametrize('dd_read,acc_read,files',
                         [(dd.read_accumulo, acc.read_table, test_tables)])
@read_table_mark
def test_skiprows(dd_read, acc_read, files):
    files = {name: content for name, content in files.items()}
    skip = 0
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', skiprows=skip)
        expected_df = pd.concat([acc_read(n, skiprows=skip) for n in sorted(files)])
        assert_eq(df, expected_df, check_dtype=False)


@pytest.mark.parametrize('dd_read,acc_read,files,units',
                         [(dd.read_accumulo, acc.read_table, test_tables, csv_units_row)])
@read_table_mark
def test_skiprows_as_list(dd_read, acc_read, files, units):
    files = {name: (content.replace(b'\n', b'\n' + units, 1)) for name, content in files.items()}
    skip = [0, 1, 2, 3, 5]
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', skiprows=skip)
        expected_df = pd.concat([acc_read(n, skiprows=skip) for n in sorted(files)])
        assert_eq(df, expected_df, check_dtype=False)


##############
#  read_csv  #
##############


@pytest.mark.parametrize('dd_read,acc_read,text,sep',
                         [(dd.read_accumulo, acc.read_table, table_text , ',')])
@read_table_mark
def test_read_accumulo(dd_read, acc_read, text, sep):
    with filetext(text) as fn:
        f = dd_read(fn, blocksize=30, lineterminator=os.linesep, sep=sep)
        assert list(f.columns) == ['name', 'amount']
        # index may be different
        result = f.compute(scheduler='sync').reset_index(drop=True)
        assert_eq(result, acc_read(fn, sep=sep))


@pytest.mark.parametrize('dd_read,acc_read,text,skip',
                         [(dd.read_accumulo, acc.read_table, table_text , 7)])
@read_table_mark
def test_read_accumulo_large_skiprows(dd_read, acc_read, text, skip):
    names = ['name', 'amount']
    with filetext(text) as fn:
        actual = dd_read(fn, skiprows=skip, names=names)
        assert_eq(actual, acc_read(fn, skiprows=skip, names=names))


@pytest.mark.parametrize('dd_read,acc_read,text,skip',
                         [(dd.read_accumulo, acc.read_table, table_text , 7)])
@read_table_mark
def test_read_accumulo_skiprows_only_in_first_partition(dd_read, acc_read, text, skip):
    names = ['name', 'amount']
    with filetext(text) as fn:
        with pytest.warns(UserWarning, match='sample=blocksize'):
            actual = dd_read(fn, blocksize=200, skiprows=skip, names=names).compute()
            assert_eq(actual, acc_read(fn, skiprows=skip, names=names))

        with pytest.warns(UserWarning):
            # if new sample does not contain all the skiprows, raise error
            with pytest.raises(ValueError):
                dd_read(fn, blocksize=30, skiprows=skip, names=names)


@pytest.mark.parametrize('dd_read,acc_read,files',
                         [(dd.read_accumulo, acc.read_table, test_tables)])
@read_table_mark
def test_read_csv_files(dd_read, acc_read, files):
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv')
        assert_eq(df, expected, check_dtype=False)

        fn = 'test_table1'
        df = dd_read(fn)
        expected2 = acc_read(BytesIO(files[fn]))
        assert_eq(df, expected2, check_dtype=False)


@pytest.mark.parametrize('dd_read,acc_read,files',
                         [(dd.read_accumulo, acc.read_table, test_tables)])
@read_table_mark
def test_read_csv_files_list(dd_read, acc_read, files):
    with filetexts(files, mode='b'):
        subset = sorted(files)[:2]  # Just first 2
        sol = pd.concat([acc_read(BytesIO(files[k])) for k in subset])
        res = dd_read(subset)
        assert_eq(res, sol, check_dtype=False)

        with pytest.raises(ValueError):
            dd_read([])


@pytest.mark.parametrize('dd_read,files',
                         [(dd.read_accumulo, test_tables)])
@read_table_mark
def test_read_accumulo_include_tablename_column_with_duplicate_name(dd_read, files):
    with filetexts(files, mode='b'):
        with pytest.raises(ValueError):
            dd_read('2014-01-*.csv', include_tablename_column='name')


@pytest.mark.parametrize('dd_read,files',
                         [(dd.read_accumulo, test_tables)])
@read_table_mark
def test_read_accumulo_include_tablename_column_is_dtype_category(dd_read, files):
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', include_tablename_column=True)
        assert df.path.dtype == 'category'
        assert has_known_categories(df.path)

        dfs = dd_read('2014-01-*.csv', include_tablename_column=True, collection=False)
        result = dfs[0].compute()
        assert result.path.dtype == 'category'
        assert has_known_categories(result.path)


# After this point, we test just using read_csv, as all functionality
# for both is implemented using the same code.


def test_read_accumulo_index():
    with filetext(table_text) as tn:
        f = dd.read_accumulo(tn, instance, user, password, zk_str,  blocksize=20).set_index('amount')
        result = f.compute(scheduler='sync')
        assert result.index.name == 'amount'

        blocks = compute_as_if_collection(dd.DataFrame, f.dask,
                                          f.__dask_keys__(),
                                          scheduler='sync')
        for i, block in enumerate(blocks):
            if i < len(f.divisions) - 2:
                assert (block.index < f.divisions[i + 1]).all()
            if i > 0:
                assert (block.index >= f.divisions[i]).all()

        expected = acc.read_table(tn).set_index('amount')
        assert_eq(result, expected)


def test_read_accumulo_skiprows_range():
    with filetext(table_text ) as tn:
        f = dd.read_accumulo(tn, instance, user, password, zk_str,  skiprows=range(5))
        result = f
        expected = acc.read_table(tn, skiprows=range(5))
        assert_eq(result, expected)


def test_usecols():
    with filetext(timeseries) as tn:
        df = dd.read_accumulo(tn, instance, user, password, zk_str, usecols=['High', 'Low'])
        expected = acc.read_table(tn, usecols=['High', 'Low'])
        assert (df.compute().values == expected.values).all()


def test_skipinitialspace():
    text = normalize_text("""
    name, amount
    Alice,100
    Bob,-200
    Charlie,300
    Dennis,400
    Edith,-500
    Frank,600
    """)

    with filetext(text) as tn:
        df = dd.read_accumulo(tn, instance, user, password, zk_str,  skipinitialspace=True, blocksize=20)

        assert 'amount' in df.columns
        assert df.amount.max().compute() == 600


def test_consistent_dtypes():
    text = normalize_text("""
    name,amount
    Alice,100.5
    Bob,-200.5
    Charlie,300
    Dennis,400
    Edith,-500
    Frank,600
    """)

    with filetext(text) as tn:
        df = dd.read_accumulo(tn, instance, user, password, zk_str,  blocksize=30)
        assert df.amount.compute().dtype == float


def test_consistent_dtypes_2():
    text1 = normalize_text("""
    name,amount
    Alice,100
    Bob,-200
    Charlie,300
    """)

    text2 = normalize_text("""
    name,amount
    1,400
    2,-500
    Frank,600
    """)

    with filetexts({'foo.1.csv': text1, 'foo.2.csv': text2}):
        df = dd.read_accumulo('foo.*.csv', blocksize=25)
        assert df.name.dtype == object
        assert df.name.compute().dtype == object


def test_categorical_dtypes():
    text1 = normalize_text("""
    fruit,count
    apple,10
    apple,25
    pear,100
    orange,15
    """)

    text2 = normalize_text("""
    fruit,count
    apple,200
    banana,300
    orange,400
    banana,10
    """)

    with filetexts({'foo.1.csv': text1, 'foo.2.csv': text2}):
        df = dd.read_accumulo('foo.*.csv', dtype={'fruit': 'category'}, blocksize=25)
        assert df.fruit.dtype == 'category'
        assert not has_known_categories(df.fruit)
        res = df.compute()
        assert res.fruit.dtype == 'category'
        assert (sorted(res.fruit.cat.categories) ==
                ['apple', 'banana', 'orange', 'pear'])


def test_categorical_known():
    text1 = normalize_text("""
    A,B
    a,a
    b,b
    a,a
    """)
    text2 = normalize_text("""
    A,B
    a,a
    b,b
    c,c
    """)
    dtype = pd.api.types.CategoricalDtype(['a', 'b', 'c'])
    with filetexts({"foo.1.csv": text1, "foo.2.csv": text2}):
        result = dd.read_accumulo("foo.*.csv", dtype={"A": 'category',
                                                      "B": 'category'})
        assert result.A.cat.known is False
        assert result.B.cat.known is False
        expected = pd.DataFrame({
            "A": pd.Categorical(['a', 'b', 'a', 'a', 'b', 'c'],
                                categories=dtype.categories),
            "B": pd.Categorical(['a', 'b', 'a', 'a', 'b', 'c'],
                                categories=dtype.categories)},
            index=[0, 1, 2, 0, 1, 2])
        assert_eq(result, expected)

        # Specify a dtype
        result = dd.read_accumulo("foo.*.csv", dtype={'A': dtype, 'B': 'category'})
        assert result.A.cat.known is True
        assert result.B.cat.known is False
        tm.assert_index_equal(result.A.cat.categories, dtype.categories)
        assert result.A.cat.ordered is False
        assert_eq(result, expected)

        # ordered
        dtype = pd.api.types.CategoricalDtype(['a', 'b', 'c'], ordered=True)
        result = dd.read_accumulo("foo.*.csv", dtype={'A': dtype, 'B': 'category'})
        expected['A'] = expected['A'].cat.as_ordered()
        assert result.A.cat.known is True
        assert result.B.cat.known is False
        assert result.A.cat.ordered is True

        assert_eq(result, expected)

        # Specify "unknown" categories
        result = dd.read_accumulo("foo.*.csv",
                                  dtype=pd.api.types.CategoricalDtype())
        assert result.A.cat.known is False

        result = dd.read_accumulo("foo.*.csv", dtype="category")
        assert result.A.cat.known is False


def test_empty_table():
    with filetext('a,b') as tn:
        df = dd.read_accumulo(tn, instance, user, password, zk_str,  header=0)
        assert len(df.compute()) == 0
        assert list(df.columns) == ['a', 'b']


def test_read_accumulo_no_sample():
    with filetexts(test_tables, mode='b') as tn:
        df = dd.read_accumulo(tn, instance, user, password, zk_str,  sample=False)
        assert list(df.columns) == ['name', 'amount', 'id']


def test_read_accumulo_sensitive_to_enforce():
    with filetexts(test_tables, mode='b'):
        a = dd.read_accumulo('2014-01-*.csv', enforce=True)
        b = dd.read_accumulo('2014-01-*.csv', enforce=False)
        assert a._name != b._name


def test_accumulo_with_integer_names():
    with filetext('alice,1\nbob,2') as tn:
        df = dd.read_accumulo(tn, instance, user, password, zk_str,  header=None)
        assert list(df.columns) == [0, 1]


def test_late_dtypes():
    text = 'numbers,names,more_numbers,integers,dates\n'
    for i in range(1000):
        text += '1,,2,3,2017-10-31 00:00:00\n'
    text += '1.5,bar,2.5,3,4998-01-01 00:00:00\n'

    date_msg = ("\n"
                "\n"
                "-------------------------------------------------------------\n"
                "\n"
                "The following columns also failed to properly parse as dates:\n"
                "\n"
                "- dates\n"
                "\n"
                "This is usually due to an invalid value in that column. To\n"
                "diagnose and fix it's recommended to drop these columns from the\n"
                "`parse_dates` keyword, and manually convert them to dates later\n"
                "using `dd.to_datetime`.")

    with filetext(text) as tn:
        sol = acc.read_table(tn)
        msg = ("Mismatched dtypes found in `acc.read_table`/`pd.read_table`.\n"
               "\n"
               "+--------------+---------+----------+\n"
               "| Column       | Found   | Expected |\n"
               "+--------------+---------+----------+\n"
               "| more_numbers | float64 | int64    |\n"
               "| names        | object  | float64  |\n"
               "| numbers      | float64 | int64    |\n"
               "+--------------+---------+----------+\n"
               "\n"
               "- names\n"
               "  ValueError(.*)\n"
               "\n"
               "Usually this is due to dask's dtype inference failing, and\n"
               "*may* be fixed by specifying dtypes manually by adding:\n"
               "\n"
               "dtype={'more_numbers': 'float64',\n"
               "       'names': 'object',\n"
               "       'numbers': 'float64'}\n"
               "\n"
               "to the call to `read_csv`/`read_table`.")

        with pytest.raises(ValueError) as e:
            dd.read_accumulo(tn, instance, user, password, zk_str,  sample=50,
                             parse_dates=['dates']).compute(scheduler='sync')
        assert e.match(msg + date_msg)

        with pytest.raises(ValueError) as e:
            dd.read_accumulo(tn, instance, user, password, zk_str,  sample=50).compute(scheduler='sync')
        assert e.match(msg)

        msg = ("Mismatched dtypes found in `acc.read_table`/`pd.read_table`.\n"
               "\n"
               "+--------------+---------+----------+\n"
               "| Column       | Found   | Expected |\n"
               "+--------------+---------+----------+\n"
               "| more_numbers | float64 | int64    |\n"
               "| numbers      | float64 | int64    |\n"
               "+--------------+---------+----------+\n"
               "\n"
               "Usually this is due to dask's dtype inference failing, and\n"
               "*may* be fixed by specifying dtypes manually by adding:\n"
               "\n"
               "dtype={'more_numbers': 'float64',\n"
               "       'numbers': 'float64'}\n"
               "\n"
               "to the call to `read_csv`/`read_table`.\n"
               "\n"
               "Alternatively, provide `assume_missing=True` to interpret\n"
               "all unspecified integer columns as floats.")

        with pytest.raises(ValueError) as e:
            dd.read_accumulo(tn, instance, user, password, zk_str,  sample=50,
                             dtype={'names': 'O'}).compute(scheduler='sync')
        assert str(e.value) == msg

        with pytest.raises(ValueError) as e:
            dd.read_accumulo(tn, instance, user, password, zk_str,  sample=50, parse_dates=['dates'],
                             dtype={'names': 'O'}).compute(scheduler='sync')
        assert str(e.value) == msg + date_msg

        msg = ("Mismatched dtypes found in `acc.read_table`/`pd.read_table`.\n"
               "\n"
               "The following columns failed to properly parse as dates:\n"
               "\n"
               "- dates\n"
               "\n"
               "This is usually due to an invalid value in that column. To\n"
               "diagnose and fix it's recommended to drop these columns from the\n"
               "`parse_dates` keyword, and manually convert them to dates later\n"
               "using `dd.to_datetime`.")

        with pytest.raises(ValueError) as e:
            dd.read_accumulo(tn, instance, user, password, zk_str,  sample=50, parse_dates=['dates'],
                             dtype={'more_numbers': float, 'names': object,
                                    'numbers': float}).compute(scheduler='sync')
        assert str(e.value) == msg

        # Specifying dtypes works
        res = dd.read_accumulo(tn, instance, user, password, zk_str,  sample=50,
                               dtype={'more_numbers': float, 'names': object,
                                      'numbers': float})
        assert_eq(res, sol)


def test_assume_missing():
    text = 'numbers,names,more_numbers,integers\n'
    for i in range(1000):
        text += '1,foo,2,3\n'
    text += '1.5,bar,2.5,3\n'
    with filetext(text) as tn:
        sol = acc.read_table(tn)

        # assume_missing affects all columns
        res = dd.read_accumulo(tn, instance, user, password, zk_str,  sample=50, assume_missing=True)
        assert_eq(res, sol.astype({'integers': float}))

        # assume_missing doesn't override specified dtypes
        res = dd.read_accumulo(tn, instance, user, password, zk_str,  sample=50, assume_missing=True,
                               dtype={'integers': 'int64'})
        assert_eq(res, sol)

        # assume_missing works with dtype=None
        res = dd.read_accumulo(tn, instance, user, password, zk_str,  sample=50, assume_missing=True, dtype=None)
        assert_eq(res, sol.astype({'integers': float}))

    text = 'numbers,integers\n'
    for i in range(1000):
        text += '1,2\n'
    text += '1.5,2\n'

    with filetext(text) as tn:
        sol = acc.read_table(tn)

        # assume_missing ignored when all dtypes specifed
        df = dd.read_accumulo(tn, instance, user, password, zk_str,  sample=30, dtype='int64', assume_missing=True)
        assert df.numbers.dtype == 'int64'


def test_index_col():
    with filetext(table_text ) as tn:
        try:
            dd.read_accumulo(tn, instance, user, password, zk_str,  blocksize=30, index_col='name')
            assert False
        except ValueError as e:
            assert 'set_index' in str(e)


def test_read_accumulo_with_datetime_index_partitions_one():
    with filetext(timeseries) as tn:
        df = acc.read_table(tn, index_col=0, header=0, usecols=[0, 4],
                            parse_dates=['Date'])
        # blocksize set to explicitly set to single chunk
        ddf = dd.read_accumulo(tn, instance, user, password, zk_str,  header=0, usecols=[0, 4],
                               parse_dates=['Date'],
                               blocksize=10000000).set_index('Date')
        assert_eq(df, ddf)

        # because fn is so small, by default, this will only be one chunk
        ddf = dd.read_accumulo(tn, instance, user, password, zk_str,  header=0, usecols=[0, 4],
                               parse_dates=['Date']).set_index('Date')
        assert_eq(df, ddf)


def test_read_accumulo_with_datetime_index_partitions_n():
    with filetext(timeseries) as tn:
        df = acc.read_table(tn, index_col=0, header=0, usecols=[0, 4], parse_dates=['Date'])
        # because fn is so small, by default, set chunksize small
        ddf = dd.read_accumulo(tn, instance, user, password, zk_str,  header=0, usecols=[0, 4],
                               parse_dates=['Date'],
                               blocksize=400).set_index('Date')
        assert_eq(df, ddf)


def test_none_usecols():
    with filetext(table_text ) as tn:
        df = dd.read_accumulo(tn, instance, user, password, zk_str,  usecols=None)
        assert_eq(df, acc.read_table(tn, usecols=None))


def test_parse_dates_multi_column():
    pdmc_text = normalize_text("""
    ID,date,time
    10,2003-11-04,180036
    11,2003-11-05,125640
    12,2003-11-01,2519
    13,2003-10-22,142559
    14,2003-10-24,163113
    15,2003-10-20,170133
    16,2003-11-11,160448
    17,2003-11-03,171759
    18,2003-11-07,190928
    19,2003-10-21,84623
    20,2003-10-25,192207
    21,2003-11-13,180156
    22,2003-11-15,131037
    """)

    with filetext(pdmc_text) as tn:
        ddf = dd.read_accumulo(tn, instance, user, password, zk_str,  parse_dates=[['date', 'time']])
        df = acc.read_table(tn, parse_dates=[['date', 'time']])

        assert (df.columns == ddf.columns).all()
        assert len(df) == len(ddf)


def test_read_accumulo_singleton_dtype():
    data = b'a,b\n1,2\n3,4\n5,6'
    with filetext(data, mode='wb') as tn:
        assert_eq(acc.read_table(tn, dtype=float),
                  dd.read_accumulo(tn, instance, user, password, zk_str,  dtype=float))


def test_robust_column_mismatch():
    files = test_tables.copy()
    k = sorted(files)[-1]
    files[k] = files[k].replace(b'name', b'Name')
    with filetexts(files, mode='b'):
        ddf = dd.read_accumulo('2014-01-*.csv')
        df = acc.read_table('test_table1')
        assert (df.columns == ddf.columns).all()
        assert_eq(ddf, ddf)


def test_error_if_sample_is_too_small():
    text = ('AAAAA,BBBBB,CCCCC,DDDDD,EEEEE\n'
            '1,2,3,4,5\n'
            '6,7,8,9,10\n'
            '11,12,13,14,15')
    with filetext(text) as tn:
        # Sample size stops mid header row
        sample = 20
        with pytest.raises(ValueError):
            dd.read_accumulo(tn, instance, user, password, zk_str,  sample=sample)

        # Saying no header means this is fine
        assert_eq(dd.read_accumulo(tn, instance, user, password, zk_str,  sample=sample, header=None),
                  acc.read_table(tn, header=None))

    skiptext = ('# skip\n'
                '# these\n'
                '# lines\n')

    text = skiptext + text
    with filetext(text) as tn:
        # Sample size stops mid header row
        sample = 20 + len(skiptext)
        with pytest.raises(ValueError):
            dd.read_accumulo(tn, instance, user, password, zk_str,  sample=sample, skiprows=3)

        # Saying no header means this is fine
        assert_eq(dd.read_accumulo(tn, instance, user, password, zk_str,  sample=sample, header=None, skiprows=3),
                  acc.read_table(tn, header=None, skiprows=3))


def test_read_accumulo_names_not_none():
    text = ('Alice,100\n'
            'Bob,-200\n'
            'Charlie,300\n'
            'Dennis,400\n'
            'Edith,-500\n'
            'Frank,600\n')
    names = ['name', 'amount']
    with filetext(text) as tn:
        ddf = dd.read_accumulo(tn, instance, user, password, zk_str,  names=names, blocksize=16)
        df = acc.read_table(tn, names=names)
        assert_eq(df, ddf, check_index=False)


############
#  to_accumulo  #
############

def test_to_accumulo():
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'],
                       'y': [1, 2, 3, 4]})

    for npartitions in [1, 2]:
        a = dd.from_pandas(df, npartitions)
        with tmpdir() as dn:
            a.to_accumulo(dn, index=False)
            result = dd.read_accumulo(os.path.join(dn, '*')).compute().reset_index(drop=True)
            assert_eq(result, df)

        with tmpdir() as dn:
            r = a.to_accumulo(dn, index=False, compute=False)
            dask.compute(*r, scheduler='sync')
            result = dd.read_accumulo(os.path.join(dn, '*')).compute().reset_index(drop=True)
            assert_eq(result, df)

        with tmpdir() as dn:
            fn = os.path.join(dn, 'data_*.csv')
            a.to_accumulo(fn, index=False)
            result = dd.read_accumulo(fn).compute().reset_index(drop=True)
            assert_eq(result, df)


def test_to_accumulo_multiple_files_cornercases():
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'],
                       'y': [1, 2, 3, 4]})
    a = dd.from_pandas(df, 2)
    with tmpdir() as dn:
        with pytest.raises(ValueError):
            fn = os.path.join(dn, "data_*_*.csv")
            a.to_accumulo(fn)

    df16 = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                               'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'],
                         'y': [1, 2, 3, 4, 5, 6, 7, 8, 9,
                               10, 11, 12, 13, 14, 15, 16]})
    a = dd.from_pandas(df16, 16)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data_*.csv')
        a.to_accumulo(fn, index=False)
        result = dd.read_accumulo(fn).compute().reset_index(drop=True)
        assert_eq(result, df16)

    # test handling existing files when links are optimized out
    a = dd.from_pandas(df, 2)
    with tmpdir() as dn:
        a.to_accumulo(dn, index=False)
        fn = os.path.join(dn, 'data_*.csv')
        a.to_accumulo(fn, mode='w', index=False)
        result = dd.read_accumulo(fn).compute().reset_index(drop=True)
        assert_eq(result, df)

    # test handling existing files when links are optimized out
    a = dd.from_pandas(df16, 16)
    with tmpdir() as dn:
        a.to_accumulo(dn, index=False)
        fn = os.path.join(dn, 'data_*.csv')
        a.to_accumulo(fn, mode='w', index=False)
        result = dd.read_accumulo(fn).compute().reset_index(drop=True)
        assert_eq(result, df16)


@pytest.mark.xfail(reason="to_accumulo does not support compression")
def test_to_accumulo_gzip():
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'],
                       'y': [1, 2, 3, 4]}, index=[1., 2., 3., 4.])

    for npartitions in [1, 2]:
        a = dd.from_pandas(df, npartitions)
        with tmpfile('csv') as fn:
            a.to_accumulo(fn, compression='gzip')
            result = acc.read_table(fn, index_col=0, compression='gzip')
            tm.assert_frame_equal(result, df)


def test_to_accumulo_simple():
    df0 = pd.DataFrame({'x': ['a', 'b', 'c', 'd'],
                        'y': [1, 2, 3, 4]}, index=[1., 2., 3., 4.])
    df = dd.from_pandas(df0, npartitions=2)
    with tmpdir() as dir:
        dir = str(dir)
        df.to_accumulo(dir)
        assert os.listdir(dir)
        result = dd.read_accumulo(os.path.join(dir, '*')).compute()
    assert (result.x.values == df0.x.values).all()


def test_to_accumulo_series():
    df0 = pd.Series(['a', 'b', 'c', 'd'], index=[1., 2., 3., 4.])
    df = dd.from_pandas(df0, npartitions=2)
    with tmpdir() as dir:
        dir = str(dir)
        df.to_accumulo(dir, header=False)
        assert os.listdir(dir)
        result = dd.read_accumulo(os.path.join(dir, '*'), header=None,
                                  names=['x']).compute()
    assert (result.x == df0).all()


def test_to_accumulo_with_get():
    from dask.multiprocessing import get as mp_get
    flag = [False]

    def my_get(*args, **kwargs):
        flag[0] = True
        return mp_get(*args, **kwargs)

    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'],
                       'y': [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=2)

    with tmpdir() as dn:
        ddf.to_accumulo(dn, index=False, scheduler=my_get)
        assert flag[0]
        result = dd.read_accumulo(os.path.join(dn, '*')).compute().reset_index(drop=True)
        assert_eq(result, df)


def test_to_accumulo_paths():
    df = pd.DataFrame({"A": range(10)})
    ddf = dd.from_pandas(df, npartitions=2)
    assert ddf.to_accumulo("foo*.csv") == ['foo0.csv', 'foo1.csv']
    os.remove('foo0.csv')
    os.remove('foo1.csv')


@pytest.mark.parametrize("header, expected", [(False, ""), (True, "x,y\n")])
def test_to_accumulo_header_empty_dataframe(header, expected):
    dfe = pd.DataFrame({'x': [],
                        'y': []})
    ddfe = dd.from_pandas(dfe, npartitions=1)

    with tmpdir() as dn:
        ddfe.to_accumulo(os.path.join(dn, "fooe*.csv"), index=False, header=header)
        assert not os.path.exists(os.path.join(dn, "fooe1.csv"))
        filename = os.path.join(dn, 'fooe0.csv')
        with open(filename, 'r') as fp:
            line = fp.readline()
            assert line == expected
        os.remove(filename)


@pytest.mark.parametrize("header,header_first_partition_only,expected_first,expected_next",
                         [
                             (False, False, "a,1\n", "d,4\n"),
                             (True, False, "x,y\n", "x,y\n"),
                             (False, True, "a,1\n", "d,4\n"),
                             (True, True, "x,y\n", "d,4\n"),
                             (['aa', 'bb'], False, "aa,bb\n", "aa,bb\n"),
                             (['aa', 'bb'], True, "aa,bb\n", "d,4\n")])
def test_to_accumulo_header(header, header_first_partition_only, expected_first, expected_next):
    partition_count = 2
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e', 'f'],
                       'y': [1, 2, 3, 4, 5, 6]})
    ddf = dd.from_pandas(df, npartitions=partition_count)

    with tmpdir() as dn:
        # Test NO header case
        # (header=False, header_first_chunk_only not passed)
        ddf.to_accumulo(os.path.join(dn, "fooa*.csv"), index=False, header=header,
                        header_first_partition_only=header_first_partition_only)
        filename = os.path.join(dn, 'fooa0.csv')
        with open(filename, 'r') as fp:
            line = fp.readline()
            assert line == expected_first
        os.remove(filename)

        filename = os.path.join(dn, 'fooa1.csv')
        with open(filename, 'r') as fp:
            line = fp.readline()
            assert line == expected_next
        os.remove(filename)
'''
