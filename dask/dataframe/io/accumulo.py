from __future__ import print_function, division, absolute_import

from ...compatibility import PY3
from ...delayed import delayed

from thrift.protocol import TCompactProtocol
from thrift.transport import TSocket

from ...accumulo import AccumuloProxy
from ...accumulo.ttypes import *

from .io import from_delayed

import time
import json
import base64
import threading
import numpy
import pandas as pd
import logging

log = logging.getLogger()


class JsonEncode(json.JSONEncoder):
    def default(self, obj):
        # print("ENCODE TYPE: {0}".format(type(obj)))
        # print("ENCODE OBJ: {0}".format(obj))

        if isinstance(obj, numpy.int64) or isinstance(obj, numpy.int32): return int(obj)

        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')

            # data = binascii.b2a_base64(obj).decode('utf-8')
            # print("ENCODED DATA: {0}".format(data))
            # return data

        return json.JSONEncoder.default(self, obj)


class JsonDecode(json.JSONDecoder):

    def default(self, obj):
        '''
        In order to decode binary types, you have to attempt to decode
        them using ascii2binary, then re-encode it, and if they match,
        then it *must* have been binary.  Which is a pain to do for every
        value read from the database, but the only other option would be
        to store the type info with each value, which we could also do.
        '''
        # obj = obj.replace('\\n', '')
        print(f"DECODE OBJ: {obj}")

        data = test = ""

        try:
            data = base64.b64decode(obj)
            ## data = binascii.a2b_base64(obj)
            log.info(f"DECODED: {data}")
        except Exception as e:
            log.info(f"ERROR Decoding OBJ: {e}")
            pass

        try:
            # now re-encode the decoded value
            # test = binascii.b2a_base64(data.encode('utf-8')).decode('utf-8')

            ## test = binascii.b2a_base64(data).decode('utf-8')\
            test = base64.b64encode(data)
            # test = test.replace('\\n', '')
            # test = test.rstrip('\n')

            # test if the data was originally basee64 encoded
            # data2 = json.dumps(obj) #json.JSONDecoder.decode(self, obj)
            print(f"obj: {obj}")
            data2 = json.JSONDecoder.decode(self, obj)
            data2 = data2.replace('\\n', '')
            data2 = data2.rstrip('\n')

            log.debug(f"data2: {data2}")
            log.debug(f"reenc: {test}")

            if test == data2:
                log.debug(f"ACC READ BINARY OBJ: {data[:50]}")
                return data
            else:
                log.debug("Not the same, returning NON-Binary")
        except Exception as e:
            log.info(f"Error testing: {e}")
            pass

        try:
            return json.JSONDecoder.decode(self, obj)
        except Exception as err:
            # print(f"UNABLE TO DECODE OBJ: {obj}, err: {err}")
            return obj


def create_table(config, tid):
    try:
        transport = TSocket.TSocket(config['proxy_host'], config['proxy_port'])
        transport = TTransport.TFramedTransport(transport)
        protocol = TCompactProtocol.TCompactProtocol(transport)
        client = AccumuloProxy.Client(protocol)
        transport.open()

        login = client.login(config['user'], {'password': config['password']})

        if not client.tableExists(login, tid):
            if not client.hasSystemPermission(login, config['user'], SystemPermission.CREATE_TABLE):
                raise Exception("User does not have permission to create tables: {0}".format(config['user']))

            try:
                client.createTable(login, tid, True, TimeType.MILLIS)
                log.info("Creating table: {0}".format(tid))
            except TableExistsException:
                log.info("Table already exists: {0}".format(tid))

        transport.close()
    except Exception as e:
        raise e


def pandas_read_dataframe(config: dict, ranges: tuple, columns: list, meta: dict):
    if config['table'] is None:
        return None

    ret = {}

    # try:
    transport = TSocket.TSocket(config['proxy_host'], config['proxy_port'])
    transport = TTransport.TFramedTransport(transport)
    protocol = TCompactProtocol.TCompactProtocol(transport)
    client = AccumuloProxy.Client(protocol)
    transport.open()

    login = client.login(config['user'], {'password': config['password']})
    auths = client.getUserAuthorizations(login, config['user'])

    # there really has to be something, even if its ('', None), meaning entire table range
    scan_ranges = [] if ranges is not None else None
    if ranges[1] is None:
        r = Range(Key(ranges[0]), True, None, True)
        scan_ranges.append(r)

        log.debug("\t{0}: Row Ranges: {1}-{2}".format(threading.get_ident(), r.start.row, ''))
    else:
        r = Range(Key(ranges[0]), True, Key(ranges[1]), False)
        scan_ranges.append(r)

        log.debug("\t{0}: Row Ranges: {1}-{2}".format(threading.get_ident(), r.start.row, r.stop.row))

    scan_columns = []
    if columns is not None:
        for column in columns:
            c = ScanColumn(column, None)
            scan_columns.append(c)

            log.debug("\t{0}: Scan Column: {1}".format(threading.get_ident(), c))
    else:
        scan_columns = None

    log.debug(f"{scan_columns}, {scan_ranges}")
    options = BatchScanOptions(auths, scan_ranges, scan_columns, [], None)
    cookie = client.createBatchScanner(login, config['table'], options)

    count = 0
    entries = 0
    while client.hasNext(cookie):
        for entry in client.nextK(cookie, 1000).results:
            log.debug("\t{0}: Read row: {1}, {2}, {3}".format(threading.get_ident(), entry.key,
                                                             entry.key.colFamily, entry.value))
            entries += 1

            if entry.key.row not in ret:
                ret[entry.key.row] = {}
                count += 1

            if entry.key.colFamily not in ret[entry.key.row]:
                ret[entry.key.row][entry.key.colFamily] = {}

            ret[entry.key.row][entry.key.colFamily][entry.key.colQualifier] = entry.value
    '''
    except TableNotFoundException:
        log.error("Table not found: {0}".format(config['table']))
        return pd.DataFrame()
    except Exception as err:
        log.error("Accumulo client error: {0}".format(err))
        return pd.DataFrame()
    '''

    log.debug("\t{0}: Read: rows {1}, entries {2}".format(threading.get_ident(), count, entries))

    return pd.DataFrame().from_dict({(i, j): ret[i][j]
                                     for i in ret.keys()
                                     for j in ret[i].keys()}, orient='index')


def pandas_write_dataframe(config: dict, data: pd.DataFrame, meta: dict):
    if config['table'] is None:
        log.info("table name cannot be None")
        raise ValueError("Table name cannot be None")

    tid = config['table']
    count = 0

    log.info("PANDAS DATAFRAME: \n{0}".format(data))

    cf = None
    cq = None

    if 'rows' in meta:
        rows = meta['rows']

        if 'cf' in rows:
            cf = rows['cf']
        if 'cq' in rows:
            cq = rows['cq']

    try:
        # create table if it doesnt exist
        create_table(config, tid)

        transport = TSocket.TSocket(config['proxy_host'], config['proxy_port'])
        transport = TTransport.TFramedTransport(transport)
        protocol = TCompactProtocol.TCompactProtocol(transport)
        client = AccumuloProxy.Client(protocol)
        transport.open()

        opt = WriterOptions(maxMemory=1000000, timeoutMs=1000, threads=2)
        lgn = client.login(config['user'], {'password': config['password']})
        wtr = client.createWriter(lgn, tid, opt)

        cells = {}
        attributes = []

        for rid, row in data.iterrows():
            log.debug("Writing row: {0}".format(rid))

            family = row[cf] if cf is not None else None
            qualifier = row[cq] if cq is not None else None

            for col in data.columns:
                if col == cf or col == cq:
                    continue

                family = col if family is None else family
                qualifier = col if qualifier is None else qualifier

                val = json.dumps(row[col], cls=JsonEncode)

                vis = ''
                if 'vis' in config and config['vis'] is not None:
                    if col in config['vis']:
                        vis = str(config['vis'][col])

                ts = time.time()
                if 'ts' in config and config['ts'] is not None:
                    if col in config['ts']:
                        ts = config['ts'][col]

                log.info("Writing kv to {0}: ({1} {2}:{3}, {4}), vis={5}".format(tid, rid, family, qualifier, val, vis))
                attributes.append(ColumnUpdate(colFamily=str(family),
                                               colQualifier=str(qualifier),
                                               colVisibility=str(vis),
                                               timestamp=int(ts),
                                               value=str(val)))

                ### Cant just jam these into cells by rid.. need to use rid + cf + cq
                
            # cells[str(rid)] = attributes
            cells.append(attributes)
            attributes = []

            count = count + 1

        client.updateAndFlush(lgn, tid, cells)
        client.closeWriter(wtr)
        transport.close()
    except TableNotFoundException:
        log.error("Table not found: {0}".format(tid))
    except Exception as err:
        log.error("Accumulo client error: {0}".format(err))

    return count


def get_splits(config: dict, max_splits=10):
    if config['table'] is None:
        return None

    transport = TSocket.TSocket(config['proxy_host'], config['proxy_port'])
    transport = TTransport.TFramedTransport(transport)
    protocol = TCompactProtocol.TCompactProtocol(transport)
    client = AccumuloProxy.Client(protocol)
    transport.open()

    login = client.login(config['user'], {'password': config['password']})

    return client.listSplits(login, config['table'], max_splits)


def read_accumulo(table: str, config: dict, npartitions=16,
                  compute=True, columns=None, meta=None):
    """
    Create dataframe from an NoSQL table.

    If neither divisions or npartitions is given, the memory footprint of the
    first few rows will be determined, and partitions of size ~256MB will
    be used.

    Parameters
    ----------
    table : string
        Table to load into dataframe
    config : dict
        Accumulo connection configuration
    npartitions : int
        Number of partitions; only when Compute True
    compute : boolean
        Return type; list of dask.Dataframe if False; dask.Dataframe otherwise
        (using from_delayed)
    columns : list of strings or None
        Which column families to select; if None, gets all; e.g.,
        ``sql.func.abs(sql.column('value')).label('abs(value)')``.
        Labeling columns created by functions or arithmetic operations is
        recommended.
    meta : empty DataFrame or None
        If provided, do not attempt to infer dtypes, but use these, coercing
        all chunks on load

    Returns
    -------
    dask.dataframe

    Examples
    --------
    >>> df = dd.read_accumulo('accounts', my_acc_config,
    ...                  npartitions=10)  # doctest: +SKIP
    """
    if 'user' not in config or \
            'password' not in config or \
            'proxy_host' not in config or \
            'proxy_port' not in config or \
            'zk_str' not in config:
        raise ValueError("Accumulo config not completely defined: {0}".format(config))

    config['table'] = table
    config['cf'] = columns

    splits = get_splits(config, max_splits=npartitions)
    log.debug("# Splits: {0}".format(len(splits)))

    splits.insert(0, '')
    splits.append(None)

    parts = []
    end = len(splits) - 1
    for i in range(end):
        ranges = (splits[i], splits[i + 1])
        log.debug("Range: {0}".format(ranges))

        parts.append(delayed(pandas_read_dataframe)(config, ranges=ranges, columns=columns, meta=meta))

    if compute:
        return from_delayed(parts).repartition(npartitions=npartitions)
    else:
        return parts


def to_accumulo(df, config: dict, meta: dict, compute=True, scheduler=None):
    """
    Paralell write of Dask DataFrame to Accumulo Table

    Parameters
    ----------
    df : Dataframe
        The dask.Dataframe to write to Accumulo
    config : dict
        Accumulo configuration to use to connect to accumulo
    meta : dict
        Data model to apply to dataframe
    compute : bool
        Should compute be called; immediately call write if True, delayed otherwise
    scheduler : str
        The scheduler to use, like “threads” or “processes”

    Returns
    -------
    The number of Accumulo rows written if they were computed right away.
    If not, the delayed tasks associated with the writing of the table
    """
    dfs = df.to_delayed()
    values = [delayed(pandas_write_dataframe)(config, d, meta) for d in dfs]

    if compute:
        return sum(delayed(values).compute(scheduler=scheduler))
    else:
        return values


if PY3:
    from ..core import _Frame

    _Frame.to_accumulo.__doc__ = to_accumulo.__doc__
