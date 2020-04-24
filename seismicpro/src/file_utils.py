""" Utility functions for files """
import re
import segyio
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..batchflow import FilesIndex
from .seismic_index import SegyFilesIndex

def write_segy_file(data, df, samples, path, sorting=None, segy_format=1):
    """Write data and headers into SEGY file.

    Parameters
    ----------
    data : array-like
        Array of traces.
    df : DataFrame
        DataFrame with trace headers data.
    samples : array, same length as traces
        Time samples for trace data.
    path : str
        Path to output file.
    sorting : int
        SEGY file sorting.
    format : int
        SEGY file format.

    Returns
    -------
    """
    spec = segyio.spec()
    spec.sorting = sorting
    spec.format = segy_format
    spec.samples = samples
    spec.tracecount = len(data)

    df.columns = [getattr(segyio.TraceField, k) for k in df.columns]
    df[getattr(segyio.TraceField, 'TRACE_SEQUENCE_FILE')] = np.arange(len(df)) + 1

    with segyio.create(path, spec) as file:
        file.trace = data
        meta = df.to_dict('index')
        for i, x in enumerate(file.header[:]):
            x.update(meta[i])

def merge_segy_files(output_path, bar=True, **kwargs):
    """Merge segy files into a single segy file.

    Parameters
    ----------
    output_path : str
        Path to output file.
    bar : bool
        Whether to how progress bar (default = True).
    kwargs : dict
        Keyword arguments to index input segy files.

    Returns
    -------
    """
    segy_index = SegyFilesIndex(**kwargs, name='data')
    spec = segyio.spec()
    spec.sorting = None
    spec.format = 1
    spec.tracecount = sum(segy_index.tracecounts)
    with segyio.open(segy_index.indices[0], strict=False) as file:
        spec.samples = file.samples

    with segyio.create(output_path, spec) as dst:
        i = 0
        iterable = tqdm(segy_index.indices) if bar else segy_index.indices
        for index in iterable:
            with segyio.open(index, strict=False) as src:
                dst.trace[i: i + src.tracecount] = src.trace
                dst.header[i: i + src.tracecount] = src.header
                for j in range(src.tracecount):
                    dst.header[i + j].update({segyio.TraceField.TRACE_SEQUENCE_FILE: i + j + 1})

            i += src.tracecount

def merge_picking_files(output_path, **kwargs):
    """Merge picking files into a single file.

    Parameters
    ----------
    output_path : str
        Path to output file.
    kwargs : dict
        Keyword arguments to index input files.

    Returns
    -------
    """
    files_index = FilesIndex(**kwargs)
    dfs = []
    for i in files_index.indices:
        path = files_index.get_fullpath(i)
        dfs.append(pd.read_csv(path))

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output_path, index=False)

def load_horizon(horizon):
    """Load horizons from file. This file should contain columns with inline, crossline
    and horizon's time for current inline and crossline.
    Columns order:   |  INLINE  |   CROSSLINE   | ...another columns... | horizon time  |
    Note: that file shouldn't contain columns' name!

    Parameters
    ----------
    horizon : str
        path to horizon

    Returns
    -------
        : pd.DataFrame with 3 columns: INLINE_3D, CROSSLINE_3D, time.
    """
    horizon_val = []
    with open(horizon) as f:
        text = f.read()
        lines = text.split('\n')
        for line in lines:
            line = re.sub(' +', ' ', line.strip())
            if line == '':
                continue
            line = line.split(' ')
            horizon_val.append([int(line[0]), int(line[1]), float(line[-1])])
        f.close()
    return pd.DataFrame(horizon_val, columns=['INLINE_3D', 'CROSSLINE_3D', 'time'])
